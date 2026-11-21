#include "rcia_energy_trace_ros/tracking_session.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using rcia_energy_trace_ros::FramePair;
using rcia_energy_trace_ros::RosFrameBuffer;
using rcia_energy_trace_ros::SessionHooks;
using rcia_energy_trace_ros::SessionParams;
using rcia_energy_trace_ros::SessionPhase;
using rcia_energy_trace_ros::SessionProgress;
using rcia_energy_trace_ros::SessionResult;
using rcia_energy_trace_ros::TrackingSession;
using rcia_energy_trace_ros::ValidateSessionParams;

constexpr std::int64_t kMs = 1000000;
const std::vector<std::int64_t> kScenarioMs = {0, 33, 66, 99, 132, 165};

void SetStamp(builtin_interfaces::msg::Time *stamp, std::int64_t stamp_ns) {
  stamp->sec = static_cast<std::int32_t>(stamp_ns / 1000000000LL);
  stamp->nanosec = static_cast<std::uint32_t>(stamp_ns % 1000000000LL);
}

sensor_msgs::msg::Image::ConstSharedPtr MakeColor(std::int64_t stamp_ns,
                                                  std::uint8_t value) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  SetStamp(&msg->header.stamp, stamp_ns);
  msg->header.frame_id = "camera_color_optical_frame";
  msg->height = 4;
  msg->width = 4;
  msg->encoding = "bgr8";
  msg->step = 12;
  msg->data.assign(4 * 12, value);
  return msg;
}

sensor_msgs::msg::Image::ConstSharedPtr MakeDepth(std::int64_t stamp_ns,
                                                  std::uint16_t value) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  SetStamp(&msg->header.stamp, stamp_ns);
  msg->header.frame_id = "camera_color_optical_frame";
  msg->height = 4;
  msg->width = 4;
  msg->encoding = "16UC1";
  msg->step = 8;
  msg->data.resize(4 * 8);
  for (std::size_t i = 0; i < msg->data.size(); i += 2) {
    std::memcpy(msg->data.data() + i, &value, sizeof(value));
  }
  return msg;
}

sensor_msgs::msg::CameraInfo MakeInfo() {
  sensor_msgs::msg::CameraInfo info;
  info.width = 4;
  info.height = 4;
  info.k = {100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0};
  return info;
}

std::shared_ptr<RosFrameBuffer> MakeBuffer() {
  auto buffer = std::make_shared<RosFrameBuffer>();
  buffer->SetColorCameraInfo(MakeInfo());
  buffer->SetDepthCameraInfo(MakeInfo());
  return buffer;
}

// Pushes color and depth frames with identical stamps, so every color frame pairs.
void PushPairs(RosFrameBuffer *buffer, const std::vector<std::int64_t> &stamps_ms) {
  std::uint8_t value = 1;
  for (const auto stamp_ms : stamps_ms) {
    buffer->SetColorImage(MakeColor(stamp_ms * kMs, value++));
    buffer->SetDepthImage(MakeDepth(stamp_ms * kMs, 1000));
  }
}

std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Records what the session does. Hooks run on the thread that calls Run(), so
// the harness needs no locking. The ROS clock is fake and set by the test.
struct Harness {
  std::vector<SessionPhase> phases;
  std::vector<SessionPhase> publish_phases;
  std::vector<int> iterations;
  std::vector<std::int64_t> published_ns;
  int setup_calls = 0;
  int start_calls = 0;
  std::int64_t ros_ns = 10000 * kMs;
  bool pose_ok = true;
  bool track_ok = true;
  std::function<void(int)> on_track = [](int) {};
  std::function<void()> on_publish = [] {};

  SessionHooks MakeHooks() {
    SessionHooks hooks;
    hooks.setup = [this] {
      ++setup_calls;
      return true;
    };
    hooks.start = [this] {
      ++start_calls;
      return true;
    };
    hooks.track = [this](int iteration) {
      iterations.push_back(iteration);
      on_track(iteration);
      return track_ok;
    };
    hooks.pose_valid = [this] { return pose_ok; };
    hooks.publish_live = [this](const FramePair &pair) {
      published_ns.push_back(pair.color_stamp_ns);
      publish_phases.push_back(phases.back());
      on_publish();
    };
    hooks.report_phase = [this](SessionPhase phase, const SessionProgress &) {
      if (phases.empty() || phases.back() != phase) phases.push_back(phase);
    };
    hooks.ros_now_ns = [this] { return ros_ns; };
    hooks.monotonic_now_ns = [] { return SteadyNowNs(); };
    return hooks;
  }

  // Iteration i is pair i in kScenarioMs. Setting the ROS clock just after that
  // pair keeps the session within the catch-up lag and the pose age limit.
  void TrackScenarioClock() {
    on_track = [this](int iteration) { ros_ns = (33 * iteration + 10) * kMs; };
  }
};

}  // namespace

TEST(TrackingSession, T0RunsSetupStartAndTrackOnceBeforeFirstPublish) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), kScenarioMs);

  Harness harness;
  harness.TrackScenarioClock();
  TrackingSession *session_ptr = nullptr;
  harness.on_publish = [&session_ptr] { session_ptr->Cancel(); };

  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  session_ptr = &session;
  const auto result = session.Run(0);

  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(result.reason, "cancelled");
  EXPECT_EQ(harness.setup_calls, 1);
  EXPECT_EQ(harness.start_calls, 1);
  ASSERT_FALSE(harness.iterations.empty());
  EXPECT_EQ(harness.iterations.front(), 0);
  EXPECT_EQ(std::count(harness.iterations.begin(), harness.iterations.end(), 0), 1);
  ASSERT_FALSE(harness.published_ns.empty());
  EXPECT_GT(harness.published_ns.front(), 0);
}

TEST(TrackingSession, CatchUpReplaysWithoutPublishThenTracksWithoutRestart) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), kScenarioMs);

  Harness harness;
  harness.TrackScenarioClock();
  TrackingSession *session_ptr = nullptr;
  harness.on_publish = [&session_ptr] { session_ptr->Cancel(); };

  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  session_ptr = &session;
  const auto result = session.Run(0);

  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(harness.phases,
            (std::vector<SessionPhase>{SessionPhase::Starting, SessionPhase::CatchingUp,
                                       SessionPhase::Tracking}));
  EXPECT_EQ(harness.setup_calls, 1);
  EXPECT_EQ(harness.start_calls, 1);
  EXPECT_EQ(result.progress.replayed_frames, 3u);
  ASSERT_FALSE(harness.published_ns.empty());
  EXPECT_GE(harness.published_ns.front(), 132 * kMs);
  for (const auto phase : harness.publish_phases) {
    EXPECT_EQ(phase, SessionPhase::Tracking);
  }
}

TEST(TrackingSession, ReplayTimeoutWhenCatchUpNeverFinishes) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), kScenarioMs);

  Harness harness;
  harness.ros_ns = 10000 * kMs;
  SessionParams params;
  params.replay_timeout_ms = 50.0;

  TrackingSession session{buffer, params, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_FALSE(result.cancelled);
  EXPECT_EQ(result.reason, "replay_timeout");
  EXPECT_TRUE(harness.published_ns.empty());
  EXPECT_EQ(harness.phases.back(), SessionPhase::CatchingUp);
}

TEST(TrackingSession, LiveCameraStallFailsWithCameraStalled) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), kScenarioMs);

  Harness harness;
  harness.TrackScenarioClock();
  SessionParams params;
  params.live_frame_wait_ms = 50.0;

  TrackingSession session{buffer, params, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_FALSE(result.cancelled);
  EXPECT_EQ(result.reason, "camera_stalled");
  ASSERT_FALSE(harness.published_ns.empty());
  for (const auto stamp : harness.published_ns) {
    EXPECT_GT(stamp, 99 * kMs);
  }
}

TEST(TrackingSession, RosClockRegressionFailsWithTrackerError) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), {0, 33, 66});

  Harness harness;
  harness.on_track = [&harness](int iteration) {
    harness.ros_ns = (iteration == 0 ? 1000 : 500) * kMs;
  };

  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_FALSE(result.cancelled);
  EXPECT_EQ(result.reason, "tracker_error");
  EXPECT_TRUE(harness.published_ns.empty());
}

TEST(TrackingSession, ColorGapBeforeCatchUpFailsWithFrameGap) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), {0, 120});

  Harness harness;
  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_EQ(result.reason, "frame_gap");
  EXPECT_TRUE(harness.published_ns.empty());
}

TEST(TrackingSession, StartPairOutsideHistoryFailsWithHistoryExpired) {
  auto buffer = MakeBuffer();
  buffer->SetColorImage(MakeColor(0, 1));
  buffer->SetColorImage(MakeColor(300 * kMs, 2));

  Harness harness;
  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  const auto result = session.Run(100 * kMs);

  EXPECT_EQ(result.reason, "history_expired");
  EXPECT_EQ(harness.setup_calls, 0);
  EXPECT_EQ(harness.start_calls, 0);
}

TEST(TrackingSession, InvalidInitialPoseFailsWithInvalidPose) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), {0, 33});

  Harness harness;
  harness.pose_ok = false;
  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_EQ(result.reason, "invalid_pose");
  EXPECT_TRUE(harness.published_ns.empty());
}

TEST(TrackingSession, TrackerFailureOnT0FailsWithTrackerError) {
  auto buffer = MakeBuffer();
  PushPairs(buffer.get(), {0, 33});

  Harness harness;
  harness.track_ok = false;
  TrackingSession session{buffer, SessionParams{}, harness.MakeHooks()};
  const auto result = session.Run(0);

  EXPECT_EQ(result.reason, "tracker_error");
  EXPECT_TRUE(harness.published_ns.empty());
}

TEST(TrackingSession, CancelWakesWaitingStartAndReturnsPromptly) {
  auto buffer = MakeBuffer();

  Harness harness;
  SessionParams params;
  params.start_frame_wait_ms = 10000.0;
  TrackingSession session{buffer, params, harness.MakeHooks()};

  std::thread canceller([&session] {
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    session.Cancel();
  });
  const auto started = std::chrono::steady_clock::now();
  const auto result = session.Run(0);
  canceller.join();

  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds{1});
  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(result.reason, "cancelled");
  EXPECT_TRUE(session.cancelled());
}

TEST(TrackingSession, ValidateSessionParamsRejectsInvalidValues) {
  EXPECT_NO_THROW(ValidateSessionParams(SessionParams{}));

  SessionParams params;
  params.replay_target_lag_ms = 250.0;
  EXPECT_THROW(ValidateSessionParams(params), std::invalid_argument);

  params = SessionParams{};
  params.start_frame_wait_ms = 0.0;
  EXPECT_THROW(ValidateSessionParams(params), std::invalid_argument);

  params = SessionParams{};
  params.live_frame_wait_ms = std::nan("");
  EXPECT_THROW(ValidateSessionParams(params), std::invalid_argument);
}

TEST(TrackingSession, ConstructorRejectsNullBufferAndMissingHooks) {
  Harness harness;
  auto make_null_buffer = [&harness] {
    TrackingSession session(std::shared_ptr<RosFrameBuffer>{}, SessionParams{},
                            harness.MakeHooks());
  };
  EXPECT_THROW(make_null_buffer(), std::invalid_argument);

  auto buffer = MakeBuffer();
  auto hooks = harness.MakeHooks();
  hooks.track = nullptr;
  auto make_missing_hook = [&buffer, &hooks] {
    TrackingSession session(buffer, SessionParams{}, hooks);
  };
  EXPECT_THROW(make_missing_hook(), std::invalid_argument);
}
