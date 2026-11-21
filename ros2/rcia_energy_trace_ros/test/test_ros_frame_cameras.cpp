#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>

namespace {

using rcia_energy_trace_ros::FrameSelectCode;
using rcia_energy_trace_ros::FrameSelection;
using rcia_energy_trace_ros::RosFrameBuffer;

constexpr std::int64_t kMs = 1000000;

void SetStamp(builtin_interfaces::msg::Time *stamp, std::int64_t stamp_ns) {
  stamp->sec = static_cast<std::int32_t>(stamp_ns / 1000000000LL);
  stamp->nanosec = static_cast<std::uint32_t>(stamp_ns % 1000000000LL);
}

// Color frames are filled with `value` so tests can identify which frame was copied.
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

sensor_msgs::msg::CameraInfo MakeInfo(double fu = 100.0, double fv = 100.0,
                                      std::uint32_t width = 4,
                                      std::uint32_t height = 4,
                                      std::int64_t stamp_ns = 0) {
  sensor_msgs::msg::CameraInfo info;
  SetStamp(&info.header.stamp, stamp_ns);
  info.width = width;
  info.height = height;
  info.k = {fu, 0.0, 2.0, 0.0, fv, 2.0, 0.0, 0.0, 1.0};
  return info;
}

std::shared_ptr<RosFrameBuffer> MakeBuffer() {
  auto buffer = std::make_shared<RosFrameBuffer>();
  buffer->SetColorCameraInfo(MakeInfo());
  buffer->SetDepthCameraInfo(MakeInfo());
  return buffer;
}

void PushColor(RosFrameBuffer *buffer, std::int64_t stamp_ms, std::uint8_t value) {
  buffer->SetColorImage(MakeColor(stamp_ms * kMs, value));
}

void PushDepth(RosFrameBuffer *buffer, std::int64_t stamp_ms, std::uint16_t value = 1000) {
  buffer->SetDepthImage(MakeDepth(stamp_ms * kMs, value));
}

}  // namespace

TEST(RosFrameCameras, SelectNextPairReturnsEarliestCompletePairAndCountsSkippedColors) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 33, 2);
  PushColor(buffer.get(), 66, 3);
  PushDepth(buffer.get(), 2);
  PushDepth(buffer.get(), 35);
  PushDepth(buffer.get(), 68);

  const auto first = buffer->SelectNextPair(-1, -1, 0, 0);
  ASSERT_EQ(first.code, FrameSelectCode::Ready);
  EXPECT_EQ(first.pair->color_stamp_ns, 0);
  EXPECT_EQ(first.pair->depth_stamp_ns, 2 * kMs);
  EXPECT_EQ(first.skipped_color_frames, 0u);

  const auto second = buffer->SelectNextPair(0, 2 * kMs, 0, 0);
  ASSERT_EQ(second.code, FrameSelectCode::Ready);
  EXPECT_EQ(second.pair->color_stamp_ns, 33 * kMs);
  EXPECT_EQ(second.pair->depth_stamp_ns, 35 * kMs);
}

TEST(RosFrameCameras, SelectNextPairSkipsUnpairableColorFramesAndCountsThem) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 33, 2);
  PushColor(buffer.get(), 66, 3);
  PushDepth(buffer.get(), 70);

  const auto selection = buffer->SelectNextPair(-1, -1, 0, 0);
  ASSERT_EQ(selection.code, FrameSelectCode::Ready);
  EXPECT_EQ(selection.pair->color_stamp_ns, 66 * kMs);
  EXPECT_EQ(selection.skipped_color_frames, 2u);
}

TEST(RosFrameCameras, SelectNextPairWaitsWhenNoCompleteSuccessorExists) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 200, 2);
  PushDepth(buffer.get(), 0);

  // The color at 200 ms has no depth yet, which must not be reported as a gap.
  const auto selection = buffer->SelectNextPair(0, 0, 0, 0);
  EXPECT_EQ(selection.code, FrameSelectCode::Waiting);
}

TEST(RosFrameCameras, PairToleranceIsTenMillisecondsByDefault) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushDepth(buffer.get(), 11);
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 0).code, FrameSelectCode::Waiting);

  auto within = MakeBuffer();
  PushColor(within.get(), 0, 1);
  PushDepth(within.get(), 10);
  const auto selection = within->SelectNextPair(-1, -1, 0, 0);
  ASSERT_EQ(selection.code, FrameSelectCode::Ready);
  EXPECT_EQ(selection.pair->depth_stamp_ns, 10 * kMs);
}

TEST(RosFrameCameras, PairToleranceCanBeConfigured) {
  auto buffer = MakeBuffer();
  buffer->SetPairTolerance(std::chrono::milliseconds{20});
  PushColor(buffer.get(), 0, 1);
  PushDepth(buffer.get(), 15);
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 0).code, FrameSelectCode::Ready);
}

TEST(RosFrameCameras, GapOverMaxFrameGapReturnsGapTooLarge) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 120, 2);
  PushDepth(buffer.get(), 0);
  PushDepth(buffer.get(), 120);

  const auto gap = buffer->SelectNextPair(0, 0, 0, 0);
  EXPECT_EQ(gap.code, FrameSelectCode::GapTooLarge);
  ASSERT_NE(gap.pair, nullptr);
  EXPECT_EQ(gap.pair->color_stamp_ns, 120 * kMs);

  const auto latest_gap = buffer->SelectLatestPair(0, 0, 0, 0);
  EXPECT_EQ(latest_gap.code, FrameSelectCode::GapTooLarge);

  buffer->SetMaxFrameGap(std::chrono::milliseconds{150});
  EXPECT_EQ(buffer->SelectNextPair(0, 0, 0, 0).code, FrameSelectCode::Ready);
}

TEST(RosFrameCameras, OldDepthFrameIsNeverReused) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 5, 2);
  PushDepth(buffer.get(), 0);

  // Color 5 ms is nearest to depth 0 ms, but depth 0 ms is not after depth 0 ms.
  EXPECT_EQ(buffer->SelectNextPair(0, 0, 0, 0).code, FrameSelectCode::Waiting);
  EXPECT_EQ(buffer->SelectLatestPair(0, 0, 0, 0).code, FrameSelectCode::Waiting);
}

TEST(RosFrameCameras, StartPairPrefersPairableColorAmongCandidatesInWindow) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 90, 1);
  PushColor(buffer.get(), 110, 2);
  PushDepth(buffer.get(), 110);

  const auto selection = buffer->SelectStartPair(100 * kMs);
  ASSERT_EQ(selection.code, FrameSelectCode::Ready);
  EXPECT_EQ(selection.pair->color_stamp_ns, 110 * kMs);
}

TEST(RosFrameCameras, StartPairReportsWaitingAndHistoryExpired) {
  auto buffer = MakeBuffer();
  EXPECT_EQ(buffer->SelectStartPair(100 * kMs).code, FrameSelectCode::Waiting);

  PushColor(buffer.get(), 0, 1);
  EXPECT_EQ(buffer->SelectStartPair(100 * kMs).code, FrameSelectCode::Waiting);

  PushColor(buffer.get(), 300, 2);
  EXPECT_EQ(buffer->SelectStartPair(100 * kMs).code, FrameSelectCode::HistoryExpired);

  auto depth_passed = MakeBuffer();
  PushColor(depth_passed.get(), 100, 1);
  PushDepth(depth_passed.get(), 200);
  EXPECT_EQ(depth_passed->SelectStartPair(100 * kMs).code, FrameSelectCode::HistoryExpired);

  auto depth_pending = MakeBuffer();
  PushColor(depth_pending.get(), 100, 1);
  EXPECT_EQ(depth_pending->SelectStartPair(100 * kMs).code, FrameSelectCode::Waiting);

  // Depth at 105 ms can still pair with a color frame at 105 ms (within 100 +- 10 ms),
  // even if an earlier color at 90 ms could not pair with depth 105 ms.
  auto color_slightly_late = MakeBuffer();
  PushColor(color_slightly_late.get(), 90, 1);
  PushDepth(color_slightly_late.get(), 105);
  EXPECT_EQ(color_slightly_late->SelectStartPair(100 * kMs).code, FrameSelectCode::Waiting);
  PushColor(color_slightly_late.get(), 105, 2);
  const auto paired = color_slightly_late->SelectStartPair(100 * kMs);
  ASSERT_EQ(paired.code, FrameSelectCode::Ready);
  EXPECT_EQ(paired.pair->color_stamp_ns, 105 * kMs);
}

TEST(RosFrameCameras, ActivePairIsolatesCopiesFromNewerFrames) {
  auto buffer = MakeBuffer();
  buffer->SetRequireActivePair(true);
  PushColor(buffer.get(), 0, 7);
  PushDepth(buffer.get(), 0, 1234);

  cv::Mat image;
  m3t::Intrinsics intrinsics{};
  EXPECT_FALSE(buffer->CopyColor(&image, &intrinsics));

  const auto start = buffer->SelectStartPair(0);
  ASSERT_EQ(start.code, FrameSelectCode::Ready);
  buffer->ActivatePair(start.pair);
  EXPECT_EQ(buffer->ActivePair(), start.pair);

  PushColor(buffer.get(), 33, 99);
  PushDepth(buffer.get(), 33, 9999);

  ASSERT_TRUE(buffer->CopyColor(&image, &intrinsics));
  EXPECT_EQ(image.at<cv::Vec3b>(0, 0)[0], 7);
  EXPECT_EQ(intrinsics.width, 4);

  float depth_scale = 0.0f;
  cv::Mat depth;
  ASSERT_TRUE(buffer->CopyDepth(&depth, &intrinsics, &depth_scale));
  EXPECT_EQ(depth.at<std::uint16_t>(0, 0), 1234);
  EXPECT_FLOAT_EQ(depth_scale, 0.001f);

  buffer->ClearActivePair();
  EXPECT_EQ(buffer->ActivePair(), nullptr);
  EXPECT_FALSE(buffer->CopyColor(&image, &intrinsics));
}

TEST(RosFrameCameras, LatestCompleteColorIgnoresColorsWithoutDepth) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushColor(buffer.get(), 33, 2);
  PushDepth(buffer.get(), 1);
  EXPECT_EQ(buffer->LatestCompleteColorStampNs(), std::optional<std::int64_t>{0});
}

TEST(RosFrameCameras, SameOrSmallBackwardStampsAreDropped) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 1000, 1);
  PushColor(buffer.get(), 1000, 2);
  PushColor(buffer.get(), 950, 3);
  EXPECT_EQ(buffer->LatestColorStampNs(), std::optional<std::int64_t>{1000 * kMs});

  const auto selection = buffer->SelectLatestPair(-1, -1, 0, 0);
  EXPECT_EQ(selection.code, FrameSelectCode::Waiting);
  EXPECT_EQ(selection.timeline_epoch, 0u);
}

TEST(RosFrameCameras, BackwardJumpOverOneHundredMsResetsTimeline) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 1000, 1);
  PushDepth(buffer.get(), 1000);
  ASSERT_EQ(buffer->SelectNextPair(999 * kMs, 999 * kMs, 0, 0).code, FrameSelectCode::Ready);

  PushColor(buffer.get(), 0, 2);
  EXPECT_EQ(buffer->LatestColorStampNs(), std::optional<std::int64_t>{0});
  EXPECT_FALSE(buffer->LatestDepthStampNs().has_value());

  PushDepth(buffer.get(), 0);
  const auto stale = buffer->SelectNextPair(-1, -1, 0, 0);
  EXPECT_EQ(stale.code, FrameSelectCode::TimeReset);
  EXPECT_EQ(stale.timeline_epoch, 1u);

  const auto fresh = buffer->SelectNextPair(-1, -1, 1, 0);
  ASSERT_EQ(fresh.code, FrameSelectCode::Ready);
  EXPECT_EQ(fresh.pair->color_stamp_ns, 0);
  EXPECT_EQ(buffer->SelectLatestPair(-1, -1, 0, 0).code, FrameSelectCode::TimeReset);
}

TEST(RosFrameCameras, DepthBackwardJumpResetsTimelineAndClearsColorHistory) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 1000, 1);
  PushDepth(buffer.get(), 1000);
  PushDepth(buffer.get(), 500);
  EXPECT_EQ(buffer->SelectStartPair(1000 * kMs).code, FrameSelectCode::Waiting);
  EXPECT_FALSE(buffer->LatestColorStampNs().has_value());
  EXPECT_EQ(buffer->LatestDepthStampNs(), std::optional<std::int64_t>{500 * kMs});
}

TEST(RosFrameCameras, FirstCameraInfoDoesNotChangeCalibrationEpoch) {
  auto buffer = std::make_shared<RosFrameBuffer>();
  buffer->SetColorCameraInfo(MakeInfo());
  buffer->SetDepthCameraInfo(MakeInfo());
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 0).calibration_epoch, 0u);
}

TEST(RosFrameCameras, CameraInfoStampOnlyChangeDoesNotChangeCalibration) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  buffer->SetColorCameraInfo(MakeInfo(100.0, 100.0, 4, 4, 5000000000LL));
  buffer->SetDepthCameraInfo(MakeInfo(100.0, 100.0, 4, 4, 5000000000LL));
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 0).calibration_epoch, 0u);
  EXPECT_EQ(buffer->LatestColorStampNs(), std::optional<std::int64_t>{0});
}

TEST(RosFrameCameras, IntrinsicOrResolutionChangeClearsHistoryAndBumpsEpoch) {
  auto buffer = MakeBuffer();
  PushColor(buffer.get(), 0, 1);
  PushDepth(buffer.get(), 0);

  buffer->SetColorCameraInfo(MakeInfo(101.0));
  EXPECT_FALSE(buffer->LatestColorStampNs().has_value());
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 0).code, FrameSelectCode::CameraChanged);
  EXPECT_EQ(buffer->SelectNextPair(-1, -1, 0, 1).calibration_epoch, 1u);

  buffer->SetDepthCameraInfo(MakeInfo(101.0, 100.0, 8, 4));
  EXPECT_EQ(buffer->SelectLatestPair(-1, -1, 0, 1).calibration_epoch, 2u);
  EXPECT_EQ(buffer->SelectLatestPair(-1, -1, 0, 1).code, FrameSelectCode::CameraChanged);
  EXPECT_EQ(buffer->SelectLatestPair(-1, -1, 0, 2).code, FrameSelectCode::Waiting);
}

TEST(RosFrameCameras, WaitForFramesWakesOnNewFrameAndOnCancel) {
  auto buffer = MakeBuffer();
  std::atomic<bool> cancelled{false};

  std::thread producer([&buffer]() {
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    PushColor(buffer.get(), 0, 1);
  });
  EXPECT_TRUE(buffer->WaitForFrames(std::chrono::milliseconds{2000}, cancelled));
  producer.join();

  std::thread canceller([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    cancelled.store(true);
    buffer->WakeWaiters();
  });
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(buffer->WaitForFrames(std::chrono::milliseconds{2000}, cancelled));
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds{1000});
  canceller.join();
}

TEST(RosFrameCameras, ConcurrentWriterAndReadersKeepPairsConsistent) {
  auto buffer = MakeBuffer();
  std::atomic<bool> done{false};
  std::atomic<int> pairs_seen{0};
  std::atomic<int> violations{0};

  std::thread writer([&]() {
    for (std::int64_t i = 0; i < 2000; ++i) {
      PushColor(buffer.get(), i * 33, static_cast<std::uint8_t>(i % 200));
      PushDepth(buffer.get(), i * 33 + 1, static_cast<std::uint16_t>(i));
    }
    done.store(true);
  });

  auto reader = [&]() {
    std::int64_t after_color = -1;
    std::int64_t after_depth = -1;
    while (!done.load()) {
      const auto selection = buffer->SelectNextPair(after_color, after_depth, 0, 0);
      if (selection.code != FrameSelectCode::Ready && selection.code != FrameSelectCode::GapTooLarge) {
        continue;
      }
      const auto &pair = selection.pair;
      if (pair->color_stamp_ns <= after_color || pair->depth_stamp_ns <= after_depth) {
        ++violations;
      }
      if (pair->depth_stamp_ns - pair->color_stamp_ns > 10 * kMs ||
          pair->color_stamp_ns - pair->depth_stamp_ns > 10 * kMs) {
        ++violations;
      }
      after_color = pair->color_stamp_ns;
      after_depth = pair->depth_stamp_ns;
      buffer->ActivatePair(pair);
      cv::Mat image;
      m3t::Intrinsics intrinsics{};
      if (!buffer->CopyColor(&image, &intrinsics) || image.empty()) ++violations;
      ++pairs_seen;
    }
  };

  std::thread reader_a(reader);
  std::thread reader_b(reader);
  writer.join();
  reader_a.join();
  reader_b.join();
  buffer->ClearActivePair();

  EXPECT_EQ(violations.load(), 0);
  EXPECT_GT(pairs_seen.load(), 0);
}
