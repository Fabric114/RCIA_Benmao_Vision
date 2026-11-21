#pragma once

#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rcia_energy_trace_ros {

enum class SessionPhase { Starting, CatchingUp, Tracking };

// All durations are in milliseconds. ValidateSessionParams rejects values that
// are not finite, not positive, or where replay_target_lag_ms is not below
// max_output_pose_age_ms.
struct SessionParams {
  double start_frame_wait_ms = 1000.0;
  double replay_timeout_ms = 2000.0;
  double replay_target_lag_ms = 75.0;
  double max_output_pose_age_ms = 250.0;
  double live_frame_wait_ms = 500.0;
};

// Throws std::invalid_argument when params are invalid.
void ValidateSessionParams(const SessionParams &params);

struct SessionProgress {
  std::int64_t initial_pose_stamp_ns = 0;
  std::int64_t processed_frame_stamp_ns = 0;
  std::int64_t latest_frame_stamp_ns = 0;
  std::size_t replayed_frames = 0;
  std::size_t skipped_color_frames = 0;
};

struct SessionHooks {
  std::function<bool()> setup;
  std::function<bool()> start;
  // track(iteration): iteration 0 is the t0 frame; later iterations replay or track.
  std::function<bool(int)> track;
  std::function<bool()> pose_valid;
  std::function<void(const FramePair &)> publish_live;
  std::function<void(SessionPhase, const SessionProgress &)> report_phase;
  std::function<std::int64_t()> ros_now_ns;
  std::function<std::int64_t()> monotonic_now_ns;
};

struct SessionResult {
  bool cancelled = false;
  // Failure reason: history_expired, frame_gap, replay_timeout, camera_stalled,
  // invalid_pose, tracker_error. Set to "cancelled" when cancelled is true.
  std::string reason;
  SessionProgress progress;
};

// Runs one start-replay-track sequence on top of a RosFrameBuffer. Run() only
// returns on failure or cancellation. Hooks are called on the thread that calls Run().
class TrackingSession {
 public:
  TrackingSession(std::shared_ptr<RosFrameBuffer> buffer, const SessionParams &params,
                  SessionHooks hooks);

  SessionResult Run(std::int64_t initial_pose_stamp_ns);
  // Thread safe. Wakes a waiting Run() so it returns promptly.
  void Cancel();
  bool cancelled() const { return cancelled_.load(); }

 private:
  std::string TrackFrame(int iteration);
  void UpdateLatestStamp(SessionProgress *progress) const;
  bool CheckRosClock();
  bool IsCaughtUp(std::int64_t last_color_ns);
  void ReportPhase(SessionPhase phase, const SessionProgress &progress) const;

  std::shared_ptr<RosFrameBuffer> buffer_;
  SessionHooks hooks_;
  std::int64_t start_frame_wait_ns_ = 0;
  std::int64_t replay_timeout_ns_ = 0;
  std::int64_t replay_target_lag_ns_ = 0;
  std::int64_t max_output_pose_age_ns_ = 0;
  std::int64_t live_frame_wait_ns_ = 0;
  std::atomic<bool> cancelled_{false};
  bool ros_clock_initialized_ = false;
  std::int64_t last_ros_now_ns_ = 0;
};

}  // namespace rcia_energy_trace_ros
