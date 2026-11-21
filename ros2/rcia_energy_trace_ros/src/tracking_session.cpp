#include "rcia_energy_trace_ros/tracking_session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rcia_energy_trace_ros {

namespace {

constexpr char kCancelledReason[] = "cancelled";
constexpr char kHistoryExpired[] = "history_expired";
constexpr char kFrameGap[] = "frame_gap";
constexpr char kReplayTimeout[] = "replay_timeout";
constexpr char kCameraStalled[] = "camera_stalled";
constexpr char kInvalidPose[] = "invalid_pose";
constexpr char kTrackerError[] = "tracker_error";

constexpr std::int64_t kClockRegressNs = 100000000LL;
constexpr std::chrono::milliseconds kPollInterval{20};

std::int64_t MsToNs(double ms) {
  return static_cast<std::int64_t>(std::llround(ms * 1000000.0));
}

bool PositiveFinite(double value) { return std::isfinite(value) && value > 0.0; }

SessionResult Stop(SessionResult result, const std::string &reason) {
  result.cancelled = reason == kCancelledReason;
  result.reason = reason;
  return result;
}

}  // namespace

void ValidateSessionParams(const SessionParams &params) {
  if (!PositiveFinite(params.start_frame_wait_ms)) {
    throw std::invalid_argument("start_frame_wait_ms must be a positive finite value");
  }
  if (!PositiveFinite(params.replay_timeout_ms)) {
    throw std::invalid_argument("replay_timeout_ms must be a positive finite value");
  }
  if (!std::isfinite(params.replay_target_lag_ms) || params.replay_target_lag_ms < 0.0) {
    throw std::invalid_argument("replay_target_lag_ms must be a non-negative finite value");
  }
  if (!PositiveFinite(params.max_output_pose_age_ms)) {
    throw std::invalid_argument("max_output_pose_age_ms must be a positive finite value");
  }
  if (!PositiveFinite(params.live_frame_wait_ms)) {
    throw std::invalid_argument("live_frame_wait_ms must be a positive finite value");
  }
  if (params.replay_target_lag_ms >= params.max_output_pose_age_ms) {
    throw std::invalid_argument("replay_target_lag_ms must be less than max_output_pose_age_ms");
  }
}

TrackingSession::TrackingSession(std::shared_ptr<RosFrameBuffer> buffer,
                                 const SessionParams &params, SessionHooks hooks)
    : buffer_{std::move(buffer)}, hooks_{std::move(hooks)} {
  if (!buffer_) throw std::invalid_argument("frame buffer must not be null");
  ValidateSessionParams(params);
  if (!hooks_.setup || !hooks_.start || !hooks_.track || !hooks_.pose_valid ||
      !hooks_.publish_live || !hooks_.report_phase || !hooks_.ros_now_ns ||
      !hooks_.monotonic_now_ns) {
    throw std::invalid_argument("all session hooks must be set");
  }
  start_frame_wait_ns_ = MsToNs(params.start_frame_wait_ms);
  replay_timeout_ns_ = MsToNs(params.replay_timeout_ms);
  replay_target_lag_ns_ = MsToNs(params.replay_target_lag_ms);
  max_output_pose_age_ns_ = MsToNs(params.max_output_pose_age_ms);
  live_frame_wait_ns_ = MsToNs(params.live_frame_wait_ms);
}

void TrackingSession::Cancel() {
  cancelled_.store(true);
  buffer_->WakeWaiters();
}

void TrackingSession::UpdateLatestStamp(SessionProgress *progress) const {
  const auto latest = buffer_->LatestColorStampNs();
  if (latest) progress->latest_frame_stamp_ns = *latest;
}

void TrackingSession::ReportPhase(SessionPhase phase,
                                  const SessionProgress &progress) const {
  hooks_.report_phase(phase, progress);
}

bool TrackingSession::CheckRosClock() {
  const auto now = hooks_.ros_now_ns();
  if (!ros_clock_initialized_) {
    ros_clock_initialized_ = true;
    last_ros_now_ns_ = now;
    return true;
  }
  if (now < last_ros_now_ns_ - kClockRegressNs) return false;
  last_ros_now_ns_ = std::max(last_ros_now_ns_, now);
  return true;
}

bool TrackingSession::IsCaughtUp(std::int64_t last_color_ns) {
  const auto latest_complete = buffer_->LatestCompleteColorStampNs();
  if (!latest_complete) return false;
  if (*latest_complete - last_color_ns > replay_target_lag_ns_) return false;
  return hooks_.ros_now_ns() - last_color_ns <= max_output_pose_age_ns_;
}

// Returns an empty string on success, kCancelledReason when cancelled, or a
// failure reason otherwise.
std::string TrackingSession::TrackFrame(int iteration) {
  if (cancelled_.load()) return kCancelledReason;
  if (!hooks_.track(iteration)) return kTrackerError;
  if (cancelled_.load()) return kCancelledReason;
  if (!hooks_.pose_valid()) return kInvalidPose;
  if (cancelled_.load()) return kCancelledReason;
  return "";
}

SessionResult TrackingSession::Run(std::int64_t initial_pose_stamp_ns) {
  SessionResult result;
  result.progress.initial_pose_stamp_ns = initial_pose_stamp_ns;
  SessionProgress &progress = result.progress;

  // Starting: select and activate the t0 pair, then initialize M3T on it.
  ReportPhase(SessionPhase::Starting, progress);
  const auto start_deadline_ns = hooks_.monotonic_now_ns() + start_frame_wait_ns_;
  FrameSelection start;
  for (;;) {
    if (cancelled_.load()) return Stop(result, kCancelledReason);
    start = buffer_->SelectStartPair(initial_pose_stamp_ns);
    UpdateLatestStamp(&progress);
    if (start.code == FrameSelectCode::Ready) break;
    if (start.code == FrameSelectCode::HistoryExpired) {
      return Stop(result, kHistoryExpired);
    }
    if (start.code != FrameSelectCode::Waiting) return Stop(result, kTrackerError);
    if (hooks_.monotonic_now_ns() >= start_deadline_ns) {
      return Stop(result, kHistoryExpired);
    }
    buffer_->WaitForFrames(kPollInterval, cancelled_);
  }

  const std::uint64_t timeline_epoch = start.timeline_epoch;
  const std::uint64_t calibration_epoch = start.calibration_epoch;
  std::shared_ptr<const FramePair> pair = start.pair;
  progress.processed_frame_stamp_ns = pair->color_stamp_ns;
  buffer_->ActivatePair(pair);
  if (cancelled_.load()) return Stop(result, kCancelledReason);
  if (!hooks_.setup()) return Stop(result, kTrackerError);
  if (cancelled_.load()) return Stop(result, kCancelledReason);
  if (!hooks_.start()) return Stop(result, kTrackerError);
  if (const auto reason = TrackFrame(0); !reason.empty()) {
    return Stop(result, reason);
  }

  // CatchingUp: replay complete pairs after t0 without publishing until the
  // processed frame is close enough to the newest complete pair.
  ReportPhase(SessionPhase::CatchingUp, progress);
  const auto replay_started_ns = hooks_.monotonic_now_ns();
  std::int64_t last_color_ns = pair->color_stamp_ns;
  std::int64_t last_depth_ns = pair->depth_stamp_ns;
  int iteration = 1;
  for (;;) {
    if (cancelled_.load()) return Stop(result, kCancelledReason);
    if (!CheckRosClock()) return Stop(result, kTrackerError);
    if (hooks_.monotonic_now_ns() - replay_started_ns > replay_timeout_ns_) {
      return Stop(result, kReplayTimeout);
    }
    if (IsCaughtUp(last_color_ns)) break;

    FrameSelection next =
        buffer_->SelectNextPair(last_color_ns, last_depth_ns, timeline_epoch, calibration_epoch);
    UpdateLatestStamp(&progress);
    if (next.code == FrameSelectCode::Waiting) {
      buffer_->WaitForFrames(kPollInterval, cancelled_);
      continue;
    }
    if (next.code == FrameSelectCode::GapTooLarge) return Stop(result, kFrameGap);
    if (next.code == FrameSelectCode::HistoryExpired) return Stop(result, kHistoryExpired);
    if (next.code != FrameSelectCode::Ready) return Stop(result, kTrackerError);

    progress.skipped_color_frames += next.skipped_color_frames;
    pair = next.pair;
    buffer_->ActivatePair(pair);
    if (const auto reason = TrackFrame(iteration); !reason.empty()) {
      return Stop(result, reason);
    }
    ++iteration;
    last_color_ns = pair->color_stamp_ns;
    last_depth_ns = pair->depth_stamp_ns;
    progress.processed_frame_stamp_ns = last_color_ns;
    ++progress.replayed_frames;
    ReportPhase(SessionPhase::CatchingUp, progress);
  }

  // Tracking: publish each newly processed pair as it is tracked.
  ReportPhase(SessionPhase::Tracking, progress);
  auto last_processed_ns = hooks_.monotonic_now_ns();
  for (;;) {
    if (cancelled_.load()) return Stop(result, kCancelledReason);
    if (!CheckRosClock()) return Stop(result, kTrackerError);

    FrameSelection live =
        buffer_->SelectLatestPair(last_color_ns, last_depth_ns, timeline_epoch, calibration_epoch);
    UpdateLatestStamp(&progress);
    if (live.code == FrameSelectCode::Waiting) {
      if (hooks_.monotonic_now_ns() - last_processed_ns > live_frame_wait_ns_) {
        return Stop(result, kCameraStalled);
      }
      buffer_->WaitForFrames(kPollInterval, cancelled_);
      continue;
    }
    if (live.code == FrameSelectCode::GapTooLarge) return Stop(result, kFrameGap);
    if (live.code != FrameSelectCode::Ready) return Stop(result, kTrackerError);

    pair = live.pair;
    buffer_->ActivatePair(pair);
    if (const auto reason = TrackFrame(iteration); !reason.empty()) {
      return Stop(result, reason);
    }
    ++iteration;
    last_color_ns = pair->color_stamp_ns;
    last_depth_ns = pair->depth_stamp_ns;
    progress.processed_frame_stamp_ns = last_color_ns;
    last_processed_ns = hooks_.monotonic_now_ns();

    if (cancelled_.load()) return Stop(result, kCancelledReason);
    if (hooks_.ros_now_ns() - last_color_ns <= max_output_pose_age_ns_) {
      hooks_.publish_live(*pair);
    }
  }
}

}  // namespace rcia_energy_trace_ros
