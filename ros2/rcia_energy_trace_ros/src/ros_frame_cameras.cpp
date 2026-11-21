#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sensor_msgs/image_encodings.hpp>

namespace rcia_energy_trace_ros {

namespace {

constexpr std::int64_t kTimeResetThresholdNs = 100000000LL;

std::int64_t StampToNanoseconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

template <typename Frame>
void PruneHistory(std::deque<Frame> *frames, std::size_t max_history_size) {
  while (frames->size() > max_history_size) frames->pop_front();
}

std::int64_t AbsDiff(std::int64_t lhs, std::int64_t rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

}  // namespace

m3t::Intrinsics RosFrameBuffer::ToIntrinsics(
    const sensor_msgs::msg::CameraInfo &msg) {
  m3t::Intrinsics intrinsics{};
  intrinsics.fu = static_cast<float>(msg.k[0]);
  intrinsics.fv = static_cast<float>(msg.k[4]);
  intrinsics.ppu = static_cast<float>(msg.k[2]);
  intrinsics.ppv = static_cast<float>(msg.k[5]);
  intrinsics.width = static_cast<int>(msg.width);
  intrinsics.height = static_cast<int>(msg.height);
  return intrinsics;
}

bool RosFrameBuffer::SameIntrinsics(const m3t::Intrinsics &lhs,
                                    const m3t::Intrinsics &rhs) {
  return lhs.fu == rhs.fu && lhs.fv == rhs.fv && lhs.ppu == rhs.ppu &&
         lhs.ppv == rhs.ppv && lhs.width == rhs.width &&
         lhs.height == rhs.height;
}

void RosFrameBuffer::SetColorCameraInfo(
    const sensor_msgs::msg::CameraInfo &msg) {
  const auto intrinsics = ToIntrinsics(msg);
  const std::lock_guard<std::mutex> lock{mutex_};
  const bool changed =
      color_intrinsics_ && !SameIntrinsics(*color_intrinsics_, intrinsics);
  color_intrinsics_ = intrinsics;
  if (changed) ChangeCalibrationLocked();
}

void RosFrameBuffer::SetDepthCameraInfo(
    const sensor_msgs::msg::CameraInfo &msg) {
  const auto intrinsics = ToIntrinsics(msg);
  const std::lock_guard<std::mutex> lock{mutex_};
  const bool changed =
      depth_intrinsics_ && !SameIntrinsics(*depth_intrinsics_, intrinsics);
  depth_intrinsics_ = intrinsics;
  if (changed) ChangeCalibrationLocked();
}

void RosFrameBuffer::SetColorImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  cv_bridge::CvImageConstPtr cv_ptr =
      cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  const auto stamp_ns = StampToNanoseconds(msg->header.stamp);
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!color_history_.empty() && stamp_ns <= color_history_.back().stamp_ns) {
    const auto backward_ns = color_history_.back().stamp_ns - stamp_ns;
    if (backward_ns <= kTimeResetThresholdNs) return;
    ResetTimelineLocked();
  }
  color_history_.push_back(StampedColorFrame{stamp_ns, cv_ptr->image.clone()});
  PruneHistory(&color_history_, max_history_size_);
  NotifyLocked();
}

void RosFrameBuffer::SetDepthImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
  const auto stamp_ns = StampToNanoseconds(msg->header.stamp);
  cv::Mat depth;
  if (cv_ptr->image.type() == CV_32FC1) {
    cv_ptr->image.convertTo(depth, CV_16UC1, 1000.0);
  } else if (cv_ptr->image.type() == CV_16UC1) {
    depth = cv_ptr->image.clone();
  } else {
    cv_ptr->image.convertTo(depth, CV_16UC1);
  }
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!depth_history_.empty() && stamp_ns <= depth_history_.back().stamp_ns) {
    const auto backward_ns = depth_history_.back().stamp_ns - stamp_ns;
    if (backward_ns <= kTimeResetThresholdNs) return;
    ResetTimelineLocked();
  }
  depth_history_.push_back(StampedDepthFrame{stamp_ns, depth.clone(), 0.001f});
  PruneHistory(&depth_history_, max_history_size_);
  NotifyLocked();
}

void RosFrameBuffer::SetMaxHistorySize(std::size_t max_history_size) {
  const std::lock_guard<std::mutex> lock{mutex_};
  max_history_size_ = std::max<std::size_t>(1, max_history_size);
  PruneHistory(&color_history_, max_history_size_);
  PruneHistory(&depth_history_, max_history_size_);
}

void RosFrameBuffer::SetStartFrameTolerance(std::chrono::nanoseconds tolerance) {
  const std::lock_guard<std::mutex> lock{mutex_};
  start_frame_tolerance_ = std::max(std::chrono::nanoseconds{0}, tolerance);
}

void RosFrameBuffer::SetPairTolerance(std::chrono::nanoseconds tolerance) {
  const std::lock_guard<std::mutex> lock{mutex_};
  pair_tolerance_ = std::max(std::chrono::nanoseconds{0}, tolerance);
}

void RosFrameBuffer::SetMaxFrameGap(std::chrono::nanoseconds max_gap) {
  const std::lock_guard<std::mutex> lock{mutex_};
  max_frame_gap_ = std::max(std::chrono::nanoseconds{0}, max_gap);
}

void RosFrameBuffer::SetRequireActivePair(bool require_active_pair) {
  const std::lock_guard<std::mutex> lock{mutex_};
  require_active_pair_ = require_active_pair;
}

FrameSelection RosFrameBuffer::BaseSelectionLocked() const {
  FrameSelection selection;
  selection.timeline_epoch = timeline_epoch_;
  selection.calibration_epoch = calibration_epoch_;
  return selection;
}

bool RosFrameBuffer::HasIntrinsicsLocked() const {
  return color_intrinsics_.has_value() && depth_intrinsics_.has_value();
}

const RosFrameBuffer::StampedDepthFrame *RosFrameBuffer::FindDepthLocked(
    std::int64_t color_ns, std::int64_t after_depth_ns) const {
  const auto tolerance_ns = pair_tolerance_.count();
  const auto first = std::lower_bound(
      depth_history_.begin(), depth_history_.end(), color_ns - tolerance_ns,
      [](const StampedDepthFrame &frame, std::int64_t value) {
        return frame.stamp_ns < value;
      });
  const StampedDepthFrame *best = nullptr;
  std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
  for (auto it = first;
       it != depth_history_.end() && it->stamp_ns <= color_ns + tolerance_ns;
       ++it) {
    if (it->stamp_ns <= after_depth_ns) continue;
    const auto delta = AbsDiff(it->stamp_ns, color_ns);
    if (delta < best_delta) {
      best_delta = delta;
      best = &(*it);
    }
  }
  return best;
}

std::shared_ptr<const FramePair> RosFrameBuffer::MakePairLocked(
    const StampedColorFrame &color, const StampedDepthFrame &depth) const {
  auto pair = std::make_shared<FramePair>();
  pair->color_stamp_ns = color.stamp_ns;
  pair->depth_stamp_ns = depth.stamp_ns;
  pair->color = color.image;
  pair->depth = depth.image;
  pair->color_intrinsics = *color_intrinsics_;
  pair->depth_intrinsics = *depth_intrinsics_;
  pair->depth_scale = depth.depth_scale;
  return pair;
}

void RosFrameBuffer::ClearHistoriesLocked() {
  color_history_.clear();
  depth_history_.clear();
}

void RosFrameBuffer::ResetTimelineLocked() {
  ++timeline_epoch_;
  ClearHistoriesLocked();
  NotifyLocked();
}

void RosFrameBuffer::ChangeCalibrationLocked() {
  ++calibration_epoch_;
  ClearHistoriesLocked();
  NotifyLocked();
}

void RosFrameBuffer::NotifyLocked() {
  ++frame_version_;
  cv_.notify_all();
}

FrameSelection RosFrameBuffer::SelectStartPair(std::int64_t target_ns) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  FrameSelection selection = BaseSelectionLocked();
  if (!HasIntrinsicsLocked() || color_history_.empty()) return selection;

  const auto start_tolerance_ns = start_frame_tolerance_.count();
  const StampedColorFrame *best_color = nullptr;
  const StampedDepthFrame *best_depth = nullptr;
  std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
  const StampedColorFrame *latest_in_window = nullptr;
  for (const auto &color : color_history_) {
    if (color.stamp_ns < target_ns - start_tolerance_ns) continue;
    if (color.stamp_ns > target_ns + start_tolerance_ns) break;
    latest_in_window = &color;
    const auto *depth = FindDepthLocked(color.stamp_ns, std::numeric_limits<std::int64_t>::min());
    if (!depth) continue;
    const auto delta = AbsDiff(color.stamp_ns, target_ns);
    if (delta < best_delta) {
      best_delta = delta;
      best_color = &color;
      best_depth = depth;
    }
  }

  if (best_color) {
    selection.code = FrameSelectCode::Ready;
    selection.pair = MakePairLocked(*best_color, *best_depth);
    return selection;
  }

  if (latest_in_window) {
    // Only give up early if the depth stream has already moved past every
    // possible depth that could pair with any color in [target - tol, target + tol].
    const bool depth_passed =
        !depth_history_.empty() &&
        depth_history_.back().stamp_ns >
            target_ns + start_tolerance_ns + pair_tolerance_.count();
    selection.code =
        depth_passed ? FrameSelectCode::HistoryExpired : FrameSelectCode::Waiting;
    return selection;
  }

  // No color frame in the window: either the target has not arrived yet or the
  // color stream already skipped over it.
  if (color_history_.back().stamp_ns > target_ns + start_tolerance_ns) {
    selection.code = FrameSelectCode::HistoryExpired;
  }
  return selection;
}

FrameSelection RosFrameBuffer::SelectNextPair(
    std::int64_t after_color_ns, std::int64_t after_depth_ns,
    std::uint64_t expected_timeline_epoch,
    std::uint64_t expected_calibration_epoch) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  FrameSelection selection = BaseSelectionLocked();
  if (expected_timeline_epoch != timeline_epoch_) {
    selection.code = FrameSelectCode::TimeReset;
    return selection;
  }
  if (expected_calibration_epoch != calibration_epoch_) {
    selection.code = FrameSelectCode::CameraChanged;
    return selection;
  }
  if (!HasIntrinsicsLocked()) return selection;

  std::size_t skipped = 0;
  for (const auto &color : color_history_) {
    if (color.stamp_ns <= after_color_ns) continue;
    const auto *depth = FindDepthLocked(color.stamp_ns, after_depth_ns);
    if (!depth) {
      ++skipped;
      continue;
    }
    selection.pair = MakePairLocked(color, *depth);
    selection.skipped_color_frames = skipped;
    selection.code = color.stamp_ns - after_color_ns > max_frame_gap_.count()
                         ? FrameSelectCode::GapTooLarge
                         : FrameSelectCode::Ready;
    return selection;
  }
  return selection;
}

FrameSelection RosFrameBuffer::SelectLatestPair(
    std::int64_t after_color_ns, std::int64_t after_depth_ns,
    std::uint64_t expected_timeline_epoch,
    std::uint64_t expected_calibration_epoch) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  FrameSelection selection = BaseSelectionLocked();
  if (expected_timeline_epoch != timeline_epoch_) {
    selection.code = FrameSelectCode::TimeReset;
    return selection;
  }
  if (expected_calibration_epoch != calibration_epoch_) {
    selection.code = FrameSelectCode::CameraChanged;
    return selection;
  }
  if (!HasIntrinsicsLocked()) return selection;

  for (auto it = color_history_.rbegin(); it != color_history_.rend(); ++it) {
    if (it->stamp_ns <= after_color_ns) break;
    const auto *depth = FindDepthLocked(it->stamp_ns, after_depth_ns);
    if (!depth) continue;
    selection.pair = MakePairLocked(*it, *depth);
    selection.code = it->stamp_ns - after_color_ns > max_frame_gap_.count()
                         ? FrameSelectCode::GapTooLarge
                         : FrameSelectCode::Ready;
    return selection;
  }
  return selection;
}

void RosFrameBuffer::ActivatePair(const std::shared_ptr<const FramePair> &pair) {
  const std::lock_guard<std::mutex> lock{mutex_};
  active_pair_ = pair;
}

std::shared_ptr<const FramePair> RosFrameBuffer::ActivePair() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return active_pair_;
}

void RosFrameBuffer::ClearActivePair() {
  const std::lock_guard<std::mutex> lock{mutex_};
  active_pair_.reset();
}

std::optional<std::int64_t> RosFrameBuffer::LatestColorStampNs() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (color_history_.empty()) return std::nullopt;
  return color_history_.back().stamp_ns;
}

std::optional<std::int64_t> RosFrameBuffer::LatestDepthStampNs() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (depth_history_.empty()) return std::nullopt;
  return depth_history_.back().stamp_ns;
}

std::optional<std::int64_t> RosFrameBuffer::LatestCompleteColorStampNs() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  for (auto it = color_history_.rbegin(); it != color_history_.rend(); ++it) {
    if (FindDepthLocked(it->stamp_ns, std::numeric_limits<std::int64_t>::min())) {
      return it->stamp_ns;
    }
  }
  return std::nullopt;
}

void RosFrameBuffer::ResetHistory() {
  const std::lock_guard<std::mutex> lock{mutex_};
  ClearHistoriesLocked();
}

bool RosFrameBuffer::WaitForFrames(std::chrono::milliseconds timeout,
                                   const std::atomic<bool> &cancelled) {
  std::unique_lock<std::mutex> lock{mutex_};
  const auto start_version = frame_version_;
  cv_.wait_for(lock, timeout, [&]() {
    return cancelled.load() || frame_version_ != start_version;
  });
  return frame_version_ != start_version;
}

void RosFrameBuffer::WakeWaiters() {
  const std::lock_guard<std::mutex> lock{mutex_};
  cv_.notify_all();
}

bool RosFrameBuffer::CopyColor(cv::Mat *image,
                               m3t::Intrinsics *intrinsics) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (active_pair_) {
    active_pair_->color.copyTo(*image);
    *intrinsics = active_pair_->color_intrinsics;
    return !image->empty();
  }
  if (require_active_pair_) return false;
  if (!color_intrinsics_ || color_history_.empty()) return false;
  color_history_.back().image.copyTo(*image);
  *intrinsics = *color_intrinsics_;
  return true;
}

bool RosFrameBuffer::CopyDepth(cv::Mat *image, m3t::Intrinsics *intrinsics,
                               float *depth_scale) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (active_pair_) {
    active_pair_->depth.copyTo(*image);
    *intrinsics = active_pair_->depth_intrinsics;
    *depth_scale = active_pair_->depth_scale;
    return !image->empty();
  }
  if (require_active_pair_) return false;
  if (!depth_intrinsics_ || depth_history_.empty()) return false;
  depth_history_.back().image.copyTo(*image);
  *intrinsics = *depth_intrinsics_;
  *depth_scale = depth_history_.back().depth_scale;
  return true;
}

bool RosFrameBuffer::HasColor() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return !color_history_.empty() && color_intrinsics_.has_value();
}

bool RosFrameBuffer::HasDepth() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return !depth_history_.empty() && depth_intrinsics_.has_value();
}

RosColorCamera::RosColorCamera(const std::string &name,
                               const std::shared_ptr<RosFrameBuffer> &buffer)
    : ColorCamera{name}, buffer_{buffer} {}

bool RosColorCamera::SetUp() {
  set_up_ = false;
  if (!UpdateImage(true)) {
    std::cerr << "ROS color camera has no image/camera_info yet" << std::endl;
    return false;
  }
  SaveMetaDataIfDesired();
  set_up_ = true;
  return true;
}

bool RosColorCamera::UpdateImage(bool synchronized) {
  (void)synchronized;
  if (!buffer_->CopyColor(&image_, &intrinsics_)) return false;
  SaveImageIfDesired();
  return true;
}

RosDepthCamera::RosDepthCamera(const std::string &name,
                               const std::shared_ptr<RosFrameBuffer> &buffer)
    : DepthCamera{name}, buffer_{buffer} {}

bool RosDepthCamera::SetUp() {
  set_up_ = false;
  if (!UpdateImage(true)) {
    std::cerr << "ROS depth camera has no image/camera_info yet" << std::endl;
    return false;
  }
  SaveMetaDataIfDesired();
  set_up_ = true;
  return true;
}

bool RosDepthCamera::UpdateImage(bool synchronized) {
  (void)synchronized;
  if (!buffer_->CopyDepth(&image_, &intrinsics_, &depth_scale_)) return false;
  SaveImageIfDesired();
  return true;
}

}  // namespace rcia_energy_trace_ros
