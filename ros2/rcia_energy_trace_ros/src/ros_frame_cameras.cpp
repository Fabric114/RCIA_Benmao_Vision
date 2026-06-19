#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sensor_msgs/image_encodings.hpp>

namespace rcia_energy_trace_ros {

namespace {

std::int64_t StampToNanoseconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

template <typename Frame>
std::optional<Frame> FindNearestFrame(const std::deque<Frame> &frames,
                                      std::int64_t target_ns,
                                      std::int64_t tolerance_ns) {
  std::optional<Frame> best_frame;
  std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
  for (const auto &frame : frames) {
    const auto delta =
        frame.stamp_ns > target_ns ? frame.stamp_ns - target_ns
                                   : target_ns - frame.stamp_ns;
    if (delta < best_delta) {
      best_delta = delta;
      best_frame = frame;
    }
  }
  if (!best_frame || best_delta > tolerance_ns) return std::nullopt;
  return best_frame;
}

template <typename Frame>
void PruneHistory(std::deque<Frame> *frames, std::size_t max_history_size) {
  while (frames->size() > max_history_size) frames->pop_front();
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

void RosFrameBuffer::SetColorCameraInfo(
    const sensor_msgs::msg::CameraInfo &msg) {
  const std::lock_guard<std::mutex> lock{mutex_};
  color_intrinsics_ = ToIntrinsics(msg);
}

void RosFrameBuffer::SetDepthCameraInfo(
    const sensor_msgs::msg::CameraInfo &msg) {
  const std::lock_guard<std::mutex> lock{mutex_};
  depth_intrinsics_ = ToIntrinsics(msg);
}

void RosFrameBuffer::SetColorImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  cv_bridge::CvImageConstPtr cv_ptr =
      cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  const std::lock_guard<std::mutex> lock{mutex_};
  cv_ptr->image.copyTo(color_image_);
  color_history_.push_back(
      StampedColorFrame{StampToNanoseconds(msg->header.stamp), color_image_.clone()});
  PruneHistory(&color_history_, max_history_size_);
}

void RosFrameBuffer::SetDepthImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
  const std::lock_guard<std::mutex> lock{mutex_};
  if (cv_ptr->image.type() == CV_32FC1) {
    cv::Mat meters;
    cv_ptr->image.convertTo(meters, CV_16UC1, 1000.0);
    meters.copyTo(depth_image_);
    depth_scale_ = 0.001f;
  } else if (cv_ptr->image.type() == CV_16UC1) {
    cv_ptr->image.copyTo(depth_image_);
    depth_scale_ = 0.001f;
  } else {
    cv::Mat converted;
    cv_ptr->image.convertTo(converted, CV_16UC1);
    converted.copyTo(depth_image_);
    depth_scale_ = 0.001f;
  }
  depth_history_.push_back(StampedDepthFrame{StampToNanoseconds(msg->header.stamp),
                                             depth_image_.clone(), depth_scale_});
  PruneHistory(&depth_history_, max_history_size_);
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

bool RosFrameBuffer::PinStartFrame(const builtin_interfaces::msg::Time &stamp) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!color_intrinsics_ || !depth_intrinsics_) return false;
  const auto target_ns = StampToNanoseconds(stamp);
  const auto tolerance_ns = start_frame_tolerance_.count();
  auto color = FindNearestFrame(color_history_, target_ns, tolerance_ns);
  auto depth = FindNearestFrame(depth_history_, target_ns, tolerance_ns);
  if (!color || !depth) return false;

  pinned_color_ = StampedColorFrame{color->stamp_ns, color->image.clone()};
  pinned_depth_ =
      StampedDepthFrame{depth->stamp_ns, depth->image.clone(), depth->depth_scale};
  return true;
}

bool RosFrameBuffer::HasPinnedStartFrame() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return pinned_color_.has_value() && pinned_depth_.has_value();
}

void RosFrameBuffer::ReleasePinnedStartFrame() {
  const std::lock_guard<std::mutex> lock{mutex_};
  pinned_color_.reset();
  pinned_depth_.reset();
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

std::optional<std::int64_t> RosFrameBuffer::PinnedColorStampNs() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!pinned_color_) return std::nullopt;
  return pinned_color_->stamp_ns;
}

std::optional<std::int64_t> RosFrameBuffer::PinnedDepthStampNs() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!pinned_depth_) return std::nullopt;
  return pinned_depth_->stamp_ns;
}

bool RosFrameBuffer::CopyColor(cv::Mat *image,
                               m3t::Intrinsics *intrinsics) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!color_intrinsics_) return false;
  if (pinned_color_) {
    pinned_color_->image.copyTo(*image);
  } else {
    if (color_image_.empty()) return false;
    color_image_.copyTo(*image);
  }
  *intrinsics = *color_intrinsics_;
  return true;
}

bool RosFrameBuffer::CopyDepth(cv::Mat *image, m3t::Intrinsics *intrinsics,
                               float *depth_scale) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!depth_intrinsics_) return false;
  if (pinned_depth_) {
    pinned_depth_->image.copyTo(*image);
    *depth_scale = pinned_depth_->depth_scale;
  } else {
    if (depth_image_.empty()) return false;
    depth_image_.copyTo(*image);
    *depth_scale = depth_scale_;
  }
  *intrinsics = *depth_intrinsics_;
  return true;
}

bool RosFrameBuffer::HasColor() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return !color_image_.empty() && color_intrinsics_.has_value();
}

bool RosFrameBuffer::HasDepth() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return !depth_image_.empty() && depth_intrinsics_.has_value();
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
