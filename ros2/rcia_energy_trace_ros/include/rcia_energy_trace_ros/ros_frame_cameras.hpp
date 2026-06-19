#pragma once

#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/time.hpp>
#include <m3t/camera.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace rcia_energy_trace_ros {

class RosFrameBuffer {
 public:
  void SetColorCameraInfo(const sensor_msgs::msg::CameraInfo &msg);
  void SetDepthCameraInfo(const sensor_msgs::msg::CameraInfo &msg);
  void SetColorImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void SetDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void SetMaxHistorySize(std::size_t max_history_size);
  void SetStartFrameTolerance(std::chrono::nanoseconds tolerance);
  bool PinStartFrame(const builtin_interfaces::msg::Time &stamp);
  bool HasPinnedStartFrame() const;
  void ReleasePinnedStartFrame();
  std::optional<std::int64_t> LatestColorStampNs() const;
  std::optional<std::int64_t> LatestDepthStampNs() const;
  std::optional<std::int64_t> PinnedColorStampNs() const;
  std::optional<std::int64_t> PinnedDepthStampNs() const;

  bool CopyColor(cv::Mat *image, m3t::Intrinsics *intrinsics) const;
  bool CopyDepth(cv::Mat *image, m3t::Intrinsics *intrinsics,
                 float *depth_scale) const;
  bool HasColor() const;
  bool HasDepth() const;

 private:
  struct StampedColorFrame {
    std::int64_t stamp_ns = 0;
    cv::Mat image;
  };

  struct StampedDepthFrame {
    std::int64_t stamp_ns = 0;
    cv::Mat image;
    float depth_scale = 0.001f;
  };

  static m3t::Intrinsics ToIntrinsics(const sensor_msgs::msg::CameraInfo &msg);

  mutable std::mutex mutex_;
  cv::Mat color_image_;
  cv::Mat depth_image_;
  std::deque<StampedColorFrame> color_history_;
  std::deque<StampedDepthFrame> depth_history_;
  std::optional<m3t::Intrinsics> color_intrinsics_;
  std::optional<m3t::Intrinsics> depth_intrinsics_;
  std::optional<StampedColorFrame> pinned_color_;
  std::optional<StampedDepthFrame> pinned_depth_;
  std::chrono::nanoseconds start_frame_tolerance_{std::chrono::milliseconds{10}};
  std::size_t max_history_size_ = 300;
  float depth_scale_ = 0.001f;
};

class RosColorCamera : public m3t::ColorCamera {
 public:
  RosColorCamera(const std::string &name,
                 const std::shared_ptr<RosFrameBuffer> &buffer);
  bool SetUp() override;
  bool UpdateImage(bool synchronized) override;

 private:
  std::shared_ptr<RosFrameBuffer> buffer_;
};

class RosDepthCamera : public m3t::DepthCamera {
 public:
  RosDepthCamera(const std::string &name,
                 const std::shared_ptr<RosFrameBuffer> &buffer);
  bool SetUp() override;
  bool UpdateImage(bool synchronized) override;

 private:
  std::shared_ptr<RosFrameBuffer> buffer_;
};

}  // namespace rcia_energy_trace_ros
