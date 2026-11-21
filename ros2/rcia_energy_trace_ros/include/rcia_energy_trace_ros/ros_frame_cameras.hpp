#pragma once

#include <cv_bridge/cv_bridge.h>
#include <m3t/camera.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>

namespace rcia_energy_trace_ros {

// Immutable after it is handed out. Image data is shared with the history
// frames, which are never modified in place, so copying a pair is cheap.
struct FramePair {
  std::int64_t color_stamp_ns = 0;
  std::int64_t depth_stamp_ns = 0;
  cv::Mat color;
  cv::Mat depth;
  m3t::Intrinsics color_intrinsics{};
  m3t::Intrinsics depth_intrinsics{};
  float depth_scale = 0.001f;
};

enum class FrameSelectCode {
  Ready,
  Waiting,
  HistoryExpired,
  GapTooLarge,
  TimeReset,
  CameraChanged,
};

struct FrameSelection {
  FrameSelectCode code = FrameSelectCode::Waiting;
  std::shared_ptr<const FramePair> pair;
  std::size_t skipped_color_frames = 0;
  std::uint64_t timeline_epoch = 0;
  std::uint64_t calibration_epoch = 0;
};

class RosFrameBuffer {
 public:
  void SetColorCameraInfo(const sensor_msgs::msg::CameraInfo &msg);
  void SetDepthCameraInfo(const sensor_msgs::msg::CameraInfo &msg);
  void SetColorImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void SetDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void SetMaxHistorySize(std::size_t max_history_size);
  void SetStartFrameTolerance(std::chrono::nanoseconds tolerance);
  void SetPairTolerance(std::chrono::nanoseconds tolerance);
  void SetMaxFrameGap(std::chrono::nanoseconds max_gap);
  // When true, CopyColor and CopyDepth fail instead of falling back to the
  // newest history frame if no active pair has been set.
  void SetRequireActivePair(bool require_active_pair);

  FrameSelection SelectStartPair(std::int64_t target_ns) const;
  FrameSelection SelectNextPair(std::int64_t after_color_ns,
                                std::int64_t after_depth_ns,
                                std::uint64_t expected_timeline_epoch,
                                std::uint64_t expected_calibration_epoch) const;
  FrameSelection SelectLatestPair(std::int64_t after_color_ns,
                                  std::int64_t after_depth_ns,
                                  std::uint64_t expected_timeline_epoch,
                                  std::uint64_t expected_calibration_epoch) const;

  void ActivatePair(const std::shared_ptr<const FramePair> &pair);
  std::shared_ptr<const FramePair> ActivePair() const;
  void ClearActivePair();

  std::optional<std::int64_t> LatestColorStampNs() const;
  std::optional<std::int64_t> LatestDepthStampNs() const;
  std::optional<std::int64_t> LatestCompleteColorStampNs() const;
  void ResetHistory();

  // Blocks until a new frame or timeline/calibration change arrives, the timeout
  // expires, or the flag is set and WakeWaiters() is called. Returns true only
  // when new data arrived.
  bool WaitForFrames(std::chrono::milliseconds timeout,
                     const std::atomic<bool> &cancelled);
  void WakeWaiters();

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
  static bool SameIntrinsics(const m3t::Intrinsics &lhs,
                             const m3t::Intrinsics &rhs);

  void ClearHistoriesLocked();
  void ResetTimelineLocked();
  void ChangeCalibrationLocked();
  void NotifyLocked();
  FrameSelection BaseSelectionLocked() const;
  const StampedDepthFrame *FindDepthLocked(std::int64_t color_ns,
                                           std::int64_t after_depth_ns) const;
  std::shared_ptr<const FramePair> MakePairLocked(
      const StampedColorFrame &color, const StampedDepthFrame &depth) const;
  bool HasIntrinsicsLocked() const;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<StampedColorFrame> color_history_;
  std::deque<StampedDepthFrame> depth_history_;
  std::optional<m3t::Intrinsics> color_intrinsics_;
  std::optional<m3t::Intrinsics> depth_intrinsics_;
  std::shared_ptr<const FramePair> active_pair_;
  std::chrono::nanoseconds start_frame_tolerance_{std::chrono::milliseconds{10}};
  std::chrono::nanoseconds pair_tolerance_{std::chrono::milliseconds{10}};
  std::chrono::nanoseconds max_frame_gap_{std::chrono::milliseconds{100}};
  std::size_t max_history_size_ = 300;
  bool require_active_pair_ = false;
  std::uint64_t timeline_epoch_ = 0;
  std::uint64_t calibration_epoch_ = 0;
  std::uint64_t frame_version_ = 0;
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
