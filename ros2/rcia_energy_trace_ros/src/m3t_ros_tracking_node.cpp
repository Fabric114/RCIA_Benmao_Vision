#include "rcia_energy_trace_ros/axis_normal_viewer.hpp"
#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"
#include "rcia_energy_trace_ros/tracking_session.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/silhouette_renderer.h>
#include <m3t/static_detector.h>
#include <m3t/tracker.h>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>

#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

#ifndef RCIA_PROJECT_ROOT
#define RCIA_PROJECT_ROOT "."
#endif

constexpr std::int64_t kCatchingUpStatusPeriodNs = 250000000LL;
constexpr char kPoseFrameId[] = "camera_color_optical_frame";

m3t::Transform3fA PoseMsgToTransform(const geometry_msgs::msg::Pose &pose) {
  Eigen::Quaternionf q{
      static_cast<float>(pose.orientation.w),
      static_cast<float>(pose.orientation.x),
      static_cast<float>(pose.orientation.y),
      static_cast<float>(pose.orientation.z),
  };
  q.normalize();

  m3t::Transform3fA transform = m3t::Transform3fA::Identity();
  transform.linear() = q.toRotationMatrix();
  transform.translation() =
      Eigen::Vector3f{static_cast<float>(pose.position.x),
                      static_cast<float>(pose.position.y),
                      static_cast<float>(pose.position.z)};
  return transform;
}

geometry_msgs::msg::Pose TransformToPoseMsg(const m3t::Transform3fA &transform) {
  Eigen::Quaternionf q{transform.linear()};
  q.normalize();
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

std::int64_t StampToNanoseconds(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time NanosecondsToStamp(std::int64_t stamp_ns) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(stamp_ns / 1000000000LL);
  stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % 1000000000LL);
  return stamp;
}

std::int64_t MsToNs(double ms) {
  return static_cast<std::int64_t>(std::llround(ms * 1000000.0));
}

std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

const char *PhaseName(rcia_energy_trace_ros::SessionPhase phase) {
  switch (phase) {
    case rcia_energy_trace_ros::SessionPhase::Starting:
      return "starting";
    case rcia_energy_trace_ros::SessionPhase::CatchingUp:
      return "catching_up";
    case rcia_energy_trace_ros::SessionPhase::Tracking:
      return "tracking";
  }
  return "starting";
}

void RequireNonNegative(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must be a non-negative finite value");
  }
}

bool IsValidInitialPose(const geometry_msgs::msg::PoseStamped &msg) {
  if (msg.header.frame_id != kPoseFrameId) return false;
  const auto &p = msg.pose.position;
  const auto &q = msg.pose.orientation;
  const double values[] = {p.x, p.y, p.z, q.x, q.y, q.z, q.w};
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  const double norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  return norm_sq > 1e-12;
}

bool IsValidBodyPose(const m3t::Transform3fA &pose) {
  if (!pose.matrix().allFinite()) return false;
  const Eigen::Matrix3f rotation = pose.linear();
  const float orthogonality_error =
      (rotation.transpose() * rotation - Eigen::Matrix3f::Identity()).norm();
  return orthogonality_error < 1e-3f && std::abs(rotation.determinant() - 1.0f) < 1e-3f;
}

struct SessionSnapshot {
  m3t::Transform3fA initial_pose;
  std::int64_t stamp_ns = 0;
};

}  // namespace

class M3TRosTrackingNode : public rclcpp::Node {
 public:
  M3TRosTrackingNode()
      : Node{"m3t_ros_tracking_node"},
        frame_buffer_{std::make_shared<rcia_energy_trace_ros::RosFrameBuffer>()} {
    color_topic_ = declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/aligned_depth_to_color/image_raw");
    color_camera_info_topic_ = declare_parameter<std::string>(
        "color_camera_info_topic", "/camera/color/camera_info");
    depth_camera_info_topic_ = declare_parameter<std::string>(
        "depth_camera_info_topic", "/camera/depth/camera_info");
    initial_pose_topic_ =
        declare_parameter<std::string>("initial_pose_topic", "/energy/initial_pose");
    visualize_ = declare_parameter<bool>("visualize", true);
    show_m3t_window_ = declare_parameter<bool>("show_m3t_window", false);
    diagnostic_overlay_topic_ = declare_parameter<std::string>(
        "diagnostic_overlay_topic", "/energy/diagnostics/m3t_overlay");
    m3t_pose_topic_ =
        declare_parameter<std::string>("m3t_pose_topic", "/energy/m3t_pose");
    reset_pose_topic_ =
        declare_parameter<std::string>("reset_pose_topic", "/energy/m3t_reset_pose");
    stop_topic_ =
        declare_parameter<std::string>("m3t_stop_topic", "/energy/m3t_stop");
    status_topic_ =
        declare_parameter<std::string>("m3t_status_topic", "/energy/m3t_status");
    enable_pose_resets_ = declare_parameter<bool>("enable_pose_resets", true);
    timing_diagnostics_ = declare_parameter<bool>("timing_diagnostics", true);
    timing_topic_ =
        declare_parameter<std::string>("timing_topic", "/energy/timing_events");
    timing_overlay_stride_ =
        std::max<int>(1, declare_parameter<int>("timing_overlay_stride", 30));

    const auto frame_history_size = declare_parameter<int>("frame_history_size", 300);
    const auto start_frame_tolerance_ms =
        declare_parameter<double>("start_frame_tolerance_ms", 10.0);
    const auto rgbd_pair_tolerance_ms =
        declare_parameter<double>("rgbd_pair_tolerance_ms", 10.0);
    const auto max_frame_gap_ms = declare_parameter<double>("max_frame_gap_ms", 100.0);
    session_params_.start_frame_wait_ms =
        declare_parameter<double>("start_frame_wait_ms", 1000.0);
    session_params_.replay_timeout_ms = declare_parameter<double>("replay_timeout_ms", 2000.0);
    session_params_.replay_target_lag_ms =
        declare_parameter<double>("replay_target_lag_ms", 75.0);
    session_params_.max_output_pose_age_ms =
        declare_parameter<double>("max_output_pose_age_ms", 250.0);
    session_params_.live_frame_wait_ms = declare_parameter<double>("live_frame_wait_ms", 500.0);

    if (frame_history_size <= 0) {
      throw std::invalid_argument("frame_history_size must be positive");
    }
    RequireNonNegative(start_frame_tolerance_ms, "start_frame_tolerance_ms");
    RequireNonNegative(rgbd_pair_tolerance_ms, "rgbd_pair_tolerance_ms");
    RequireNonNegative(max_frame_gap_ms, "max_frame_gap_ms");
    rcia_energy_trace_ros::ValidateSessionParams(session_params_);

    frame_buffer_->SetMaxHistorySize(static_cast<std::size_t>(frame_history_size));
    frame_buffer_->SetStartFrameTolerance(std::chrono::nanoseconds{MsToNs(start_frame_tolerance_ms)});
    frame_buffer_->SetPairTolerance(std::chrono::nanoseconds{MsToNs(rgbd_pair_tolerance_ms)});
    frame_buffer_->SetMaxFrameGap(std::chrono::nanoseconds{MsToNs(max_frame_gap_ms)});
    frame_buffer_->SetRequireActivePair(true);

    color_sub_ = create_subscription<sensor_msgs::msg::Image>(
        color_topic_, 10,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          frame_buffer_->SetColorImage(msg);
        });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, 10,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          frame_buffer_->SetDepthImage(msg);
        });
    color_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        color_camera_info_topic_, 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          frame_buffer_->SetColorCameraInfo(*msg);
        });
    depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        depth_camera_info_topic_, 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          frame_buffer_->SetDepthCameraInfo(*msg);
        });
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        initial_pose_topic_, 10,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          HandleInitialPose(*msg);
        });
    reset_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        reset_pose_topic_, 10,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          HandleResetPose(*msg);
        });
    stop_sub_ = create_subscription<std_msgs::msg::Empty>(
        stop_topic_, 10,
        [this](const std_msgs::msg::Empty::SharedPtr) { HandleStop(); });
    m3t_overlay_pub_ =
        create_publisher<sensor_msgs::msg::Image>(diagnostic_overlay_topic_, 10);
    m3t_pose_pub_ =
        create_publisher<geometry_msgs::msg::PoseStamped>(m3t_pose_topic_, 10);
    timing_pub_ = create_publisher<std_msgs::msg::String>(timing_topic_, 50);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, 20);
    PublishStatus("stopped", now().nanoseconds(), "node ready");

    RCLCPP_INFO(get_logger(),
                "M3T ROS tracking node waiting for RGB-D camera_info and initial pose");
  }

  ~M3TRosTrackingNode() override { CancelSession(); }

 private:
  void HandleInitialPose(const geometry_msgs::msg::PoseStamped &msg) {
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      if (session_active_) {
        RCLCPP_WARN(get_logger(), "Ignoring initial pose while an M3T session is active");
        return;
      }
    }
    const auto stamp_ns = StampToNanoseconds(msg.header.stamp);
    if (!IsValidInitialPose(msg)) {
      RejectInvalidPose("initial", msg);
      return;
    }
    RCLCPP_INFO(get_logger(),
                "Received FoundationPose initial pose in frame '%s' at %d.%09u",
                msg.header.frame_id.c_str(), msg.header.stamp.sec,
                msg.header.stamp.nanosec);
    have_initial_pose_.store(true);
    PublishStatus("starting", stamp_ns, "initial pose received",
                  InitialPoseProgress(stamp_ns));
    PublishTiming("m3t_initial_pose_received", stamp_ns, std::nullopt,
                  "\"frame_id\":\"" + msg.header.frame_id + "\"");
    StartSession(SessionSnapshot{PoseMsgToTransform(msg.pose), stamp_ns});
  }

  void HandleResetPose(const geometry_msgs::msg::PoseStamped &msg) {
    if (!enable_pose_resets_) return;
    const auto stamp_ns = StampToNanoseconds(msg.header.stamp);
    if (!IsValidInitialPose(msg)) {
      bool active = false;
      {
        const std::lock_guard<std::mutex> lock{session_mutex_};
        active = session_active_;
      }
      if (active) {
        RCLCPP_ERROR(get_logger(),
                     "Ignoring invalid M3T reset pose in frame '%s' while session is active",
                     msg.header.frame_id.c_str());
      } else {
        RejectInvalidPose("reset", msg);
      }
      return;
    }
    RCLCPP_WARN(get_logger(), "Received M3T reset pose in frame '%s' at %d.%09u",
                msg.header.frame_id.c_str(), msg.header.stamp.sec,
                msg.header.stamp.nanosec);
    PublishStatus("starting", stamp_ns, "reset pose received", InitialPoseProgress(stamp_ns));
    PublishTiming("m3t_reset_pose_received", stamp_ns, std::nullopt,
                  "\"frame_id\":\"" + msg.header.frame_id + "\"");
    CancelSession();
    have_initial_pose_.store(true);
    StartSession(SessionSnapshot{PoseMsgToTransform(msg.pose), stamp_ns});
  }

  void HandleStop() {
    const auto frame_stamp_ns =
        frame_buffer_->LatestColorStampNs().value_or(now().nanoseconds());
    std::int64_t active_initial_stamp_ns = 0;
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      active_initial_stamp_ns = active_initial_pose_stamp_ns_;
    }
    RCLCPP_WARN(get_logger(), "Received M3T stop request; clearing active pose");
    PublishStatus("stopping", frame_stamp_ns, "stop request received",
                  InitialPoseProgress(active_initial_stamp_ns));
    CancelSession();
    have_initial_pose_.store(false);
    overlay_frame_count_.store(0);
    PublishStatus("stopped", frame_stamp_ns, "stop complete",
                  InitialPoseProgress(active_initial_stamp_ns));
    PublishTiming("m3t_stop_received", frame_stamp_ns);
  }

  void RejectInvalidPose(const char *which, const geometry_msgs::msg::PoseStamped &msg) {
    const auto stamp_ns = StampToNanoseconds(msg.header.stamp);
    RCLCPP_ERROR(get_logger(), "Rejected invalid M3T %s pose in frame '%s'", which,
                 msg.header.frame_id.c_str());
    PublishStatus("setup_failed", stamp_ns, "invalid_pose", InitialPoseProgress(stamp_ns));
    PublishStatus("stopped", stamp_ns, "invalid_pose", InitialPoseProgress(stamp_ns));
  }

  // Joins the finished worker before starting a new one. Callbacks run on one
  // executor thread, so this does not race with other callbacks.
  void StartSession(const SessionSnapshot &snapshot) {
    std::thread finished_worker;
    std::uint64_t generation = 0;
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      if (session_active_) return;
      finished_worker = std::move(tracker_thread_);
      generation = ++session_generation_;
      session_active_ = true;
      active_initial_pose_stamp_ns_ = snapshot.stamp_ns;
      tracker_thread_ = std::thread([this, snapshot, generation]() {
        RunSession(snapshot, generation);
      });
    }
    if (finished_worker.joinable()) finished_worker.join();
  }

  // Bumps the generation so the worker cannot publish or register anymore, then
  // cancels and joins it. Callers must not hold session_mutex_.
  void CancelSession() {
    std::shared_ptr<rcia_energy_trace_ros::TrackingSession> session;
    std::thread worker;
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      ++session_generation_;
      session = session_ptr_;
      worker = std::move(tracker_thread_);
    }
    if (session) session->Cancel();
    frame_buffer_->WakeWaiters();
    if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) {
      worker.join();
    }
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      session_ptr_.reset();
      session_active_ = false;
      active_initial_pose_stamp_ns_ = 0;
      tracking_started_.store(false);
    }
    frame_buffer_->ClearActivePair();
  }

  bool IsCurrentSession(std::uint64_t generation) {
    const std::lock_guard<std::mutex> lock{session_mutex_};
    return generation == session_generation_;
  }

  void RunSession(const SessionSnapshot snapshot, std::uint64_t generation) {
    rcia_energy_trace_ros::SessionResult result;
    try {
      result = RunTracker(snapshot, generation);
    } catch (const std::exception &error) {
      RCLCPP_ERROR(get_logger(), "M3T session raised an exception: %s", error.what());
      result.reason = "tracker_error";
    }
    FinishSession(snapshot, generation, result);
  }

  // The single exit point for a worker. A session taken over by stop or reset
  // (generation mismatch) publishes nothing here.
  void FinishSession(const SessionSnapshot &snapshot, std::uint64_t generation,
                     const rcia_energy_trace_ros::SessionResult &result) {
    const std::lock_guard<std::mutex> lock{session_mutex_};
    if (generation != session_generation_) return;
    session_ptr_.reset();
    tracking_started_.store(false);
    if (!result.cancelled) {
      RCLCPP_ERROR(get_logger(), "M3T tracking session failed: %s", result.reason.c_str());
      PublishStatus("setup_failed", snapshot.stamp_ns, result.reason, result.progress);
    }
    PublishStatus("stopped", snapshot.stamp_ns,
                  result.reason.empty() ? "tracker session ended" : result.reason,
                  result.progress);
    frame_buffer_->ClearActivePair();
    session_active_ = false;
    active_initial_pose_stamp_ns_ = 0;
  }

  void PublishM3TPose(const m3t::Transform3fA &pose, std::int64_t frame_stamp_ns) {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = NanosecondsToStamp(frame_stamp_ns);
    msg.header.frame_id = kPoseFrameId;
    msg.pose = TransformToPoseMsg(pose);
    m3t_pose_pub_->publish(msg);
  }

  void PublishTiming(const std::string &event, std::int64_t frame_stamp_ns,
                     std::optional<std::int64_t> steady_started_ns = std::nullopt,
                     const std::string &extra_json_fields = "") {
    if (!timing_diagnostics_) return;
    const auto node_now_ns = now().nanoseconds();
    std::ostringstream out;
    out << "{\"age_ms\":" << static_cast<double>(node_now_ns - frame_stamp_ns) / 1000000.0
        << ",\"event\":\"" << event << "\""
        << ",\"frame_stamp_ns\":" << frame_stamp_ns
        << ",\"node_now_ns\":" << node_now_ns;
    if (steady_started_ns) {
      out << ",\"duration_ms\":"
          << static_cast<double>(SteadyNowNs() - *steady_started_ns) / 1000000.0;
    }
    if (!extra_json_fields.empty()) out << "," << extra_json_fields;
    out << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    timing_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "TIMING %s", msg.data.c_str());
  }

  void PublishStatus(const std::string &state, std::int64_t stamp_ns,
                     const std::string &reason = "",
                     const rcia_energy_trace_ros::SessionProgress &progress = {}) {
    if (!status_pub_) return;
    std::ostringstream out;
    out << "{\"have_initial_pose\":" << (have_initial_pose_.load() ? "true" : "false")
        << ",\"initial_pose_stamp_ns\":" << progress.initial_pose_stamp_ns
        << ",\"latest_frame_stamp_ns\":" << progress.latest_frame_stamp_ns
        << ",\"processed_frame_stamp_ns\":" << progress.processed_frame_stamp_ns
        << ",\"reason\":\"" << EscapeJson(reason) << "\""
        << ",\"replayed_frames\":" << progress.replayed_frames
        << ",\"stamp_ns\":" << stamp_ns
        << ",\"state\":\"" << EscapeJson(state) << "\""
        << ",\"tracking_started\":"
        << (tracking_started_.load() ? "true" : "false") << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    status_pub_->publish(msg);
  }

  static rcia_energy_trace_ros::SessionProgress InitialPoseProgress(std::int64_t stamp_ns) {
    rcia_energy_trace_ros::SessionProgress progress;
    progress.initial_pose_stamp_ns = stamp_ns;
    return progress;
  }

  static std::string EscapeJson(const std::string &value) {
    std::ostringstream out;
    for (const char c : value) {
      if (c == '\\' || c == '"') {
        out << '\\' << c;
      } else if (c == '\n') {
        out << "\\n";
      } else {
        out << c;
      }
    }
    return out.str();
  }

  static void AppendOptionalField(std::ostringstream *out, const std::string &name,
                                  const std::optional<std::int64_t> &value) {
    *out << ",\"" << name << "\":";
    if (value) {
      *out << *value;
    } else {
      *out << "null";
    }
  }

  // Runs on the worker thread. Builds M3T objects, then runs the session.
  rcia_energy_trace_ros::SessionResult RunTracker(const SessionSnapshot &snapshot,
                                                  std::uint64_t generation) {
    const std::filesystem::path project_root =
        std::filesystem::weakly_canonical(RCIA_PROJECT_ROOT);
    const std::filesystem::path body_yaml_path =
        project_root / "models/m3t/Str3D_body.yaml";
    const std::filesystem::path model_directory = project_root / "models/m3t";
    const std::string body_name = "Str3D";

    auto tracker_ptr = std::make_shared<m3t::Tracker>("tracker");
    auto renderer_geometry_ptr =
        std::make_shared<m3t::RendererGeometry>("renderer geometry");
    auto color_camera_ptr =
        std::make_shared<rcia_energy_trace_ros::RosColorCamera>("ros_color", frame_buffer_);
    auto depth_camera_ptr =
        std::make_shared<rcia_energy_trace_ros::RosDepthCamera>("ros_depth", frame_buffer_);

    auto color_depth_renderer_ptr =
        std::make_shared<m3t::FocusedBasicDepthRenderer>(
            "color_depth_renderer", renderer_geometry_ptr, color_camera_ptr);
    auto depth_depth_renderer_ptr =
        std::make_shared<m3t::FocusedBasicDepthRenderer>(
            "depth_depth_renderer", renderer_geometry_ptr, depth_camera_ptr);
    auto color_silhouette_renderer_ptr =
        std::make_shared<m3t::FocusedSilhouetteRenderer>(
            "color_silhouette_renderer", renderer_geometry_ptr, color_camera_ptr);

    (void)color_depth_renderer_ptr;
    (void)depth_depth_renderer_ptr;
    (void)color_silhouette_renderer_ptr;

    auto body_ptr = std::make_shared<m3t::Body>(body_name, body_yaml_path);
    renderer_geometry_ptr->AddBody(body_ptr);

    int current_iteration = 0;
    if (visualize_) {
      auto color_viewer_ptr =
          std::make_shared<rcia_energy_trace_ros::AxisNormalColorViewer>(
              "color_viewer", color_camera_ptr, renderer_geometry_ptr, body_ptr,
              0.5f, 0.06f);
      color_viewer_ptr->set_display_images(show_m3t_window_);
      color_viewer_ptr->set_image_callback([this](const cv::Mat &image) {
        const auto active_pair = frame_buffer_->ActivePair();
        const std::int64_t frame_stamp_ns =
            active_pair ? active_pair->color_stamp_ns : now().nanoseconds();
        std_msgs::msg::Header header;
        header.stamp = NanosecondsToStamp(frame_stamp_ns);
        header.frame_id = kPoseFrameId;
        auto msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        m3t_overlay_pub_->publish(*msg);
        const int overlay_count = ++overlay_frame_count_;
        if (overlay_count % timing_overlay_stride_ == 0) {
          const auto latest_depth_stamp = frame_buffer_->LatestDepthStampNs();
          std::ostringstream extra;
          extra << "\"overlay_count\":" << overlay_count;
          AppendOptionalField(&extra, "latest_depth_stamp_ns", latest_depth_stamp);
          AppendOptionalField(&extra, "pose_header_stamp_ns", frame_stamp_ns);
          extra << ",\"pose_publish_age_ms\":"
                << static_cast<double>(now().nanoseconds() - frame_stamp_ns) / 1000000.0;
          if (active_pair) {
            extra << ",\"rgb_depth_delta_ms\":"
                  << static_cast<double>(
                         (active_pair->color_stamp_ns > active_pair->depth_stamp_ns)
                             ? (active_pair->color_stamp_ns - active_pair->depth_stamp_ns)
                             : (active_pair->depth_stamp_ns - active_pair->color_stamp_ns)) /
                         1000000.0;
          }
          PublishTiming("m3t_overlay_pose_published", frame_stamp_ns, std::nullopt,
                        extra.str());
        }
      });
      tracker_ptr->AddViewer(color_viewer_ptr);
      tracker_ptr->set_viewer_time(1);
    }

    auto region_model_ptr = std::make_shared<m3t::RegionModel>(
        body_name + "_region_model", body_ptr, model_directory / "Str3D_region.bin");
    auto depth_model_ptr = std::make_shared<m3t::DepthModel>(
        body_name + "_depth_model", body_ptr, model_directory / "Str3D_depth.bin");

    auto region_modality_ptr = std::make_shared<m3t::RegionModality>(
        body_name + "_region_modality", body_ptr, color_camera_ptr, region_model_ptr);
    auto depth_modality_ptr = std::make_shared<m3t::DepthModality>(
        body_name + "_depth_modality", body_ptr, depth_camera_ptr, depth_model_ptr);

    region_modality_ptr->set_visualize_pose_result(show_m3t_window_);
    region_modality_ptr->MeasureOcclusions(depth_camera_ptr);
    depth_modality_ptr->MeasureOcclusions();

    auto link_ptr = std::make_shared<m3t::Link>(body_name + "_link", body_ptr);
    link_ptr->AddModality(region_modality_ptr);
    link_ptr->AddModality(depth_modality_ptr);

    auto optimizer_ptr =
        std::make_shared<m3t::Optimizer>(body_name + "_optimizer", link_ptr);
    tracker_ptr->AddOptimizer(optimizer_ptr);

    body_ptr->set_body2world_pose(snapshot.initial_pose);
    link_ptr->set_link2world_pose(snapshot.initial_pose);
    auto detector_ptr = std::make_shared<m3t::StaticDetector>(
        body_name + "_detector", optimizer_ptr, snapshot.initial_pose, true);
    tracker_ptr->AddDetector(detector_ptr);

    rcia_energy_trace_ros::SessionHooks hooks;
    std::optional<rcia_energy_trace_ros::SessionPhase> last_phase;
    std::int64_t last_status_steady_ns = 0;

    hooks.setup = [this, tracker_ptr, snapshot]() {
      const auto setup_started_ns = SteadyNowNs();
      RCLCPP_INFO(get_logger(), "Starting M3T ROS session on pinned RGB-D frame");
      if (!tracker_ptr->SetUp()) {
        RCLCPP_ERROR(get_logger(), "M3T ROS tracking setup failed");
        return false;
      }
      PublishTiming("m3t_setup_done", snapshot.stamp_ns, setup_started_ns);
      return true;
    };
    hooks.start = [tracker_ptr]() {
      tracker_ptr->StartTracking();
      return tracker_ptr->UpdateCameras(0) && tracker_ptr->UpdateSubscribers(0) &&
             tracker_ptr->CalculateConsistentPoses() &&
             tracker_ptr->ExecuteStartingStep(0);
    };
    hooks.track = [tracker_ptr, &current_iteration](int iteration) {
      current_iteration = iteration;
      if (iteration == 0) {
        // start() already copied the t0 images and ran the starting step.
        return tracker_ptr->ExecuteTrackingStep(0) && tracker_ptr->UpdatePublishers(0);
      }
      return tracker_ptr->UpdateCameras(iteration) &&
             tracker_ptr->UpdateSubscribers(iteration) &&
             tracker_ptr->CalculateConsistentPoses() &&
             tracker_ptr->ExecuteTrackingStep(iteration) &&
             tracker_ptr->UpdatePublishers(iteration);
    };
    hooks.pose_valid = [body_ptr]() {
      return IsValidBodyPose(body_ptr->body2world_pose());
    };
    hooks.publish_live = [this, generation, tracker_ptr, body_ptr,
                          &current_iteration](const rcia_energy_trace_ros::FramePair &pair) {
      if (!IsCurrentSession(generation)) return;
      PublishM3TPose(body_ptr->body2world_pose(), pair.color_stamp_ns);
      if (visualize_) tracker_ptr->UpdateViewers(current_iteration);
    };
    hooks.report_phase = [this, generation, snapshot, &last_phase, &last_status_steady_ns](
                             rcia_energy_trace_ros::SessionPhase phase,
                             const rcia_energy_trace_ros::SessionProgress &progress) {
      if (!IsCurrentSession(generation)) return;
      const auto now_steady_ns = SteadyNowNs();
      const bool phase_changed = !last_phase || *last_phase != phase;
      if (!phase_changed && phase == rcia_energy_trace_ros::SessionPhase::CatchingUp &&
          now_steady_ns - last_status_steady_ns < kCatchingUpStatusPeriodNs) {
        return;
      }
      last_phase = phase;
      last_status_steady_ns = now_steady_ns;
      tracking_started_.store(phase == rcia_energy_trace_ros::SessionPhase::Tracking);
      PublishStatus(PhaseName(phase), snapshot.stamp_ns, "", progress);
    };
    hooks.ros_now_ns = [this]() { return now().nanoseconds(); };
    hooks.monotonic_now_ns = []() { return SteadyNowNs(); };

    auto session = std::make_shared<rcia_energy_trace_ros::TrackingSession>(
        frame_buffer_, session_params_, std::move(hooks));
    {
      const std::lock_guard<std::mutex> lock{session_mutex_};
      if (generation != session_generation_) {
        rcia_energy_trace_ros::SessionResult cancelled;
        cancelled.cancelled = true;
        cancelled.reason = "cancelled";
        return cancelled;
      }
      session_ptr_ = session;
    }
    return session->Run(snapshot.stamp_ns);
  }

  std::shared_ptr<rcia_energy_trace_ros::RosFrameBuffer> frame_buffer_;
  rcia_energy_trace_ros::SessionParams session_params_;
  std::mutex session_mutex_;
  std::thread tracker_thread_;
  std::uint64_t session_generation_ = 0;
  std::int64_t active_initial_pose_stamp_ns_ = 0;
  bool session_active_ = false;
  std::shared_ptr<rcia_energy_trace_ros::TrackingSession> session_ptr_;
  std::atomic<bool> tracking_started_{false};
  std::atomic<bool> have_initial_pose_{false};
  bool visualize_ = true;
  bool show_m3t_window_ = false;
  bool enable_pose_resets_ = true;
  bool timing_diagnostics_ = true;
  int timing_overlay_stride_ = 30;
  std::atomic<int> overlay_frame_count_{0};

  std::string color_topic_;
  std::string depth_topic_;
  std::string color_camera_info_topic_;
  std::string depth_camera_info_topic_;
  std::string initial_pose_topic_;
  std::string diagnostic_overlay_topic_;
  std::string m3t_pose_topic_;
  std::string reset_pose_topic_;
  std::string stop_topic_;
  std::string status_topic_;
  std::string timing_topic_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr reset_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m3t_overlay_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m3t_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr timing_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<M3TRosTrackingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
