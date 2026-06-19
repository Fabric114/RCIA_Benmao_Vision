#include "rcia_energy_trace_ros/axis_normal_viewer.hpp"
#include "rcia_energy_trace_ros/ros_frame_cameras.hpp"

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
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace {

#ifndef RCIA_PROJECT_ROOT
#define RCIA_PROJECT_ROOT "."
#endif

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
    const auto frame_history_size =
        std::max<long>(1, declare_parameter<int>("frame_history_size", 300));
    frame_buffer_->SetMaxHistorySize(static_cast<std::size_t>(frame_history_size));
    const auto start_frame_tolerance_ms =
        declare_parameter<double>("start_frame_tolerance_ms", 10.0);
    frame_buffer_->SetStartFrameTolerance(std::chrono::nanoseconds{
        static_cast<std::int64_t>(start_frame_tolerance_ms * 1000000.0)});

    color_sub_ = create_subscription<sensor_msgs::msg::Image>(
        color_topic_, 10,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          frame_buffer_->SetColorImage(msg);
          TryStartTracking();
        });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, 10,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          frame_buffer_->SetDepthImage(msg);
          TryStartTracking();
        });
    color_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        color_camera_info_topic_, 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          frame_buffer_->SetColorCameraInfo(*msg);
          TryStartTracking();
        });
    depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        depth_camera_info_topic_, 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          frame_buffer_->SetDepthCameraInfo(*msg);
          TryStartTracking();
        });
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        initial_pose_topic_, 10,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          initial_pose_ = PoseMsgToTransform(msg->pose);
          initial_pose_stamp_ = msg->header.stamp;
          have_initial_pose_ = true;
          RCLCPP_INFO(get_logger(),
                      "Received FoundationPose initial pose in frame '%s' at %d.%09u",
                      msg->header.frame_id.c_str(), msg->header.stamp.sec,
                      msg->header.stamp.nanosec);
          PublishStatus("starting", StampToNanoseconds(msg->header.stamp),
                        "initial pose received");
          PublishTiming("m3t_initial_pose_received",
                        StampToNanoseconds(msg->header.stamp), std::nullopt,
                        "\"frame_id\":\"" + msg->header.frame_id + "\"");
          TryStartTracking();
        });
    reset_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        reset_pose_topic_, 10,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          if (!enable_pose_resets_) return;
          initial_pose_ = PoseMsgToTransform(msg->pose);
          initial_pose_stamp_ = msg->header.stamp;
          have_initial_pose_ = true;
          RCLCPP_WARN(get_logger(),
                      "Received M3T reset pose in frame '%s' at %d.%09u",
                      msg->header.frame_id.c_str(), msg->header.stamp.sec,
                      msg->header.stamp.nanosec);
          PublishStatus("starting", StampToNanoseconds(msg->header.stamp),
                        "reset pose received");
          PublishTiming("m3t_reset_pose_received",
                        StampToNanoseconds(msg->header.stamp), std::nullopt,
                        "\"frame_id\":\"" + msg->header.frame_id + "\"");
          StopTrackerThread();
          tracking_started_.store(false);
          TryStartTracking();
        });
    stop_sub_ = create_subscription<std_msgs::msg::Empty>(
        stop_topic_, 10,
        [this](const std_msgs::msg::Empty::SharedPtr) {
          const auto frame_stamp_ns =
              frame_buffer_->LatestColorStampNs().value_or(now().nanoseconds());
          RCLCPP_WARN(get_logger(), "Received M3T stop request; clearing active pose");
          PublishStatus("stopping", frame_stamp_ns, "stop request received");
          StopTrackerThread();
          tracking_started_.store(false);
          have_initial_pose_ = false;
          overlay_frame_count_.store(0);
          PublishStatus("stopped", frame_stamp_ns, "stop complete");
          PublishTiming("m3t_stop_received", frame_stamp_ns);
        });
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

  ~M3TRosTrackingNode() override {
    StopTrackerThread();
  }

 private:
  void StopTrackerThread() {
    std::shared_ptr<m3t::Tracker> tracker;
    {
      const std::lock_guard<std::mutex> lock{tracker_mutex_};
      tracker = tracker_ptr_;
    }
    if (tracker) tracker->QuitTrackerProcess();
    if (tracker_thread_.joinable() &&
        tracker_thread_.get_id() != std::this_thread::get_id()) {
      tracker_thread_.join();
    }
    {
      const std::lock_guard<std::mutex> lock{tracker_mutex_};
      tracker_ptr_.reset();
    }
    frame_buffer_->ReleasePinnedStartFrame();
  }

  void PublishM3TPose(const m3t::Transform3fA &pose) {
    geometry_msgs::msg::PoseStamped msg;
    const auto latest_color_stamp = frame_buffer_->LatestColorStampNs();
    if (latest_color_stamp) {
      msg.header.stamp = NanosecondsToStamp(*latest_color_stamp);
    } else {
      msg.header.stamp = now();
    }
    msg.header.frame_id = "camera_color_optical_frame";
    msg.pose = TransformToPoseMsg(pose);
    m3t_pose_pub_->publish(msg);
  }

  void PublishTiming(const std::string &event, std::int64_t frame_stamp_ns,
                     std::optional<std::int64_t> started_ns = std::nullopt,
                     const std::string &extra_json_fields = "") {
    if (!timing_diagnostics_) return;
    const auto node_now_ns = now().nanoseconds();
    std::ostringstream out;
    out << "{\"age_ms\":" << static_cast<double>(node_now_ns - frame_stamp_ns) / 1000000.0
        << ",\"event\":\"" << event << "\""
        << ",\"frame_stamp_ns\":" << frame_stamp_ns
        << ",\"node_now_ns\":" << node_now_ns;
    if (started_ns) {
      out << ",\"duration_ms\":"
          << static_cast<double>(node_now_ns - *started_ns) / 1000000.0;
    }
    if (!extra_json_fields.empty()) out << "," << extra_json_fields;
    out << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    timing_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "TIMING %s", msg.data.c_str());
  }

  void PublishStatus(const std::string &state, std::int64_t frame_stamp_ns,
                     const std::string &reason = "") {
    if (!status_pub_) return;
    std::ostringstream out;
    out << "{\"have_initial_pose\":" << (have_initial_pose_ ? "true" : "false")
        << ",\"reason\":\"" << EscapeJson(reason) << "\""
        << ",\"stamp_ns\":" << frame_stamp_ns
        << ",\"state\":\"" << EscapeJson(state) << "\""
        << ",\"tracking_started\":"
        << (tracking_started_.load() ? "true" : "false") << "}";
    std_msgs::msg::String msg;
    msg.data = out.str();
    status_pub_->publish(msg);
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

  void TryStartTracking() {
    if (tracking_started_.load()) return;
    if (!have_initial_pose_) return;
    if (!frame_buffer_->HasColor() || !frame_buffer_->HasDepth()) return;
    if (!frame_buffer_->PinStartFrame(initial_pose_stamp_)) return;

    bool expected = false;
    if (!tracking_started_.compare_exchange_strong(expected, true)) return;

    RCLCPP_INFO(get_logger(), "Pinned M3T start RGB-D frame to initial pose stamp %d.%09u",
                initial_pose_stamp_.sec, initial_pose_stamp_.nanosec);
    const auto initial_ns = StampToNanoseconds(initial_pose_stamp_);
    std::ostringstream extra;
    AppendOptionalField(&extra, "pinned_color_stamp_ns",
                        frame_buffer_->PinnedColorStampNs());
    AppendOptionalField(&extra, "pinned_depth_stamp_ns",
                        frame_buffer_->PinnedDepthStampNs());
    AppendOptionalField(&extra, "latest_color_stamp_ns",
                        frame_buffer_->LatestColorStampNs());
    AppendOptionalField(&extra, "latest_depth_stamp_ns",
                        frame_buffer_->LatestDepthStampNs());
    PublishTiming("m3t_start_frame_pinned", initial_ns, std::nullopt,
                  extra.str().substr(1));
    tracker_thread_ = std::thread([this]() { RunM3T(); });
  }

  void RunM3T() {
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

    if (visualize_) {
      auto color_viewer_ptr =
          std::make_shared<rcia_energy_trace_ros::AxisNormalColorViewer>(
              "color_viewer", color_camera_ptr, renderer_geometry_ptr, body_ptr,
              0.5f, 0.06f);
      color_viewer_ptr->set_display_images(show_m3t_window_);
      color_viewer_ptr->set_image_callback([this, body_ptr](const cv::Mat &image) {
        std_msgs::msg::Header header;
        header.stamp = now();
        header.frame_id = "camera_color_optical_frame";
        auto msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        m3t_overlay_pub_->publish(*msg);
        PublishM3TPose(body_ptr->body2world_pose());
        const int overlay_count = ++overlay_frame_count_;
        if (overlay_count % timing_overlay_stride_ == 0) {
          const auto latest_color_stamp = frame_buffer_->LatestColorStampNs();
          const auto latest_depth_stamp = frame_buffer_->LatestDepthStampNs();
          std::ostringstream extra;
          extra << "\"overlay_count\":" << overlay_count;
          AppendOptionalField(&extra, "latest_depth_stamp_ns", latest_depth_stamp);
          AppendOptionalField(&extra, "initial_pose_stamp_ns",
                              StampToNanoseconds(initial_pose_stamp_));
          AppendOptionalField(&extra, "pose_header_stamp_ns", latest_color_stamp);
          if (latest_color_stamp) {
            extra << ",\"pose_publish_age_ms\":"
                  << static_cast<double>(now().nanoseconds() - *latest_color_stamp) /
                         1000000.0;
          }
          if (latest_color_stamp && latest_depth_stamp) {
            extra << ",\"rgb_depth_delta_ms\":"
                  << static_cast<double>(
                         (*latest_color_stamp > *latest_depth_stamp)
                             ? (*latest_color_stamp - *latest_depth_stamp)
                             : (*latest_depth_stamp - *latest_color_stamp)) /
                         1000000.0;
          }
          PublishTiming("m3t_overlay_pose_published",
                        latest_color_stamp.value_or(StampToNanoseconds(header.stamp)),
                        std::nullopt, extra.str());
        }
      });
      tracker_ptr->AddViewer(color_viewer_ptr);
      tracker_ptr->set_viewer_time(30);
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

    body_ptr->set_body2world_pose(initial_pose_);
    link_ptr->set_link2world_pose(initial_pose_);
    auto detector_ptr = std::make_shared<m3t::StaticDetector>(
        body_name + "_detector", optimizer_ptr, initial_pose_, true);
    tracker_ptr->AddDetector(detector_ptr);

    {
      const std::lock_guard<std::mutex> lock{tracker_mutex_};
      tracker_ptr_ = tracker_ptr;
    }
    RCLCPP_INFO(get_logger(), "Starting M3T with ROS RGB-D frame source");
    const auto setup_started_ns = now().nanoseconds();
    if (!tracker_ptr->SetUp()) {
      RCLCPP_ERROR(get_logger(), "M3T ROS tracking setup failed");
      frame_buffer_->ReleasePinnedStartFrame();
      tracking_started_.store(false);
      PublishStatus("setup_failed", StampToNanoseconds(initial_pose_stamp_),
                    "tracker setup failed");
      return;
    }
    {
      std::ostringstream extra;
      AppendOptionalField(&extra, "pinned_color_stamp_ns",
                          frame_buffer_->PinnedColorStampNs());
      AppendOptionalField(&extra, "pinned_depth_stamp_ns",
                          frame_buffer_->PinnedDepthStampNs());
      PublishTiming("m3t_setup_done", StampToNanoseconds(initial_pose_stamp_),
                    setup_started_ns, extra.str().substr(1));
    }
    frame_buffer_->ReleasePinnedStartFrame();
    PublishStatus("tracking", StampToNanoseconds(initial_pose_stamp_),
                  "tracker setup done");
    if (!tracker_ptr->RunTrackerProcess(false, true)) {
      RCLCPP_ERROR(get_logger(), "M3T ROS tracking process failed");
      tracking_started_.store(false);
      PublishStatus("setup_failed", StampToNanoseconds(initial_pose_stamp_),
                    "tracker process failed");
      return;
    }
    tracking_started_.store(false);
    PublishStatus("stopped", StampToNanoseconds(initial_pose_stamp_),
                  "tracker process ended");
  }

  std::shared_ptr<rcia_energy_trace_ros::RosFrameBuffer> frame_buffer_;
  std::shared_ptr<m3t::Tracker> tracker_ptr_;
  std::thread tracker_thread_;
  std::mutex tracker_mutex_;
  std::atomic<bool> tracking_started_{false};
  bool have_initial_pose_ = false;
  bool visualize_ = true;
  bool show_m3t_window_ = false;
  bool enable_pose_resets_ = true;
  bool timing_diagnostics_ = true;
  int timing_overlay_stride_ = 30;
  std::atomic<int> overlay_frame_count_{0};
  m3t::Transform3fA initial_pose_ = m3t::Transform3fA::Identity();
  builtin_interfaces::msg::Time initial_pose_stamp_;

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
