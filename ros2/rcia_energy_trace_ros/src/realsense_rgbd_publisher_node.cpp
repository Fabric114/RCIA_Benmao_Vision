#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>

class RealSenseRgbdPublisherNode : public rclcpp::Node {
 public:
  RealSenseRgbdPublisherNode() : Node{"realsense_rgbd_publisher_node"} {
    width_ = declare_parameter<int>("width", 640);
    height_ = declare_parameter<int>("height", 480);
    fps_ = declare_parameter<int>("fps", 30);
    color_topic_ = declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
    depth_topic_ = declare_parameter<std::string>(
        "depth_topic", "/camera/aligned_depth_to_color/image_raw");
    color_info_topic_ = declare_parameter<std::string>(
        "color_camera_info_topic", "/camera/color/camera_info");
    depth_info_topic_ = declare_parameter<std::string>(
        "depth_camera_info_topic", "/camera/depth/camera_info");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera_color_optical_frame");
    show_preview_ = declare_parameter<bool>("show_preview", false);
    preview_stride_ = declare_parameter<int>("preview_stride", 1);

    color_pub_ = create_publisher<sensor_msgs::msg::Image>(color_topic_, 10);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic_, 10);
    color_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(color_info_topic_, 10);
    depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(depth_info_topic_, 10);

    config_.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_BGR8, fps_);
    config_.enable_stream(RS2_STREAM_DEPTH, width_, height_, RS2_FORMAT_Z16, fps_);
    profile_ = pipe_.start(config_);
    align_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);

    const auto color_profile =
        profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    color_intrinsics_ = color_profile.get_intrinsics();

    RCLCPP_INFO(get_logger(),
                "Publishing aligned RealSense RGB-D %dx%d@%d on %s and %s",
                width_, height_, fps_, color_topic_.c_str(), depth_topic_.c_str());
    if (show_preview_) {
      cv::namedWindow("RCIA Energy Trace RGB-D", cv::WINDOW_NORMAL);
      cv::resizeWindow("RCIA Energy Trace RGB-D", width_ * 2, height_);
      RCLCPP_INFO(get_logger(),
                  "OpenCV preview enabled. Press 'q' in the preview window to quit.");
    }

    timer_ = create_wall_timer(
        std::chrono::milliseconds{1},
        [this]() { PublishFrame(); });
  }

 private:
  sensor_msgs::msg::CameraInfo MakeCameraInfo(const rclcpp::Time &stamp) const {
    sensor_msgs::msg::CameraInfo info;
    info.header.stamp = stamp;
    info.header.frame_id = frame_id_;
    info.width = static_cast<uint32_t>(color_intrinsics_.width);
    info.height = static_cast<uint32_t>(color_intrinsics_.height);
    info.k = {
        color_intrinsics_.fx, 0.0, color_intrinsics_.ppx,
        0.0, color_intrinsics_.fy, color_intrinsics_.ppy,
        0.0, 0.0, 1.0,
    };
    info.p = {
        color_intrinsics_.fx, 0.0, color_intrinsics_.ppx, 0.0,
        0.0, color_intrinsics_.fy, color_intrinsics_.ppy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    };
    info.r = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    info.distortion_model = "plumb_bob";
    info.d.resize(5, 0.0);
    return info;
  }

  void PublishFrame() {
    rs2::frameset frames;
    if (!pipe_.poll_for_frames(&frames)) return;
    frames = align_->process(frames);

    const rs2::video_frame color = frames.get_color_frame();
    const rs2::depth_frame depth = frames.get_depth_frame();
    if (!color || !depth) return;

    const rclcpp::Time stamp = now();
    auto color_msg = sensor_msgs::msg::Image();
    color_msg.header.stamp = stamp;
    color_msg.header.frame_id = frame_id_;
    color_msg.height = static_cast<uint32_t>(color.get_height());
    color_msg.width = static_cast<uint32_t>(color.get_width());
    color_msg.encoding = "bgr8";
    color_msg.is_bigendian = false;
    color_msg.step = color_msg.width * 3;
    color_msg.data.resize(color_msg.step * color_msg.height);
    std::memcpy(color_msg.data.data(), color.get_data(), color_msg.data.size());

    auto depth_msg = sensor_msgs::msg::Image();
    depth_msg.header.stamp = stamp;
    depth_msg.header.frame_id = frame_id_;
    depth_msg.height = static_cast<uint32_t>(depth.get_height());
    depth_msg.width = static_cast<uint32_t>(depth.get_width());
    depth_msg.encoding = "16UC1";
    depth_msg.is_bigendian = false;
    depth_msg.step = depth_msg.width * sizeof(uint16_t);
    depth_msg.data.resize(depth_msg.step * depth_msg.height);
    std::memcpy(depth_msg.data.data(), depth.get_data(), depth_msg.data.size());

    const auto info = MakeCameraInfo(stamp);
    color_pub_->publish(color_msg);
    depth_pub_->publish(depth_msg);
    color_info_pub_->publish(info);
    depth_info_pub_->publish(info);

    if (show_preview_) ShowPreview(color, depth);
  }

  void ShowPreview(const rs2::video_frame &color, const rs2::depth_frame &depth) {
    if (++preview_frame_count_ % std::max(1, preview_stride_) != 0) return;

    cv::Mat color_bgr(
        cv::Size{color.get_width(), color.get_height()},
        CV_8UC3,
        const_cast<void *>(color.get_data()),
        cv::Mat::AUTO_STEP);
    cv::Mat depth_u16(
        cv::Size{depth.get_width(), depth.get_height()},
        CV_16UC1,
        const_cast<void *>(depth.get_data()),
        cv::Mat::AUTO_STEP);

    cv::Mat depth_8u;
    depth_u16.convertTo(depth_8u, CV_8UC1, 255.0 / 4000.0);
    cv::Mat depth_color;
    cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);

    cv::Mat preview;
    cv::hconcat(color_bgr, depth_color, preview);
    cv::putText(preview, "RGB", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                {0, 255, 0}, 2);
    cv::putText(preview, "Aligned depth", {width_ + 12, 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);
    cv::imshow("RCIA Energy Trace RGB-D", preview);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) {
      RCLCPP_INFO(get_logger(), "Preview quit requested");
      rclcpp::shutdown();
    }
  }

  int width_{640};
  int height_{480};
  int fps_{30};
  std::string color_topic_;
  std::string depth_topic_;
  std::string color_info_topic_;
  std::string depth_info_topic_;
  std::string frame_id_;
  bool show_preview_{false};
  int preview_stride_{1};
  int preview_frame_count_{0};

  rs2::pipeline pipe_;
  rs2::config config_;
  rs2::pipeline_profile profile_;
  std::unique_ptr<rs2::align> align_;
  rs2_intrinsics color_intrinsics_{};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RealSenseRgbdPublisherNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
