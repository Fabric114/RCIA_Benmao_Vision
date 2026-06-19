#include "rcia_energy_trace_ros/axis_normal_viewer.hpp"

#include <m3t/common.h>

#include <iostream>
#include <utility>

namespace rcia_energy_trace_ros {

namespace {

cv::Mat CalculateAlphaBlend(const cv::Mat &camera_image,
                            const cv::Mat &renderer_image, float opacity) {
  cv::Mat image{camera_image.size(), CV_8UC3};
  const float alpha_scale = opacity / 255.0f;
  for (int v = 0; v < camera_image.rows; ++v) {
    const auto *camera_row = camera_image.ptr<cv::Vec3b>(v);
    const auto *renderer_row = renderer_image.ptr<cv::Vec4b>(v);
    auto *image_row = image.ptr<cv::Vec3b>(v);
    for (int u = 0; u < camera_image.cols; ++u) {
      const float alpha = static_cast<float>(renderer_row[u][3]) * alpha_scale;
      const float alpha_inv = 1.0f - alpha;
      for (int c = 0; c < 3; ++c) {
        image_row[u][c] = static_cast<uchar>(
            camera_row[u][c] * alpha_inv + renderer_row[u][c] * alpha);
      }
    }
  }
  return image;
}

}  // namespace

AxisNormalColorViewer::AxisNormalColorViewer(
    const std::string &name,
    const std::shared_ptr<m3t::ColorCamera> &color_camera_ptr,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    const std::shared_ptr<m3t::Body> &body_ptr, float opacity, float axis_scale)
    : ColorViewer{name, color_camera_ptr},
      renderer_geometry_ptr_{renderer_geometry_ptr},
      body_ptr_{body_ptr},
      renderer_{"axis_renderer", renderer_geometry_ptr, color_camera_ptr},
      opacity_{opacity},
      axis_scale_{axis_scale} {}

bool AxisNormalColorViewer::SetUp() {
  set_up_ = false;
  if (!color_camera_ptr_ || !color_camera_ptr_->set_up()) {
    std::cerr << "Color camera is not set up for axis normal viewer"
              << std::endl;
    return false;
  }
  if (!renderer_geometry_ptr_ || !renderer_geometry_ptr_->set_up()) {
    std::cerr << "Renderer geometry is not set up for axis normal viewer"
              << std::endl;
    return false;
  }
  if (!body_ptr_ || !body_ptr_->set_up()) {
    std::cerr << "Body is not set up for axis normal viewer" << std::endl;
    return false;
  }
  renderer_.set_renderer_geometry_ptr(renderer_geometry_ptr_);
  renderer_.set_camera_ptr(color_camera_ptr_);
  if (!renderer_.SetUp()) return false;
  set_up_ = true;
  return true;
}

bool AxisNormalColorViewer::UpdateViewer(int save_index) {
  if (!set_up_) {
    std::cerr << "Set up axis normal color viewer " << name_ << " first"
              << std::endl;
    return false;
  }

  renderer_.StartRendering();
  renderer_.FetchNormalImage();
  cv::Mat image = CalculateAlphaBlend(color_camera_ptr_->image(),
                                      renderer_.normal_image(), opacity_);
  DrawBodyAxes(&image);
  if (image_callback_) image_callback_(image);
  DisplayAndSaveImage(save_index, image);
  return true;
}

std::shared_ptr<m3t::RendererGeometry>
AxisNormalColorViewer::renderer_geometry_ptr() const {
  return renderer_geometry_ptr_;
}

void AxisNormalColorViewer::set_image_callback(
    std::function<void(const cv::Mat &)> callback) {
  image_callback_ = std::move(callback);
}

bool AxisNormalColorViewer::ProjectPoint(const Eigen::Vector3f &point_f_camera,
                                         cv::Point *point) const {
  if (point_f_camera.z() <= 0.001f) return false;
  const auto &intrinsics = color_camera_ptr_->intrinsics();
  const int u = static_cast<int>(point_f_camera.x() * intrinsics.fu /
                                     point_f_camera.z() +
                                 intrinsics.ppu + 0.5f);
  const int v = static_cast<int>(point_f_camera.y() * intrinsics.fv /
                                     point_f_camera.z() +
                                 intrinsics.ppv + 0.5f);
  *point = cv::Point{u, v};
  return true;
}

Eigen::Vector3f AxisNormalColorViewer::BodyCentroid() const {
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  const auto &vertices = body_ptr_->vertices();
  if (vertices.empty()) return centroid;
  for (const auto &vertex : vertices) centroid += body_ptr_->geometry2body_pose() * vertex;
  centroid /= static_cast<float>(vertices.size());
  return centroid;
}

void AxisNormalColorViewer::DrawBodyAxes(cv::Mat *image) const {
  const m3t::Transform3fA body2camera_pose =
      color_camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
  const Eigen::Vector3f origin_f_body = BodyCentroid();
  const Eigen::Vector3f origin_f_camera = body2camera_pose * origin_f_body;

  struct Axis {
    Eigen::Vector3f direction;
    cv::Scalar color;
    const char *label;
  };
  const Axis axes[] = {
      {Eigen::Vector3f::UnitX(), cv::Scalar{0, 0, 255}, "X"},
      {Eigen::Vector3f::UnitY(), cv::Scalar{0, 255, 0}, "Y"},
      {Eigen::Vector3f::UnitZ(), cv::Scalar{255, 0, 0}, "Z"},
  };

  cv::Point origin_px;
  if (!ProjectPoint(origin_f_camera, &origin_px)) return;
  cv::circle(*image, origin_px, 4, cv::Scalar{255, 255, 255}, cv::FILLED,
             cv::LINE_AA);

  for (const auto &axis : axes) {
    const Eigen::Vector3f end_f_body =
        origin_f_body + axis.direction * axis_scale_;
    const Eigen::Vector3f end_f_camera = body2camera_pose * end_f_body;
    cv::Point end_px;
    if (!ProjectPoint(end_f_camera, &end_px)) continue;
    cv::arrowedLine(*image, origin_px, end_px, axis.color, 2, cv::LINE_AA, 0,
                    0.18);
    cv::putText(*image, axis.label, end_px + cv::Point{4, -4},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, axis.color, 2, cv::LINE_AA);
  }
}

}  // namespace rcia_energy_trace_ros
