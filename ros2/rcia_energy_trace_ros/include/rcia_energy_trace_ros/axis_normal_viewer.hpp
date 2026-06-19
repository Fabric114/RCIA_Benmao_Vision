#pragma once

#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/normal_renderer.h>
#include <m3t/renderer_geometry.h>
#include <m3t/viewer.h>

#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <string>

namespace rcia_energy_trace_ros {

class AxisNormalColorViewer : public m3t::ColorViewer {
 public:
  AxisNormalColorViewer(
      const std::string &name,
      const std::shared_ptr<m3t::ColorCamera> &color_camera_ptr,
      const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<m3t::Body> &body_ptr, float opacity = 0.5f,
      float axis_scale = 0.06f);

  bool SetUp() override;
  bool UpdateViewer(int save_index) override;
  std::shared_ptr<m3t::RendererGeometry> renderer_geometry_ptr() const override;
  void set_image_callback(std::function<void(const cv::Mat &)> callback);

 private:
  bool ProjectPoint(const Eigen::Vector3f &point_f_camera,
                    cv::Point *point) const;
  Eigen::Vector3f BodyCentroid() const;
  void DrawBodyAxes(cv::Mat *image) const;

  std::shared_ptr<m3t::RendererGeometry> renderer_geometry_ptr_;
  std::shared_ptr<m3t::Body> body_ptr_;
  m3t::FullNormalRenderer renderer_;
  std::function<void(const cv::Mat &)> image_callback_;
  float opacity_ = 0.5f;
  float axis_scale_ = 0.06f;
};

}  // namespace rcia_energy_trace_ros
