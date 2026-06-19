//RBGT/build/examples

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <m3t/realsense_camera.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/common.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/normal_viewer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/silhouette_renderer.h>
#include <m3t/static_detector.h>
#include <m3t/texture_modality.h>
#include <m3t/tracker.h>

using namespace std;

// 自定义Viewer：在NormalColorViewer基础上叠加三维坐标系
class CoordinateAxisViewer : public m3t::NormalColorViewer {
public:
    CoordinateAxisViewer(
        const std::string &name,
        const std::shared_ptr<m3t::ColorCamera> &color_camera_ptr,
        const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
        const std::shared_ptr<m3t::Body> &body_ptr,
        const cv::Mat &camera_matrix,
        float opacity = 0.5f,
        float axis_length = 0.05f)
        : NormalColorViewer(name, color_camera_ptr, renderer_geometry_ptr, opacity),
          body_ptr_(body_ptr),
          K_(camera_matrix),
          axis_length_(axis_length) {}

    bool UpdateViewer(int save_index) override {
        // 先调用父类的UpdateViewer获取渲染后的图像
        if (!NormalColorViewer::UpdateViewer(save_index)) {
            return false;
        }

        // 获取当前显示的图像并绘制坐标系
        cv::Mat display_image = color_camera_ptr()->image().clone();

        // 渲染overlay
        m3t::FullNormalRenderer temp_renderer("temp_renderer", renderer_geometry_ptr(), color_camera_ptr());
        temp_renderer.SetUp();
        temp_renderer.StartRendering();
        temp_renderer.FetchNormalImage();

        // Alpha blend
        cv::Mat blended = CalculateAlphaBlend(display_image, temp_renderer.normal_image(), opacity());

        // 绘制三维坐标系
        DrawCoordinateAxis(blended);

        // 显示图像
        if (display_images()) {
            cv::imshow(name(), blended);
            cv::waitKey(1);
        }

        return true;
    }

private:
    void DrawCoordinateAxis(cv::Mat &image) {
        // 获取物体在相机坐标系下的位姿
        m3t::Transform3fA body2world = body_ptr_->body2world_pose();
        Eigen::Matrix4f pose = body2world.matrix();

        // 提取物体中心（平移部分）
        Eigen::Vector3f center(pose(0, 3), pose(1, 3), pose(2, 3));

        // 提取旋转矩阵
        Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0);

        // 定义三个坐标轴方向（在物体坐标系中）
        Eigen::Vector3f x_axis = rotation.col(0) * axis_length_;
        Eigen::Vector3f y_axis = rotation.col(1) * axis_length_;
        Eigen::Vector3f z_axis = rotation.col(2) * axis_length_;

        // 投影到图像平面
        cv::Point2f center_2d = Project3DPoint(center);
        cv::Point2f x_end_2d = Project3DPoint(center + x_axis);
        cv::Point2f y_end_2d = Project3DPoint(center + y_axis);
        cv::Point2f z_end_2d = Project3DPoint(center + z_axis);

        // 绘制坐标轴（X=红色, Y=绿色, Z=蓝色）
        int thickness = 3;
        cv::arrowedLine(image, center_2d, x_end_2d, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA, 0, 0.2);  // X轴-红色
        cv::arrowedLine(image, center_2d, y_end_2d, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA, 0, 0.2);  // Y轴-绿色
        cv::arrowedLine(image, center_2d, z_end_2d, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA, 0, 0.2);  // Z轴-蓝色

        // 添加轴标签
        cv::putText(image, "X", x_end_2d + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, "Y", y_end_2d + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, "Z", z_end_2d + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    }

    cv::Point2f Project3DPoint(const Eigen::Vector3f &point_3d) {
        // 使用相机内参投影3D点到2D
        // K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        double fx = K_.at<double>(0, 0);
        double fy = K_.at<double>(1, 1);
        double cx = K_.at<double>(0, 2);
        double cy = K_.at<double>(1, 2);

        float x = point_3d.x() / point_3d.z() * fx + cx;
        float y = point_3d.y() / point_3d.z() * fy + cy;

        return cv::Point2f(x, y);
    }

    cv::Mat CalculateAlphaBlend(const cv::Mat &camera_image,
                                const cv::Mat &renderer_image, float opacity) {
        cv::Mat image{camera_image.size(), CV_8UC3};
        int v, u;
        const cv::Vec3b *ptr_camera_image;
        const cv::Vec4b *ptr_renderer_image;
        cv::Vec3b *ptr_image;
        const uchar *val_camera_image;
        const uchar *val_renderer_image;
        uchar *val_image;
        float alpha, alpha_inv;
        float alpha_scale = opacity / 255.0f;

        for (v = 0; v < camera_image.rows; ++v) {
            ptr_camera_image = camera_image.ptr<cv::Vec3b>(v);
            ptr_renderer_image = renderer_image.ptr<cv::Vec4b>(v);
            ptr_image = image.ptr<cv::Vec3b>(v);
            for (u = 0; u < camera_image.cols; ++u) {
                val_camera_image = ptr_camera_image[u].val;
                val_renderer_image = ptr_renderer_image[u].val;
                val_image = ptr_image[u].val;

                alpha = float(val_renderer_image[3]) * alpha_scale;
                alpha_inv = 1.0f - alpha;
                val_image[0] = char(val_camera_image[0] * alpha_inv + val_renderer_image[0] * alpha);
                val_image[1] = char(val_camera_image[1] * alpha_inv + val_renderer_image[1] * alpha);
                val_image[2] = char(val_camera_image[2] * alpha_inv + val_renderer_image[2] * alpha);
            }
        }
        return image;
    }

    std::shared_ptr<m3t::Body> body_ptr_;
    cv::Mat K_;
    float axis_length_;
};

#ifndef RCIA_PROJECT_ROOT
#define RCIA_PROJECT_ROOT "."
#endif

// 相机内参
cv::Mat K_mat = (cv::Mat_<double>(3, 3) << 606.1544849121768, 0, 320.71911360697, 0,
                 606.2304160969798, 239.7231533262084, 0, 0, 1);

Eigen::Matrix4f ReadFoundationPose(const std::string &path)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

    // 等待文件出现（最多等 120 秒）
    const int max_wait_ms = 120000;
    const int poll_interval_ms = 200;
    int waited = 0;

    cout << "等待 FoundationPose 写入初始位姿..." << endl;
    while (!std::filesystem::exists(path) && waited < max_wait_ms)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        waited += poll_interval_ms;
    }

    if (!std::filesystem::exists(path))
    {
        cerr << "超时：未收到 FoundationPose 初始位姿文件 " << path << endl;
        cerr << "请先运行 scripts/run_energy_tracking.sh 或 scripts/foundationpose_init.py 生成初始位姿" << endl;
        exit(1);
    }

    // 读取文件
    std::ifstream f(path);
    if (!f.is_open())
    {
        cerr << "无法打开位姿文件: " << path << endl;
        exit(1);
    }

    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            f >> pose(r, c);
    f.close();

    cout << "已读取 FoundationPose 初始位姿:" << endl;
    cout << pose << endl;
    return pose;
}


int main()
{
    const std::filesystem::path project_root = std::filesystem::weakly_canonical(RCIA_PROJECT_ROOT);
    const std::filesystem::path body_yaml_path = project_root / "models/m3t/Str3D_body.yaml";
    const std::filesystem::path model_directory = project_root / "models/m3t";
    const std::string body_name = "Str3D";
    const std::filesystem::path pose_file = project_root / "runtime/foundation_pose.txt";

    // 读取 FoundationPose 生成的初始位姿
    Eigen::Matrix4f initial_pose = ReadFoundationPose(pose_file.string());

    // ==================== M3T 跟踪 ====================
    constexpr bool kUseRegionModality = true;
    constexpr bool kUseDepthModality = true;
    constexpr bool kMeasureOcclusions = true;
    constexpr bool kVisualizePoseResult = true;

    auto tracker_ptr{std::make_shared<m3t::Tracker>("tracker")};
    auto renderer_geometry_ptr{std::make_shared<m3t::RendererGeometry>("renderer geometry")};

    auto color_camera_ptr{std::make_shared<m3t::RealSenseColorCamera>("realsense_color")};
    auto depth_camera_ptr{std::make_shared<m3t::RealSenseDepthCamera>("realsense_depth")};

    auto body_ptr{std::make_shared<m3t::Body>(body_name, body_yaml_path)};
    renderer_geometry_ptr->AddBody(body_ptr);

    // 使用自定义的CoordinateAxisViewer替代NormalColorViewer
    auto color_viewer_ptr{std::make_shared<CoordinateAxisViewer>(
        "color_viewer", color_camera_ptr, renderer_geometry_ptr, body_ptr, K_mat, 0.5f, 0.05f)};
    tracker_ptr->AddViewer(color_viewer_ptr);

    auto color_depth_renderer_ptr{std::make_shared<m3t::FocusedBasicDepthRenderer>("color_depth_renderer", renderer_geometry_ptr, color_camera_ptr)};
    auto depth_depth_renderer_ptr{std::make_shared<m3t::FocusedBasicDepthRenderer>("depth_depth_renderer", renderer_geometry_ptr, depth_camera_ptr)};
    auto color_silhouette_renderer_ptr{std::make_shared<m3t::FocusedSilhouetteRenderer>("color_silhouette_renderer", renderer_geometry_ptr, color_camera_ptr)};

    auto region_model_ptr{std::make_shared<m3t::RegionModel>(body_name + "_region_model", body_ptr, model_directory / "Str3D_region.bin")};
    auto depth_model_ptr{std::make_shared<m3t::DepthModel>(body_name + "_depth_model", body_ptr, model_directory / "Str3D_depth.bin")};

    auto region_modality_ptr{std::make_shared<m3t::RegionModality>(body_name + "_region_modality", body_ptr, color_camera_ptr, region_model_ptr)};
    auto depth_modality_ptr{std::make_shared<m3t::DepthModality>(body_name + "_depth_modality", body_ptr, depth_camera_ptr, depth_model_ptr)};

    if (kVisualizePoseResult)
        region_modality_ptr->set_visualize_pose_result(true);

    if (kMeasureOcclusions)
    {
        region_modality_ptr->MeasureOcclusions(depth_camera_ptr);
        depth_modality_ptr->MeasureOcclusions();
    }

    auto link_ptr{std::make_shared<m3t::Link>(body_name + "_link", body_ptr)};
    if (kUseRegionModality)
        link_ptr->AddModality(region_modality_ptr);
    if (kUseDepthModality)
        link_ptr->AddModality(depth_modality_ptr);

    auto optimizer_ptr{std::make_shared<m3t::Optimizer>(body_name + "_optimizer", link_ptr)};
    tracker_ptr->AddOptimizer(optimizer_ptr);

    // 设置 FoundationPose 提供的初始位姿
    m3t::Transform3fA pose_transform;
    pose_transform.matrix() = initial_pose;
    body_ptr->set_body2world_pose(pose_transform);
    link_ptr->set_link2world_pose(pose_transform);

    auto detector_ptr{std::make_shared<m3t::StaticDetector>(body_name + "_detector", optimizer_ptr, pose_transform, true)};
    tracker_ptr->AddDetector(detector_ptr);

    tracker_ptr->set_viewer_time(30);

    // 不执行检测（避免覆盖初始位姿），直接开始跟踪
    if (!tracker_ptr->SetUp() || !tracker_ptr->RunTrackerProcess(false, true)) {
        cout << "M3T 启动失败" << endl;
        return -1;
    }

    return 0;
}
