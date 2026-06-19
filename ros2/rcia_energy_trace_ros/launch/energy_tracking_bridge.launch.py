from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_camera_info_topic = LaunchConfiguration("depth_camera_info_topic")
    initial_pose_topic = LaunchConfiguration("initial_pose_topic")
    target_class = LaunchConfiguration("target_class")
    detect_conf = LaunchConfiguration("detect_conf")
    stable_frames = LaunchConfiguration("stable_frames")
    visualize = LaunchConfiguration("visualize")
    sync_tolerance_ms = LaunchConfiguration("sync_tolerance_ms")
    sync_queue_size = LaunchConfiguration("sync_queue_size")
    frame_history_size = LaunchConfiguration("frame_history_size")
    start_frame_tolerance_ms = LaunchConfiguration("start_frame_tolerance_ms")
    diagnostic_visualization = LaunchConfiguration("diagnostic_visualization")
    show_diagnostic_windows = LaunchConfiguration("show_diagnostic_windows")

    return LaunchDescription([
        DeclareLaunchArgument("color_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("depth_camera_info_topic", default_value="/camera/depth/camera_info"),
        DeclareLaunchArgument("initial_pose_topic", default_value="/energy/initial_pose"),
        DeclareLaunchArgument("target_class", default_value="0"),
        DeclareLaunchArgument("detect_conf", default_value="0.75"),
        DeclareLaunchArgument("stable_frames", default_value="3"),
        DeclareLaunchArgument("visualize", default_value="true"),
        DeclareLaunchArgument("sync_tolerance_ms", default_value="5.0"),
        DeclareLaunchArgument("sync_queue_size", default_value="60"),
        DeclareLaunchArgument("frame_history_size", default_value="300"),
        DeclareLaunchArgument("start_frame_tolerance_ms", default_value="10.0"),
        DeclareLaunchArgument("diagnostic_visualization", default_value="true"),
        DeclareLaunchArgument("show_diagnostic_windows", default_value="false"),
        Node(
            package="rcia_energy_trace_ros",
            executable="foundationpose_init_node.py",
            name="foundationpose_init",
            output="screen",
            parameters=[{
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "initial_pose_topic": initial_pose_topic,
                "target_class": ParameterValue(target_class, value_type=int),
                "detect_conf": ParameterValue(detect_conf, value_type=float),
                "register_conf": ParameterValue(detect_conf, value_type=float),
                "stable_frames": ParameterValue(stable_frames, value_type=int),
                "sync_tolerance_ms": ParameterValue(sync_tolerance_ms, value_type=float),
                "sync_queue_size": ParameterValue(sync_queue_size, value_type=int),
                "diagnostic_visualization": ParameterValue(
                    diagnostic_visualization, value_type=bool
                ),
                "show_diagnostic_windows": ParameterValue(
                    show_diagnostic_windows, value_type=bool
                ),
            }],
        ),
        Node(
            package="rcia_energy_trace_ros",
            executable="m3t_ros_tracking_node",
            name="m3t_tracking",
            output="screen",
            parameters=[{
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "color_camera_info_topic": camera_info_topic,
                "depth_camera_info_topic": depth_camera_info_topic,
                "initial_pose_topic": initial_pose_topic,
                "visualize": ParameterValue(visualize, value_type=bool),
                "frame_history_size": ParameterValue(frame_history_size, value_type=int),
                "start_frame_tolerance_ms": ParameterValue(
                    start_frame_tolerance_ms, value_type=float
                ),
            }],
        ),
    ])
