from datetime import datetime
from pathlib import Path
from threading import Lock

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.event_handlers import OnProcessIO
from launch.actions import RegisterEventHandler
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


def _make_output_capture(log_file):
    """Copy stdout/stderr emitted by launch-managed processes to one file."""
    write_lock = Lock()
    initialized = False

    def write_output(event):
        nonlocal initialized

        text = event.text
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if not text:
            return

        with write_lock:
            if not initialized:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(
                    "# full_energy_tracking launch log\n"
                    f"# started: {datetime.now().isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                initialized = True

            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(text)
                if not text.endswith("\n"):
                    stream.write("\n")

    return [
        RegisterEventHandler(
            OnProcessIO(
                on_stdout=write_output,
                on_stderr=write_output,
            )
        )
    ]


def _foundationpose_cmd():
    project_root = LaunchConfiguration("project_root")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    initial_pose_topic = LaunchConfiguration("initial_pose_topic")
    init_request_topic = LaunchConfiguration("init_request_topic")
    yolo_detections_topic = LaunchConfiguration("yolo_detections_topic")
    candidate_pose_topic = LaunchConfiguration("candidate_pose_topic")
    candidate_status_topic = LaunchConfiguration("candidate_status_topic")
    yolo_model = LaunchConfiguration("yolo_model")
    target_class = LaunchConfiguration("target_class")
    detect_conf = LaunchConfiguration("detect_conf")
    stable_frames = LaunchConfiguration("stable_frames")
    publish_once = LaunchConfiguration("publish_once")
    monitor_after_init = LaunchConfiguration("monitor_after_init")
    monitor_stride = LaunchConfiguration("monitor_stride")
    latest_frame_max_age_ms = LaunchConfiguration("latest_frame_max_age_ms")
    process_timer_hz = LaunchConfiguration("process_timer_hz")
    timing_diagnostics = LaunchConfiguration("timing_diagnostics")
    timing_topic = LaunchConfiguration("timing_topic")
    show_foundationpose_windows = LaunchConfiguration("show_foundationpose_windows")
    est_refine_iter = LaunchConfiguration("est_refine_iter")
    device = LaunchConfiguration("device")
    render_backend = LaunchConfiguration("render_backend")
    net_backend = LaunchConfiguration("net_backend")
    ov_device = LaunchConfiguration("ov_device")
    ov_precision = LaunchConfiguration("ov_precision")
    ov_num_threads = LaunchConfiguration("ov_num_threads")
    ov_cache_dir = LaunchConfiguration("ov_cache_dir")
    symmetry_mode = LaunchConfiguration("symmetry_mode")
    canonicalize_yaw = LaunchConfiguration("canonicalize_yaw")
    yaw_fallback_deg = LaunchConfiguration("yaw_fallback_deg")

    return [
        "eval \"$(conda shell.bash hook)\" && "
        "conda activate foundationpose_ros && "
        "source /opt/ros/humble/setup.bash && "
        "cd ",
        project_root,
        " && source scripts/foundationpose_env.sh && "
        "python ros2/rcia_energy_trace_ros/scripts/foundationpose_init_node.py "
        "--ros-args "
        "-p color_topic:=",
        color_topic,
        " -p depth_topic:=",
        depth_topic,
        " -p camera_info_topic:=",
        camera_info_topic,
        " -p initial_pose_topic:=",
        initial_pose_topic,
        " -p init_request_topic:=",
        init_request_topic,
        " -p yolo_detections_topic:=",
        yolo_detections_topic,
        " -p candidate_pose_topic:=",
        candidate_pose_topic,
        " -p candidate_status_topic:=",
        candidate_status_topic,
        " -p direct_initial_pose_enabled:=false",
        " -p request_required:=true",
        " -p yolo_model:=",
        yolo_model,
        " -p target_class:=",
        target_class,
        " -p detect_conf:=",
        detect_conf,
        " -p register_conf:=",
        detect_conf,
        " -p stable_frames:=",
        stable_frames,
        " -p publish_once:=",
        publish_once,
        " -p monitor_after_init:=",
        monitor_after_init,
        " -p monitor_stride:=",
        monitor_stride,
        " -p supervisor_enabled:=false",
        " -p latest_frame_max_age_ms:=",
        latest_frame_max_age_ms,
        " -p process_timer_hz:=",
        process_timer_hz,
        " -p timing_diagnostics:=",
        timing_diagnostics,
        " -p timing_topic:=",
        timing_topic,
        " -p show_diagnostic_windows:=",
        show_foundationpose_windows,
        " -p est_refine_iter:=",
        est_refine_iter,
        " -p device:=",
        device,
        " -p render_backend:=",
        render_backend,
        " -p net_backend:=",
        net_backend,
        " -p ov_device:=",
        ov_device,
        " -p ov_precision:=",
        ov_precision,
        " -p ov_num_threads:=",
        ov_num_threads,
        " -p ov_cache_dir:=",
        ov_cache_dir,
        " -p symmetry_mode:=",
        symmetry_mode,
        " -p canonicalize_yaw:=",
        canonicalize_yaw,
        " -p yaw_fallback_deg:=",
        yaw_fallback_deg,
    ]


def generate_launch_description():
    project_root = Path(DEFAULT_PROJECT_ROOT)
    log_root = project_root / "runtime" / "logs"
    full_log_root = log_root / "full"
    ros_log_root = log_root / "ros2"
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    ros_run_log_root = ros_log_root / run_stamp
    log_file = full_log_root / f"full_energy_tracking_{run_stamp}.log"

    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    depth_camera_info_topic = LaunchConfiguration("depth_camera_info_topic")
    initial_pose_topic = LaunchConfiguration("initial_pose_topic")
    yolo_detections_topic = LaunchConfiguration("yolo_detections_topic")
    m3t_pose_topic = LaunchConfiguration("m3t_pose_topic")
    m3t_stop_topic = LaunchConfiguration("m3t_stop_topic")
    m3t_status_topic = LaunchConfiguration("m3t_status_topic")
    trusted_pose_topic = LaunchConfiguration("trusted_pose_topic")
    tracking_state_topic = LaunchConfiguration("tracking_state_topic")
    tracking_health_topic = LaunchConfiguration("tracking_health_topic")
    init_request_topic = LaunchConfiguration("init_request_topic")
    candidate_pose_topic = LaunchConfiguration("candidate_pose_topic")
    candidate_status_topic = LaunchConfiguration("candidate_status_topic")
    timing_topic = LaunchConfiguration("timing_topic")
    visualize = LaunchConfiguration("visualize")
    show_m3t_window = LaunchConfiguration("show_m3t_window")
    show_pipeline_window = LaunchConfiguration("show_pipeline_window")
    record = LaunchConfiguration("record")
    record_dir = LaunchConfiguration("record_dir")
    pipeline_refresh_hz = LaunchConfiguration("pipeline_refresh_hz")
    record_fps = LaunchConfiguration("record_fps")
    frame_history_size = LaunchConfiguration("frame_history_size")
    start_frame_tolerance_ms = LaunchConfiguration("start_frame_tolerance_ms")
    yolo_topic = LaunchConfiguration("yolo_topic")
    mask_topic = LaunchConfiguration("mask_topic")
    foundationpose_topic = LaunchConfiguration("foundationpose_topic")
    m3t_topic = LaunchConfiguration("m3t_topic")
    pipeline_topic = LaunchConfiguration("pipeline_topic")
    canonicalize_yaw = LaunchConfiguration("canonicalize_yaw")
    yaw_fallback_deg = LaunchConfiguration("yaw_fallback_deg")

    return LaunchDescription([
        SetEnvironmentVariable(
            name="ROS_LOG_DIR",
            value=str(ros_run_log_root),
        ),
        *_make_output_capture(log_file),
        DeclareLaunchArgument(
            "project_root",
            default_value=DEFAULT_PROJECT_ROOT,
        ),
        DeclareLaunchArgument("color_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument(
            "depth_camera_info_topic",
            default_value="/camera/depth/camera_info",
        ),
        DeclareLaunchArgument("initial_pose_topic", default_value="/energy/initial_pose"),
        DeclareLaunchArgument(
            "yolo_model",
            default_value=PathJoinSubstitution(
                [LaunchConfiguration("project_root"), "models", "yolo", "1117_openvino_model"]
            ),
        ),
        DeclareLaunchArgument("init_request_topic", default_value="/energy/init_request"),
        DeclareLaunchArgument(
            "candidate_pose_topic",
            default_value="/energy/foundationpose_candidate_pose",
        ),
        DeclareLaunchArgument(
            "candidate_status_topic",
            default_value="/energy/foundationpose_candidate_status",
        ),
        DeclareLaunchArgument("yolo_detections_topic", default_value="/energy/yolo_detections"),
        DeclareLaunchArgument("m3t_pose_topic", default_value="/energy/m3t_pose"),
        DeclareLaunchArgument("m3t_stop_topic", default_value="/energy/m3t_stop"),
        DeclareLaunchArgument("m3t_status_topic", default_value="/energy/m3t_status"),
        DeclareLaunchArgument("trusted_pose_topic", default_value="/energy/trusted_pose"),
        DeclareLaunchArgument("tracking_state_topic", default_value="/energy/tracking_state"),
        DeclareLaunchArgument("tracking_health_topic", default_value="/energy/tracking_health"),
        DeclareLaunchArgument("timing_topic", default_value="/energy/timing_events"),
        DeclareLaunchArgument("yolo_topic", default_value="/energy/diagnostics/yolo"),
        DeclareLaunchArgument("mask_topic", default_value="/energy/diagnostics/mask"),
        DeclareLaunchArgument(
            "foundationpose_topic",
            default_value="/energy/diagnostics/foundationpose",
        ),
        DeclareLaunchArgument("m3t_topic", default_value="/energy/diagnostics/m3t_overlay"),
        DeclareLaunchArgument(
            "pipeline_topic",
            default_value="/energy/diagnostics/pipeline_view",
        ),
        DeclareLaunchArgument("width", default_value="640"),
        DeclareLaunchArgument("height", default_value="480"),
        DeclareLaunchArgument("fps", default_value="30"),
        DeclareLaunchArgument("target_class", default_value="0"),
        DeclareLaunchArgument("detect_conf", default_value="0.75"),
        DeclareLaunchArgument("stable_frames", default_value="3"),
        DeclareLaunchArgument("publish_once", default_value="true"),
        DeclareLaunchArgument("monitor_after_init", default_value="true"),
        DeclareLaunchArgument("monitor_stride", default_value="3"),
        DeclareLaunchArgument("latest_frame_max_age_ms", default_value="120.0"),
        DeclareLaunchArgument("process_timer_hz", default_value="30.0"),
        DeclareLaunchArgument("supervisor_rate_hz", default_value="8.0"),
        DeclareLaunchArgument("candidate_timeout_ms", default_value="8000.0"),
        DeclareLaunchArgument("candidate_frame_tolerance_ms", default_value="40.0"),
        DeclareLaunchArgument("max_candidate_age_ms", default_value="10000.0"),
        DeclareLaunchArgument("enforce_candidate_freshness", default_value="false"),
        DeclareLaunchArgument("rgbd_history_size", default_value="240"),
        DeclareLaunchArgument("supervisor_max_yolo_age_ms", default_value="500.0"),
        DeclareLaunchArgument("supervisor_max_pose_age_ms", default_value="250.0"),
        DeclareLaunchArgument("supervisor_lost_frames", default_value="6"),
        DeclareLaunchArgument("supervisor_out_of_view_frames", default_value="10"),
        DeclareLaunchArgument("supervisor_min_rendered_overlap_pixels", default_value="80"),
        DeclareLaunchArgument("supervisor_min_yolo_projection_iou", default_value="0.35"),
        DeclareLaunchArgument(
            "supervisor_max_projection_center_distance_ratio",
            default_value="0.75",
        ),
        DeclareLaunchArgument("supervisor_min_projection_area_ratio", default_value="0.25"),
        DeclareLaunchArgument("supervisor_tracking_min_depth_ratio", default_value="0.60"),
        DeclareLaunchArgument("supervisor_max_pose_jump_translation_m", default_value="0.08"),
        DeclareLaunchArgument("supervisor_max_pose_jump_rotation_deg", default_value="30.0"),
        DeclareLaunchArgument("supervisor_require_smooth_pose", default_value="true"),
        DeclareLaunchArgument("filter_initial_presence_score", default_value="0.80"),
        DeclareLaunchArgument("filter_initial_health_score", default_value="0.65"),
        DeclareLaunchArgument("filter_startup_grace_frames", default_value="40"),
        DeclareLaunchArgument("filter_occlusion_grace_frames", default_value="24"),
        DeclareLaunchArgument("filter_owned_confirm_frames", default_value="3"),
        DeclareLaunchArgument("filter_recover_frames", default_value="2"),
        DeclareLaunchArgument("filter_suspect_frames", default_value="6"),
        DeclareLaunchArgument("filter_reinit_conflict_frames", default_value="5"),
        DeclareLaunchArgument("filter_bad_tracking_frames", default_value="5"),
        DeclareLaunchArgument("filter_out_of_view_frames", default_value="12"),
        DeclareLaunchArgument("filter_min_rendered_pixels_for_owned", default_value="40"),
        DeclareLaunchArgument("filter_locked_health_threshold", default_value="0.70"),
        DeclareLaunchArgument("filter_suspect_health_threshold", default_value="0.40"),
        DeclareLaunchArgument("filter_degraded_health_threshold", default_value="0.40"),
        DeclareLaunchArgument("filter_degraded_presence_threshold", default_value="0.55"),
        DeclareLaunchArgument("filter_lost_health_threshold", default_value="0.35"),
        DeclareLaunchArgument("filter_lost_presence_threshold", default_value="0.35"),
        DeclareLaunchArgument("preinit_roi_gate_enabled", default_value="false"),
        DeclareLaunchArgument("preinit_min_valid_depth_ratio", default_value="0.20"),
        DeclareLaunchArgument("preinit_min_low_saturation_ratio", default_value="0.45"),
        DeclareLaunchArgument("preinit_min_black_white_support", default_value="0.20"),
        DeclareLaunchArgument("timing_diagnostics", default_value="true"),
        DeclareLaunchArgument("timing_overlay_stride", default_value="30"),
        DeclareLaunchArgument("visualize", default_value="true"),
        DeclareLaunchArgument("show_m3t_window", default_value="false"),
        DeclareLaunchArgument("show_camera_preview", default_value="false"),
        DeclareLaunchArgument("show_pipeline_window", default_value="true"),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("record_dir", default_value="record"),
        DeclareLaunchArgument("pipeline_refresh_hz", default_value="10.0"),
        DeclareLaunchArgument("record_fps", default_value="0.0"),
        DeclareLaunchArgument("show_foundationpose_windows", default_value="false"),
        DeclareLaunchArgument("frame_history_size", default_value="300"),
        DeclareLaunchArgument("start_frame_tolerance_ms", default_value="80.0"),
        DeclareLaunchArgument("est_refine_iter", default_value="3"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("render_backend", default_value="moderngl"),
        DeclareLaunchArgument("net_backend", default_value="openvino"),
        DeclareLaunchArgument("ov_device", default_value="CPU"),
        DeclareLaunchArgument("ov_precision", default_value="int8"),
        DeclareLaunchArgument("ov_num_threads", default_value="0"),
        # ROS 2 拒绝空值的 -p 参数，所以用 none 表示不启用缓存。
        DeclareLaunchArgument("ov_cache_dir", default_value="none"),
        DeclareLaunchArgument("symmetry_mode", default_value="z12"),
        DeclareLaunchArgument("canonicalize_yaw", default_value="true"),
        DeclareLaunchArgument("yaw_fallback_deg", default_value="15.0"),
        Node(
            package="rcia_energy_trace_ros",
            executable="realsense_rgbd_publisher_node",
            name="realsense_rgbd_publisher",
            output="screen",
            parameters=[{
                "width": ParameterValue(LaunchConfiguration("width"), value_type=int),
                "height": ParameterValue(LaunchConfiguration("height"), value_type=int),
                "fps": ParameterValue(LaunchConfiguration("fps"), value_type=int),
                "show_preview": ParameterValue(
                    LaunchConfiguration("show_camera_preview"),
                    value_type=bool,
                ),
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "color_camera_info_topic": camera_info_topic,
                "depth_camera_info_topic": depth_camera_info_topic,
                "frame_id": "camera_color_optical_frame",
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
                "m3t_pose_topic": m3t_pose_topic,
                "m3t_stop_topic": m3t_stop_topic,
                "m3t_status_topic": m3t_status_topic,
                "enable_pose_resets": False,
                "timing_diagnostics": ParameterValue(
                    LaunchConfiguration("timing_diagnostics"),
                    value_type=bool,
                ),
                "timing_topic": timing_topic,
                "timing_overlay_stride": ParameterValue(
                    LaunchConfiguration("timing_overlay_stride"),
                    value_type=int,
                ),
                "visualize": ParameterValue(visualize, value_type=bool),
                "show_m3t_window": ParameterValue(show_m3t_window, value_type=bool),
                "diagnostic_overlay_topic": m3t_topic,
                "frame_history_size": ParameterValue(frame_history_size, value_type=int),
                "start_frame_tolerance_ms": ParameterValue(
                    start_frame_tolerance_ms,
                    value_type=float,
                ),
            }],
        ),
        Node(
            package="rcia_energy_trace_ros",
            executable="tracking_supervisor_node.py",
            name="tracking_supervisor",
            output="screen",
            parameters=[{
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "yolo_detections_topic": yolo_detections_topic,
                "candidate_pose_topic": candidate_pose_topic,
                "candidate_status_topic": candidate_status_topic,
                "m3t_status_topic": m3t_status_topic,
                "m3t_pose_topic": m3t_pose_topic,
                "init_request_topic": init_request_topic,
                "initial_pose_topic": initial_pose_topic,
                "m3t_stop_topic": m3t_stop_topic,
                "trusted_pose_topic": trusted_pose_topic,
                "tracking_state_topic": tracking_state_topic,
                "tracking_health_topic": tracking_health_topic,
                "target_class": ParameterValue(
                    LaunchConfiguration("target_class"),
                    value_type=int,
                ),
                "supervisor_rate_hz": ParameterValue(
                    LaunchConfiguration("supervisor_rate_hz"),
                    value_type=float,
                ),
                "max_yolo_age_ms": ParameterValue(
                    LaunchConfiguration("supervisor_max_yolo_age_ms"),
                    value_type=float,
                ),
                "max_pose_age_ms": ParameterValue(
                    LaunchConfiguration("supervisor_max_pose_age_ms"),
                    value_type=float,
                ),
                "candidate_timeout_ms": ParameterValue(
                    LaunchConfiguration("candidate_timeout_ms"),
                    value_type=float,
                ),
                "candidate_frame_tolerance_ms": ParameterValue(
                    LaunchConfiguration("candidate_frame_tolerance_ms"),
                    value_type=float,
                ),
                "max_candidate_age_ms": ParameterValue(
                    LaunchConfiguration("max_candidate_age_ms"),
                    value_type=float,
                ),
                "enforce_candidate_freshness": ParameterValue(
                    LaunchConfiguration("enforce_candidate_freshness"),
                    value_type=bool,
                ),
                "rgbd_history_size": ParameterValue(
                    LaunchConfiguration("rgbd_history_size"),
                    value_type=int,
                ),
                "stable_yolo_frames": ParameterValue(
                    LaunchConfiguration("stable_frames"),
                    value_type=int,
                ),
                "lost_frames": ParameterValue(
                    LaunchConfiguration("supervisor_lost_frames"),
                    value_type=int,
                ),
                "out_of_view_frames": ParameterValue(
                    LaunchConfiguration("supervisor_out_of_view_frames"),
                    value_type=int,
                ),
                "min_rendered_overlap_pixels": ParameterValue(
                    LaunchConfiguration("supervisor_min_rendered_overlap_pixels"),
                    value_type=int,
                ),
                "min_yolo_projection_iou": ParameterValue(
                    LaunchConfiguration("supervisor_min_yolo_projection_iou"),
                    value_type=float,
                ),
                "max_projection_center_distance_ratio": ParameterValue(
                    LaunchConfiguration(
                        "supervisor_max_projection_center_distance_ratio"
                    ),
                    value_type=float,
                ),
                "min_projection_area_ratio": ParameterValue(
                    LaunchConfiguration("supervisor_min_projection_area_ratio"),
                    value_type=float,
                ),
                "tracking_min_depth_consistency_ratio": ParameterValue(
                    LaunchConfiguration("supervisor_tracking_min_depth_ratio"),
                    value_type=float,
                ),
                "max_pose_jump_translation_m": ParameterValue(
                    LaunchConfiguration("supervisor_max_pose_jump_translation_m"),
                    value_type=float,
                ),
                "max_pose_jump_rotation_deg": ParameterValue(
                    LaunchConfiguration("supervisor_max_pose_jump_rotation_deg"),
                    value_type=float,
                ),
                "require_smooth_pose": ParameterValue(
                    LaunchConfiguration("supervisor_require_smooth_pose"),
                    value_type=bool,
                ),
                "filter_initial_presence_score": ParameterValue(
                    LaunchConfiguration("filter_initial_presence_score"),
                    value_type=float,
                ),
                "filter_initial_health_score": ParameterValue(
                    LaunchConfiguration("filter_initial_health_score"),
                    value_type=float,
                ),
                "filter_startup_grace_frames": ParameterValue(
                    LaunchConfiguration("filter_startup_grace_frames"),
                    value_type=int,
                ),
                "filter_occlusion_grace_frames": ParameterValue(
                    LaunchConfiguration("filter_occlusion_grace_frames"),
                    value_type=int,
                ),
                "filter_owned_confirm_frames": ParameterValue(
                    LaunchConfiguration("filter_owned_confirm_frames"),
                    value_type=int,
                ),
                "filter_recover_frames": ParameterValue(
                    LaunchConfiguration("filter_recover_frames"),
                    value_type=int,
                ),
                "filter_suspect_frames": ParameterValue(
                    LaunchConfiguration("filter_suspect_frames"),
                    value_type=int,
                ),
                "filter_reinit_conflict_frames": ParameterValue(
                    LaunchConfiguration("filter_reinit_conflict_frames"),
                    value_type=int,
                ),
                "filter_bad_tracking_frames": ParameterValue(
                    LaunchConfiguration("filter_bad_tracking_frames"),
                    value_type=int,
                ),
                "filter_out_of_view_frames": ParameterValue(
                    LaunchConfiguration("filter_out_of_view_frames"),
                    value_type=int,
                ),
                "filter_min_rendered_pixels_for_owned": ParameterValue(
                    LaunchConfiguration("filter_min_rendered_pixels_for_owned"),
                    value_type=int,
                ),
                "filter_locked_health_threshold": ParameterValue(
                    LaunchConfiguration("filter_locked_health_threshold"),
                    value_type=float,
                ),
                "filter_suspect_health_threshold": ParameterValue(
                    LaunchConfiguration("filter_suspect_health_threshold"),
                    value_type=float,
                ),
                "filter_degraded_health_threshold": ParameterValue(
                    LaunchConfiguration("filter_degraded_health_threshold"),
                    value_type=float,
                ),
                "filter_degraded_presence_threshold": ParameterValue(
                    LaunchConfiguration("filter_degraded_presence_threshold"),
                    value_type=float,
                ),
                "filter_lost_health_threshold": ParameterValue(
                    LaunchConfiguration("filter_lost_health_threshold"),
                    value_type=float,
                ),
                "filter_lost_presence_threshold": ParameterValue(
                    LaunchConfiguration("filter_lost_presence_threshold"),
                    value_type=float,
                ),
                "preinit_roi_gate_enabled": ParameterValue(
                    LaunchConfiguration("preinit_roi_gate_enabled"),
                    value_type=bool,
                ),
                "preinit_min_valid_depth_ratio": ParameterValue(
                    LaunchConfiguration("preinit_min_valid_depth_ratio"),
                    value_type=float,
                ),
                "preinit_min_low_saturation_ratio": ParameterValue(
                    LaunchConfiguration("preinit_min_low_saturation_ratio"),
                    value_type=float,
                ),
                "preinit_min_black_white_support": ParameterValue(
                    LaunchConfiguration("preinit_min_black_white_support"),
                    value_type=float,
                ),
                "canonicalize_yaw": ParameterValue(canonicalize_yaw, value_type=bool),
                "yaw_fallback_deg": ParameterValue(yaw_fallback_deg, value_type=float),
            }],
        ),
        Node(
            package="rcia_energy_trace_ros",
            executable="pipeline_visualizer_node.py",
            name="pipeline_visualizer",
            output="screen",
            parameters=[{
                "color_topic": color_topic,
                "yolo_topic": yolo_topic,
                "mask_topic": mask_topic,
                "foundationpose_topic": foundationpose_topic,
                "m3t_topic": m3t_topic,
                "tracking_health_topic": tracking_health_topic,
                "output_topic": pipeline_topic,
                "show_window": ParameterValue(show_pipeline_window, value_type=bool),
                "record_enabled": ParameterValue(record, value_type=bool),
                "record_dir": record_dir,
                "refresh_hz": ParameterValue(pipeline_refresh_hz, value_type=float),
                "record_fps": ParameterValue(record_fps, value_type=float),
            }],
        ),
        ExecuteProcess(
            cmd=["/bin/bash", "-lc", _foundationpose_cmd()],
            name="foundationpose_init",
            output="screen",
        ),
    ])
