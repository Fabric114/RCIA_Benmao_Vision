# RCIA Energy Trace ROS2 Bridge

This package is the first implementation slice of方案C: the RealSense camera is owned by an external ROS2 camera publisher, while FoundationPose and M3T consume the same RGB-D stream.

## Contract

- The camera publisher is the only process that opens the D435.
- `foundationpose_init_node.py` subscribes to RGB-D frames, matches depth to each color frame by timestamp, runs YOLO/depth-window mask/FoundationPose, and publishes `/energy/initial_pose` with that color frame stamp.
- `m3t_ros_tracking_node` subscribes to RGB-D frames and `/energy/initial_pose`, pins its first M3T setup frame to the pose stamp, then continues tracking from live ROS-backed camera adapters.
- Diagnostic images are written to `runtime/foundationpose_ros_debug/diagnostics/`: `yolo_latest.png`, `yolo_selected.png`, and one-shot `foundationpose_initialization.png`.
- The M3T color viewer keeps the existing colored model overlay and adds an XYZ coordinate frame at the tracked model centroid.
- The current convention is preserved: M3T treats the camera frame as world, and the initial pose is object/body in that camera/world frame.

## Expected Topics

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
/camera/depth/camera_info
/energy/initial_pose
```

## Build

From a ROS2 workspace:

```bash
mkdir -p ~/energy_trace_ws/src
ln -s /path/to/RCIA_Energy_trace/ros2/rcia_energy_trace_ros ~/energy_trace_ws/src/
cd ~/energy_trace_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select rcia_energy_trace_ros
```

The existing non-ROS pipeline remains available through `scripts/run_energy_tracking.sh`.

## Run

Start a camera publisher first. For example, with `realsense2_camera`:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

Then start the bridge:

```bash
source ~/energy_trace_ws/install/setup.bash
ros2 launch rcia_energy_trace_ros energy_tracking_bridge.launch.py \
  target_class:=0 detect_conf:=0.75 stable_frames:=3 \
  sync_tolerance_ms:=5.0 sync_queue_size:=60 \
  start_frame_tolerance_ms:=10.0 frame_history_size:=300 \
  diagnostic_visualization:=true show_diagnostic_windows:=false
```

For a direct OpenCV RGB-D preview from the single camera-owner node:

```bash
/tmp/rcia_energy_trace_ros_build/realsense_rgbd_publisher_node \
  --ros-args \
  -p show_preview:=true \
  -p color_topic:=/camera/color/image_raw \
  -p depth_topic:=/camera/aligned_depth_to_color/image_raw
```

The preview shows RGB on the left and aligned depth on the right. Press `q` or `Esc` in the window to quit.
