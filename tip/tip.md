# Full Energy Tracking Launch

```bash
cd /path/to/RCIA_Energy_trace
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR=/tmp/rcia_ros_logs

ros2 launch rcia_energy_trace_ros full_energy_tracking.launch.py \
  yolo_model:=/path/to/RCIA_Energy_trace/models/yolo/best-v2.pt \
  target_class:=0 \
  detect_conf:=0.75 \
  stable_frames:=3 \
  preinit_roi_gate_enabled:=false \
  max_candidate_age_ms:=10000.0 \
  enforce_candidate_freshness:=false \
  supervisor_min_yolo_projection_iou:=0.10 \
  supervisor_max_projection_center_distance_ratio:=1.50 \
  supervisor_min_projection_area_ratio:=0.05 \
  supervisor_require_smooth_pose:=false \
  filter_initial_presence_score:=0.80 \
  filter_initial_health_score:=0.65 \
  filter_startup_grace_frames:=40 \
  filter_owned_confirm_frames:=3 \
  filter_recover_frames:=2 \
  filter_suspect_frames:=6 \
  filter_reinit_conflict_frames:=5 \
  filter_bad_tracking_frames:=5 \
  filter_out_of_view_frames:=12 \
  filter_min_rendered_pixels_for_owned:=40
```

## Diagnostics

```bash
ros2 topic echo /energy/tracking_health
ros2 topic echo /energy/tracking_state
```

## Tracking Debug Node

另开一个终端运行：

```bash
cd /path/to/RCIA_Energy_trace
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR=/tmp/rcia_ros_logs

ros2 run rcia_energy_trace_ros tracking_health_debug_node.py
```

再看结构化 debug 输出：

```bash
ros2 topic echo /energy/tracking_debug
```

这个节点只在状态切换、`request_reinit`、`stop_m3t` 或关键异常时输出。想每帧都输出可以运行：

```bash
ros2 run rcia_energy_trace_ros tracking_health_debug_node.py --ros-args -p emit_all:=true
```

重点看 `/energy/tracking_health` 里的 `tracking_filter`：

- `target_presence_score`: 目标是否还在画面内的置信度。
- `tracker_health_score`: 当前 M3T 实例是否还在正确跟踪的置信度。
- `reliability_state`: `M3T_STARTING` / `M3T_OWNED` / `M3T_SUSPECT` / `REINIT_NEEDED` / `LOST`。
- `m3t_good_streak`: M3T 自身连续可信的帧数。
- `m3t_bad_streak`: M3T 自身不可信的连续帧数。
- `external_conflict_streak`: YOLO/深度/外观/投影持续冲突帧数。
- `out_of_view_streak`: 目标离开画面的连续证据帧数。
- `weak_validation_fail_streak`: 外部弱验证失败但 M3T 仍可能正常的连续帧数。
- `request_reinit`: 目标大概率还在，但 M3T 不可信，应该重新初始化。
- `stop_m3t`: 目标和跟踪都低可信，应该销毁实例回到搜索。

YOLO 现在只是 tracking 阶段的弱证据，不是真值：

- M3T 接管后，YOLO 漏检不应单独触发 reinit/stop。
- YOLO 连续稳定检测到目标且 M3T 投影持续冲突，才会增加 `external_conflict_streak`。
- 物体真正离开画面时，应看到 `out_of_view_streak` 连续增加，最终进入 `LOST`。
