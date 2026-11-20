# Full Energy Tracking Launch

```bash
cd /path/to/RCIA_Energy_trace
source /opt/ros/humble/setup.bash
source ~/energy_trace_ws/install/setup.bash
export ROS_LOG_DIR=/tmp/rcia_ros_logs

ros2 launch rcia_energy_trace_ros full_energy_tracking.launch.py \
  yolo_model:=/path/to/RCIA_Energy_trace/models/yolo/1117_openvino_model \
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
source ~/energy_trace_ws/install/setup.bash
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

## 小电脑部署（Intel 12 代 i7 + 核显，CPU 推理）

默认配置不使用 CUDA：

- FoundationPose 的 refine 网络和 score 网络：OpenVINO FP32，设备 CPU。
- 渲染：moderngl + EGL。
- YOLO：OpenVINO 导出模型。
- 初始化迭代次数 `est_refine_iter` 为 3，对称方式 `symmetry_mode` 为 `z12`。

以下步骤按 Ubuntu 22.04 + ROS2 Humble + RealSense D435 编写。下面的路径都以仓库根目录为准。

### 1. 系统准备

安装编译 mycpp 和 ROS 功能包需要的系统依赖。mycpp 的 CMake 需要 Boost 的 system 和 program_options 组件：

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-colcon-common-extensions \
  libeigen3-dev libglew-dev libglfw3-dev librealsense2-dev \
  libboost-system-dev libboost-program-options-dev libopencv-dev
```

把用户加入 `render` 组：

```bash
sudo usermod -aG render $USER
```

重新登录后确认用户已在 `render` 组，并能看到渲染设备：

```bash
id -nG | grep -w render
ls -l /dev/dri
```

不在 `render` 组时，EGL 会退化为软件渲染（`llvmpipe`），速度会慢很多，后面的自检会失败。

### 2. 下载公开权重

```bash
cd /path/to/RCIA_Energy_trace
bash scripts/download_public_weights.sh
```

### 3. 安装 Python 环境

```bash
bash scripts/setup_foundationpose_ros_env.sh
```

这个脚本会安装 CUDA 版 torch，并编译 mycpp 扩展。pytorch3d、nvdiffrast、bundlesdf mycuda 已从安装流程中删除，CPU 路径不需要它们。cu121 版 torch 也能在 CPU 上运行，只是下载体积大。运行时仍然需要 torch，张量运算靠它完成。在没有 NVIDIA GPU 的小电脑上，这一步还没有验证，失败时先保留完整报错。

脚本最后会执行 `scripts/export_foundationpose_openvino.py`，生成两个 IR：

```text
foundation_pose/weights/2023-10-28-18-33-37/openvino/refine_fp32.xml
foundation_pose/weights/2024-01-11-20-02-45/openvino/score_fp32.xml
```

需要单独导出时运行：

```bash
conda activate foundationpose_ros
python scripts/export_foundationpose_openvino.py
```

### 4. 编译 ROS 功能包

```bash
mkdir -p ~/energy_trace_ws/src
ln -s /path/to/RCIA_Energy_trace/ros2/rcia_energy_trace_ros ~/energy_trace_ws/src/
cd ~/energy_trace_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rcia_energy_trace_ros
```

必须加 `--symlink-install`。launch 文件用自身路径推算项目根目录（`Path(__file__).resolve().parents[3]`）。不加这个参数时，install 里放的是副本，推出的根目录会变成 `~/energy_trace_ws/install/rcia_energy_trace_ros`，`cd project_root`、`foundationpose_env.sh` 和默认的 `yolo_model` 都会指向错误位置。如果不能使用 symlink，启动时加 `project_root:=/path/to/RCIA_Energy_trace`。

### 5. 自检

```bash
cd /path/to/RCIA_Energy_trace
source ~/miniconda3/etc/profile.d/conda.sh
conda activate foundationpose_ros
source /opt/ros/humble/setup.bash
source scripts/foundationpose_env.sh
python scripts/verify_foundationpose_ros_env.py
```

通过的标志：

- 最后一行是 `FOUNDATIONPOSE_ROS_ENV_OK`。
- 输出的 `GL_RENDERER` 是 Mesa Intel 渲染器（例如 `Mesa Intel(R) ...`），不能是 `llvmpipe`。
- `openvino` 版本以 `2024.6.0` 开头。

自检失败时，脚本会以退出码 1 结束，并说明缺少什么。

### 6. 启动

先启动相机，然后在 ROS 环境中运行。在上面 `Full Energy Tracking Launch` 的命令后面追加下面的部署参数即可，其余参数沿用原命令：

```bash
source /opt/ros/humble/setup.bash
source ~/energy_trace_ws/install/setup.bash
export ROS_LOG_DIR=/tmp/rcia_ros_logs

ros2 launch rcia_energy_trace_ros full_energy_tracking.launch.py \
  device:=cpu \
  render_backend:=moderngl \
  net_backend:=openvino \
  ov_device:=CPU \
  symmetry_mode:=z12 \
  est_refine_iter:=3 \
  canonicalize_yaw:=true
```

这些参数的默认值也是上面的值，因此不写也会使用同样的配置。

### 7. 检查运行结果

- 初始化日志里 `GL_RENDERER` 不是 `llvmpipe`。
- 初始化完成后，`/energy/timing_events` 中的 `foundationpose_register_done` 会记录 `duration_ms`：

  ```bash
  ros2 topic echo /energy/timing_events
  ```

- 物体直立、正对镜头时，`/energy/trusted_pose` 旋转矩阵的第 0 列应接近 `(0, 0, -1)`。yaw 规范化后，这个方向是固定的。
- 候选位姿状态：

  ```bash
  ros2 topic echo /energy/foundationpose_candidate_status
  ```

### 8. 常见问题

- `GL_RENDERER` 是 `llvmpipe`：当前用户不在 `render` 组，或 `/dev/dri` 不可访问。按第 1 步处理后重新登录。
- 报 `FileNotFoundError` 并提示导出命令：IR 没有导出，运行第 3 步的导出命令。
- `openvino` 版本不对：不要用 `pip install -U` 升级。`foundationpose_env.sh` 设置了 `PYTHONNOUSERSITE=1`，用来屏蔽 `~/.local` 里另一份 OpenVINO。
- 初始化耗时很长：先看 `foundationpose_register_done` 的 `duration_ms`，再确认 `est_refine_iter` 是否为 3。
- 切回 CUDA 路径，用于和 GPU 结果对比。这条路径需要 nvdiffrast，安装流程已不再包含它，需要手动安装：

  ```bash
  ros2 launch rcia_energy_trace_ros full_energy_tracking.launch.py device:=cuda render_backend:=nvdiffrast net_backend:=torch
  ```

### 9. 尚未在小电脑上确定的参数

核显（`ov_device:=GPU`）、OpenVINO 线程数、INT8 量化还没有在小电脑上测过。当前默认是 CPU 和 FP32。测完之后再改默认值。
