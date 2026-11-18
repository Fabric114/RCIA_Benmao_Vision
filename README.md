# RCIA_Energy_trace

## 项目背景

由于26赛季工程机器人对矿取矿形式大改，曾经的矿石识别算法已经成为历史。同时由于本赛季的的矿石修改成类似圆柱体无任何视觉特征点的能量单元。简单识别提取特征点进行定位的方式已无方向，因此诞生了识别其6dof的想法。

本项目的目标是使用 RGB-D 相机对能量单元进行 6DoF 位姿初始化与实时跟踪。当前主路线为：

```text
RealSense D435 RGB-D
  -> YOLO 2D 检测
  -> 深度窗口 mask
  -> FoundationPose 6DoF 初始化
  -> M3T/SRT3D 风格模型跟踪
  -> tracking supervisor 状态融合与重初始化管理
  -> 置信度 6DoF 位姿输出 / 可视化
```

## 阅前须知

1. 此项目及其演示视频全部跑在相机坐标系下。想要将位姿用于工程机械臂，需要进行手眼标定，将 `camera_color_optical_frame` 下的位姿转换到对应的 `base_link` 或机器人工作坐标系下。实现坐标系统一就不会出现foundtionpose输出的初始化位姿到m3t时出现转变的情况
2. 由于缺少了场上数据集，且仅仅标注了少量角度和少量的数据集，因此建议只作为测试demo使用
3. 此项目保留里了linemod/line2d算法的复现代码，但后续落地仅仅使用了部分验证算法，有兴趣可以尝试继续复现，论文：https://stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

## 硬件设备

相机：realsense D435

## 环境准备
# 1. 基础环境

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- NVIDIA GPU + CUDA
- RealSense D435 / `librealsense2`
- OpenCV、CMake、colcon

常用依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  build-essential cmake \
  libopencv-dev libeigen3-dev \
  libglew-dev libglfw3-dev \
  librealsense2-dev
```

### 2. FoundationPose 环境

可直接运行安装脚本：
```bash
cd /path/to/RCIA_Energy_trace
bash scripts/setup_foundationpose_ros_env.sh
```
安装后验证：

```bash
conda activate foundationpose_ros
source /opt/ros/humble/setup.bash
source scripts/foundationpose_env.sh
python scripts/verify_foundationpose_ros_env.py
```
### 3. 模型文件

仓库会保留项目自有模型，保证基础 demo 资源完整：

```text
models/object/Str3D.obj
models/object/Str3D_bbox_center.obj
models/m3t/Str3D_region.bin
models/m3t/Str3D_depth.bin
models/m3t/Str3D_body.yaml
models/yolo/best.pt
models/yolo/best-v2.pt
```

以下公开权重不直接提交到 Git，使用脚本下载：

```text
foundation_pose/weights/*/model_best.pth
```

下载命令：

```bash
bash scripts/download_public_weights.sh
```

公开来源：

- FoundationPose 官方仓库：https://github.com/NVlabs/FoundationPose
- FoundationPose 官方 Google Drive 权重：https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing
- FoundationPose HuggingFace 镜像：https://huggingface.co/gpue/foundationpose-weights

### 4. 部署到 Intel 小电脑（CPU + 核显）

默认配置不使用 CUDA：`device=cpu`、`render_backend=moderngl`（EGL）、`net_backend=openvino`、`ov_device=CPU`，`est_refine_iter=3`。

1. 让当前用户能访问 GPU 渲染设备，否则 EGL 会退化为软件渲染（`llvmpipe`），速度会很慢：
   ```bash
   sudo usermod -aG render $USER
   ```
   重新登录后生效。
2. 导出 OpenVINO IR（安装脚本的最后一步已自动执行，只在需要重新导出时手动运行）：
   ```bash
   python scripts/export_foundationpose_openvino.py
   ```
3. 自检。输出中的 `GL_RENDERER` 为 `llvmpipe` 时脚本以退出码 1 失败：
   ```bash
   python scripts/verify_foundationpose_ros_env.py
   ```
4. YOLO 默认使用 `models/yolo/best_openvino_model`。换成新训练的模型时，先用 `yolo export model=<新模型>.pt format=openvino` 导出，再通过 `yolo_model:=<导出目录>` 指定。

切回 CUDA 路径，用于和 GPU 结果对比。这条路径需要 nvdiffrast，安装脚本已不再包含它，需要手动安装：

```bash
ros2 launch rcia_energy_trace_ros full_energy_tracking.launch.py device:=cuda render_backend:=nvdiffrast net_backend:=torch
```

## 项目结构

```text
RCIA_Energy_trace/
├── CMakeLists.txt          # 旧版非 ROS M3T demo 的 CMake 构建入口
├── README.md               # 项目说明、系统架构、运行流程和开源注意事项
├── foundation_pose/        # FoundationPose 本地集成代码、推理组件和权重目录
├── m3t_app/                # 旧版非 ROS M3T 跟踪 demo，保留作最小 C++ 示例
├── models/                 # 目标 3D 模型、M3T 预计算模型、YOLO 权重
├── ros2/                   # ROS2 功能包，包含相机发布、初始化、跟踪、状态机和可视化节点
├── scripts/                # 算法工具、状态融合、验证逻辑、LINEMOD 实验和环境脚本
├── tests/                  # Python 单元测试和模块级回归测试
├── third_party/            # 第三方源码依赖，当前主要是 M3T
├── tip/                    # 本地调试命令、参数组合和运行经验记录
└── runtime/                # 运行时输出、诊断图、缓存和实验结果，默认不提交
```

## 实现流程

### 1. RGB-D 采集

`realsense_rgbd_publisher_node` 打开 RealSense D435，发布以下话题：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
/camera/depth/camera_info
```

使用的是彩色相机光心坐标系：
```text
camera_color_optical_frame
```
因此输出的 6DoF 位姿默认表示为：
```text
object_in_camera_color_optical_frame
```
### 2. YOLO 目标检测

这里的yolo是作为检测部分的第一部分，如果yolo模型本身效果很好的话可以少很多后续的验证算法

`foundationpose_init_node.py` 内部加载 YOLO 模型，对 RGB 图像进行目标检测。

YOLO 的作用是：
- 在初始化前提供目标 bbox。
- 给深度窗口 mask 提供框范围。
- 给 tracking supervisor 一个“目标可能存在”的观测信号。

注意：YOLO 不是 6DoF 位姿来源，也不是目标是否存在的绝对真值。因为我的训练数据有限，当目标被遮挡、旋转到特殊角度、只露出侧面或远离训练分布时，YOLO 可能漏检；也可能在背景上误检。

### 3. 深度窗口 mask

在 YOLO bbox 内取框中心 35% 区域的有效深度中位数作为参考深度，保留深度与参考值相差不超过 0.11 m 的像素。不再使用 SAM 分割。

深度窗口 mask 的作用是：

- 限制 FoundationPose 注册区域。
- 减少背景对 6DoF 初始化的干扰。

### 4. FoundationPose 初始化

这里解释一下，其实是使用Foundtionpose就已经可以得到较为准确的6dof位姿了，但是由于Foundtionpose算法对显卡的强依赖且追踪效果一般。因此决定仅仅作为全局初始化工具使用，减少Foundtionpose的大量参与。

FoundationPose 使用：

- RGB 图像。
- 对齐深度图。
- 相机内参。
- 深度窗口 mask。
- `models/object/Str3D.obj`。
- FoundationPose 预训练权重。

输出一个候选 6DoF 位姿。

当前系统中，FoundationPose 默认不直接长期接管跟踪，只是作为全局初始化器使用。初始化节点把结果发布到：

```text
/energy/foundationpose_candidate_pose
/energy/foundationpose_candidate_status
```

### 5. Initialization Gate

`tracking_supervisor_node.py` 收到 FoundationPose candidate 后，会通过 `scripts/initialization_gate.py` 做一个轻量初始化 gate。

gate 只检查：
- 位姿矩阵是否为合法 4x4 齐次矩阵。
- 是否全部为有限数值。
- 最后一行是否为 `[0, 0, 0, 1]`。
- candidate 是否过旧，以及是否启用强制 freshness。

### 6. M3T 跟踪

这里使用M3T的原因最主要的原因就是其强大的追踪能力，对设备的要求也相对要低。只要给一个相对准确的初始化位姿态，基本就可以实现很不错的跟踪能力

`m3t_ros_tracking_node` 订阅 `/energy/initial_pose`，在对应 RGB-D 帧附近启动 M3T。
##使用时还是建议先进行手眼标定，在相机坐标系下会出现Foundtionpose的初始化由于相机坐标系的改变而被错误的输入到m3t当中

M3T 使用：
- `models/object/Str3D.obj`。
- `models/m3t/Str3D_region.bin`。
- `models/m3t/Str3D_depth.bin`。
- 实时 RGB-D 帧。
- FoundationPose 给出的初始位姿。

M3T 启动后持续输出：

```text
/energy/m3t_pose
/energy/m3t_status
/energy/diagnostics/m3t_overlay
```
这里的overlay 会显示彩色模型叠加和目标模型重心处的 XYZ 坐标轴。

### 7. Tracking Health

M3T 启动后，系统每帧对当前 M3T pose 生成健康度观测。

主要证据包括：

- `depth_ratio`：按当前 M3T pose 渲染目标深度，与真实深度图比较的一致比例。
- `black_white_ok`：投影区域是否符合黑白目标的低饱和和黑白亮度分布特征。
- `projection_iou`：M3T 投影 bbox 与 YOLO bbox 的重合程度。
- `projection_center_ok`：投影中心与 YOLO bbox 中心是否差距过大。
- `projection_area_ok`：投影面积与 YOLO bbox 面积是否差距过大。
- `smoothness_ok`：当前 M3T pose 相对上一帧是否存在异常跳变。
- `rendered_pixel_count`：当前位姿下目标投影是否仍有足够像素落在画面内。

这些条件不会被简单理解为单项失败就杀死 M3T。它们只是带噪声的证据而已。

### 8. 权重状态机 / Tracking State Filter

这里就是对整个m3t追踪器的监管器了，m3t本身是不知道自己到底是不是追踪的正不正确的，而且当创建完一次实例之后也是不会自动销毁其实例的，导致出问题之后没办法重新初始化。因此设计的一个管理层。这里说一下，一开始的版本是采用强门控的，但是效果并不理想，经常误杀，所以采用了现在这一版本将每一个可以作为观测的证据进行打分，这部分可以升级成卡尔曼滤波，对整个m3t的追踪状态进行信念分布从而进行更好的状态管理。

`scripts/tracking_state_filter.py` 是当前系统的跟踪生命周期管理器。核心思想是：

不把 YOLO、深度一致性、黑白外观、投影 IoU 任意一个单项条件当作真值；
而是把它们融合成目标存在分数和跟踪健康分数，再通过连续帧状态决定是否保留、怀疑、重初始化或销毁 M3T。

状态包括：
```text
M3T_STARTING
M3T_OWNED
M3T_SUSPECT
REINIT_NEEDED
LOST
```
状态含义：
- `M3T_STARTING`：已向 M3T 发送初始位姿，等待 M3T 输出稳定跟踪。
- `M3T_OWNED`：M3T 已接管跟踪，系统发布可信 pose。
- `M3T_SUSPECT`：M3T 仍可能正确，但部分验证条件持续偏弱，进入观察状态。
- `REINIT_NEEDED`：目标大概率还在画面内，但 M3T 当前实例不可信，需要停止旧实例并重新请求 FoundationPose 初始化。
- `LOST`：目标大概率离开画面，或 M3T 无有效 pose 且没有目标存在证据，需要销毁实例并回到搜索。

状态机维护两类分数：

- `target_presence_score`：目标是否仍在画面内的置信度。
- `tracker_health_score`：当前 M3T 跟踪是否可信的置信度。

还维护连续帧计数：

- `m3t_good_streak`：M3T 连续健康帧数。
- `m3t_bad_streak`：M3T 连续异常帧数。
- `external_conflict_streak`：外部证据与 M3T 明显冲突的连续帧数。
- `out_of_view_streak`：目标离开视野证据的连续帧数。
- `weak_validation_fail_streak`：弱验证持续失败的连续帧数。

典型策略：

- YOLO 检测到目标时，提高目标存在置信度，但不单独决定 M3T 正确。
- YOLO 未检测到目标时，不立即认为目标消失，因为可能是遮挡、特殊角度或 YOLO 数据集不足。
- M3T pose 平滑且没有强视觉矛盾时，倾向继续保留 M3T。
- 深度一致性极低、黑白外观失败、投影异常、pose 不平滑且连续多帧出现时，认为 M3T 可能跟丢或误跟踪。
- 目标仍可能在画面内但 M3T 不可信时，进入 `REINIT_NEEDED`，请求 FoundationPose 重新初始化。
- 目标和 M3T 投影都缺少有效证据，并持续多帧时，进入 `LOST`，停止 M3T 实例。

这个设计是为了解决两个矛盾：

1. YOLO 不稳定，不能因为 YOLO 漏检就立刻杀死正常 M3T。
2. M3T 也可能跟丢、停在错误位置或跟到错误目标，不能无条件永久保留 M3T 实例。

## 当前问题

- 当前 YOLO 模型训练数据不足，对遮挡、侧面、特殊角度不稳定。
- 所有输出默认在相机坐标系下，需要额外手眼标定才能用于机器人执行坐标系。
- Tracking state filter 仍需要根据真实赛场数据继续调参。
- m3t在跟踪期间如果出现了偏差，可能会出现权重状态机没办法正确估计此刻m3t的状态的情况，导致没办法及时进行重新初始化的处理，这部分还是需要调整，后续将现在的权重状态机升级成卡尔曼进行维护

## 终语
25赛季大一刚入队的时候就是担任的工程视觉这一角色，当时还是简单的对矿石和兑换站进行特征识别，但是由于26赛季的规则大改导致了最终还是胎死腹中。后面做了许多东西，反小自瞄，大符自瞄，由于人手紧缺最终还去担任了工程的电控，但是也许是些许执念作祟还是想要将工程视觉做出来。结果由于RCIA26赛季完整形态挂了导致我的RM生涯也是就此结束了，可谓是令人感叹。此开源更多是想要纪念我的RM经历，同时希望可以实现抛砖引玉的效果。
