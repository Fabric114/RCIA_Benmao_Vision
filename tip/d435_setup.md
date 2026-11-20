# D435 相机到手前的准备

## 1. 安装 librealsense2

建议用 Intel 官方 apt 源，Ubuntu 默认源里的版本可能偏旧。添加源的步骤见官方文档：
https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

核心包：

```bash
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev
```

本机内核是 6.8，`librealsense2-dkms` 有可能编不过。编不过也没关系，D435 走标准 UVC 一般也能用，只是部分元数据功能会缺失。

## 2. 用 realsense-viewer 确认硬件

插上相机后先运行：

```bash
realsense-viewer
```

确认能看到彩色和深度画面。一定要插 USB 3 口，插 USB 2 的话深度流会受限甚至打不开。

## 3. 单独跑项目的相机节点

先编译 ROS2 包。`CMakeLists.txt` 里写了 `find_package(realsense2 REQUIRED)`，没装 `librealsense2-dev` 的话编译会直接失败。

```bash
mkdir -p ~/energy_trace_ws/src
ln -s ~/RCIA_Benmao_Vision/ros2/rcia_energy_trace_ros ~/energy_trace_ws/src/
cd ~/energy_trace_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select rcia_energy_trace_ros
source install/setup.bash
```

然后启动相机节点，确认话题正常：

```bash
ros2 run rcia_energy_trace_ros realsense_rgbd_publisher_node --ros-args -p show_preview:=true
ros2 topic hz /camera/color/image_raw
```
