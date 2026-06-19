#!/bin/bash
# 测试可视化功能的脚本

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "可视化功能测试脚本"
echo "=========================================="
echo ""
echo "此脚本将测试三个可视化功能："
echo "1. YOLO检测可视化 - 显示检测框和置信度"
echo "2. FoundationPose初始化可视化 - 显示初始位姿、坐标系和遮罩轮廓"
echo "3. M3T跟踪可视化 - 在模型重心显示三维坐标系"
echo ""
echo "=========================================="
echo ""

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "foundationpose" ]]; then
    echo "错误: 请先激活foundationpose环境"
    echo "运行: conda activate foundationpose"
    exit 1
fi

# 设置环境变量
source scripts/foundationpose_env.sh

# 清理旧的位姿文件
if [ -f "runtime/foundation_pose.txt" ]; then
    echo "清理旧的位姿文件..."
    rm -f runtime/foundation_pose.txt
fi

echo "=========================================="
echo "阶段1: YOLO检测 + FoundationPose初始化"
echo "=========================================="
echo ""
echo "操作说明："
echo "- 将看到实时相机画面和YOLO检测框"
echo "- 检测到目标后会自动进行初始化（或按空格键手动确认）"
echo "- 初始化完成后会显示："
echo "  * 3D边界框"
echo "  * 三维坐标系（X=红, Y=绿, Z=蓝）"
echo "  * SAM分割遮罩轮廓（青色）"
echo "- 按任意键继续到下一阶段"
echo ""
read -p "按回车键开始..." dummy

# 运行FoundationPose初始化（带可视化）
python scripts/foundationpose_init.py \
    --init_only \
    --target_class 0 \
    --detect_conf 0.75 \
    --auto_register \
    --auto_register_stable_frames 3 \
    --debug 1

if [ ! -f "runtime/foundation_pose.txt" ]; then
    echo ""
    echo "错误: 初始化失败，未生成位姿文件"
    exit 1
fi

echo ""
echo "=========================================="
echo "阶段2: M3T实时跟踪（带三维坐标系）"
echo "=========================================="
echo ""
echo "操作说明："
echo "- 将看到彩色模型叠加在实时画面上"
echo "- 模型重心会显示三维坐标系："
echo "  * X轴 = 红色箭头"
echo "  * Y轴 = 绿色箭头"
echo "  * Z轴 = 蓝色箭头"
echo "- 观察坐标系可以判断："
echo "  * 旋转是否正确"
echo "  * 是否有抖动"
echo "- 按ESC键退出跟踪"
echo ""
read -p "按回车键开始M3T跟踪..." dummy

# 运行M3T跟踪
./build/run_m3t_tracking

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo ""
echo "如果发现问题："
echo "1. YOLO误识别 -> 调整 --detect_conf 参数"
echo "2. FoundationPose初始化不准 -> 检查遮罩质量和深度数据"
echo "3. M3T跟踪抖动 -> 可能是光照变化或遮挡导致"
echo ""
