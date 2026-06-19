#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POSE_FILE="${PROJECT_ROOT}/runtime/foundation_pose.txt"
M3T_BINARY="${PROJECT_ROOT}/build/run_m3t_tracking"

rm -f "${POSE_FILE}"

echo "启动 FoundationPose 初始化。检测到目标后按 SPACE 注册初始位姿。"
python3 "${PROJECT_ROOT}/scripts/foundationpose_init.py" --init_only --pose_output "${POSE_FILE}"

if [ ! -x "${M3T_BINARY}" ]; then
  echo "M3T 跟踪程序不存在，开始构建。"
  cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" -DCMAKE_BUILD_TYPE=Release -DUSE_REALSENSE=ON -DUSE_GTEST=OFF
  cmake --build "${PROJECT_ROOT}/build" -j"$(nproc)"
fi

echo "启动 M3T 跟踪。"
"${M3T_BINARY}"
