#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "foundationpose" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Activate the foundationpose environment first." >&2
    exit 1
  fi
  exec conda run -n foundationpose bash "$0" "$@"
fi

cd "${PROJECT_ROOT}"
source scripts/foundationpose_env.sh

POSE_FILE="${PROJECT_ROOT}/runtime/foundation_pose.txt"
TARGET_CLASS="${TARGET_CLASS:-0}"
DETECT_CONF="${DETECT_CONF:-0.75}"
REGISTER_CONF="${REGISTER_CONF:-${DETECT_CONF}}"
STABLE_FRAMES="${STABLE_FRAMES:-3}"
EST_REFINE_ITER="${EST_REFINE_ITER:-2}"
TRACK_REFINE_ITER="${TRACK_REFINE_ITER:-2}"

if [[ ! -x "${PROJECT_ROOT}/build/run_m3t_tracking" ]]; then
  echo "M3T executable is missing: build/run_m3t_tracking" >&2
  echo "Build it with: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_REALSENSE=ON -DUSE_GTEST=OFF && cmake --build build -j\"\$(nproc)\"" >&2
  exit 1
fi

rm -f "${POSE_FILE}"

echo "Starting automatic FoundationPose registration"
echo "  target_class=${TARGET_CLASS}"
echo "  detect_conf=${DETECT_CONF}"
echo "  register_conf=${REGISTER_CONF}"
echo "  stable_frames=${STABLE_FRAMES}"

python scripts/foundationpose_init.py \
  --init_only \
  --auto_register \
  --target_class "${TARGET_CLASS}" \
  --detect_conf "${DETECT_CONF}" \
  --register_conf "${REGISTER_CONF}" \
  --auto_register_stable_frames "${STABLE_FRAMES}" \
  --est_refine_iter "${EST_REFINE_ITER}" \
  --track_refine_iter "${TRACK_REFINE_ITER}"

if [[ ! -s "${POSE_FILE}" ]]; then
  echo "FoundationPose did not produce ${POSE_FILE}" >&2
  exit 1
fi

echo "FoundationPose pose written to ${POSE_FILE}"
echo "Starting M3T tracking"
exec "${PROJECT_ROOT}/build/run_m3t_tracking"
