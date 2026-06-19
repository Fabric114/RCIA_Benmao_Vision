#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

export RCIA_ENERGY_TRACE_ROOT="${PROJECT_ROOT}"
export MPLCONFIGDIR="${PROJECT_ROOT}/runtime/matplotlib"
export YOLO_CONFIG_DIR="${PROJECT_ROOT}/runtime/ultralytics"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONNOUSERSITE=1

if [[ -z "${CUDA_HOME:-}" ]]; then
  for CUDA_CANDIDATE in \
    "${CONDA_PREFIX}" \
    "/usr/local/cuda"; do
    if [[ -x "${CUDA_CANDIDATE}/bin/nvcc" && -f "${CUDA_CANDIDATE}/include/cuda.h" ]]; then
      export CUDA_HOME="${CUDA_CANDIDATE}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "foundationpose_env.sh: please activate the foundationpose conda env first" >&2
  return 1 2>/dev/null || exit 1
fi

PYTHON_SITE_SUFFIX="$(python - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
PY
)"
SITE_PACKAGES="${CONDA_PREFIX}/lib/${PYTHON_SITE_SUFFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
if [[ -d "${SITE_PACKAGES}/nvidia" ]]; then
  CUDA_LIBS="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type d -name lib | paste -sd: -)"
  CUDA_INCS="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type d -name include | paste -sd: -)"
  export LD_LIBRARY_PATH="${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_LIBS}:${LIBRARY_PATH:-}"
  export CPATH="${CUDA_INCS}:${CPATH:-}"
fi

mkdir -p "${MPLCONFIGDIR}" "${YOLO_CONFIG_DIR}" "${PROJECT_ROOT}/runtime"
