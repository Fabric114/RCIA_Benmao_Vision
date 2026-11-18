#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
ENV_NAME="${ENV_NAME:-foundationpose_ros}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found" >&2
  exit 1
fi

CONDA_SOLVER="${CONDA_SOLVER:-conda}"
CREATE_FROM_BASE="${CREATE_FROM_BASE:-0}"
if [[ "${CONDA_SOLVER}" == "mamba" ]] && command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="$(command -v mamba)"
else
  CONDA_BIN="$(command -v conda)"
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${CREATE_FROM_BASE}" == "1" ]]; then
    conda create -y -n "${ENV_NAME}" --clone base
  else
    "${CONDA_BIN}" create -y \
      --solver classic \
      --override-channels \
      -c conda-forge \
      -n "${ENV_NAME}" \
      "python=${PYTHON_VERSION}" \
      pip
  fi
fi

eval "$(conda shell.bash hook)"
set +u
conda activate "${ENV_NAME}"
source /opt/ros/humble/setup.bash
set -u
source "${PROJECT_ROOT}/scripts/foundationpose_env.sh"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake was not found. Install the system packages in tip/tip.md step 1 first." >&2
  exit 1
fi
for pkg in libboost-system-dev libboost-program-options-dev libeigen3-dev; do
  if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
    echo "${pkg} is not installed. Install the system packages in tip/tip.md step 1 first." >&2
    exit 1
  fi
done

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121

SITE_PACKAGES="${CONDA_PREFIX}/lib/python$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)/site-packages"
python -m pip uninstall -y numpy scipy || true
rm -rf \
  "${SITE_PACKAGES}/numpy" \
  "${SITE_PACKAGES}/numpy.libs" \
  "${SITE_PACKAGES}"/numpy-*.dist-info \
  "${SITE_PACKAGES}/scipy" \
  "${SITE_PACKAGES}/scipy.libs" \
  "${SITE_PACKAGES}"/scipy-*.dist-info

python -m pip install \
  numpy==1.26.4 \
  pybind11==2.12.0 \
  opencv-python==4.9.0.80 \
  opencv-contrib-python==4.9.0.80 \
  scipy==1.11.4 \
  scikit-learn==1.4.1.post1 \
  scikit-image==0.22.0 \
  imageio==2.34.0 \
  trimesh==4.12.2 \
  open3d==0.19.0 \
  omegaconf==2.3.0 \
  ruamel.yaml==0.18.6 \
  PyYAML==6.0.1 \
  kornia==0.7.2 \
  "moderngl>=5.10" \
  openvino==2024.6.0 \
  transformations==2024.6.1 \
  pandas \
  matplotlib \
  Pillow \
  joblib \
  threadpoolctl \
  psutil \
  tqdm \
  ultralytics==8.4.53 \
  ultralytics-thop \
  pyrealsense2==2.56.5.9235 \
  h5py \
  pyrender==0.1.45 \
  "PyOpenGL>=3.1.0" \
  pyglet==1.5.28 \
  rtree==1.2.0 \
  xatlas==0.0.9 \
  pysdf==0.1.9 \
  einops==0.7.0 \
  timm==0.9.16 \
  warp-lang==1.0.2 \
  fvcore==0.1.5.post20221221 \
  typeguard \
  coloredlogs \
  flatbuffers \
  protobuf \
  lazy_loader \
  tifffile \
  huggingface_hub \
  safetensors \
  tabulate

CV2_DIR="${SITE_PACKAGES}/cv2"
if [[ -d "${CV2_DIR}/python-3.10" ]]; then
  rm -rf "${CV2_DIR}/python-3.10.conda-disabled"
  mv "${CV2_DIR}/python-3.10" "${CV2_DIR}/python-3.10.conda-disabled"
fi
if [[ -f "${CV2_DIR}/config-3.10.py" ]]; then
  if [[ ! -f "${CV2_DIR}/config-3.10.py.conda-disabled" ]]; then
    cp "${CV2_DIR}/config-3.10.py" "${CV2_DIR}/config-3.10.py.conda-disabled"
  fi
  printf '%s\n' 'PYTHON_EXTENSIONS_PATHS = [LOADER_DIR] + PYTHON_EXTENSIONS_PATHS' > "${CV2_DIR}/config-3.10.py"
fi

bash "${PROJECT_ROOT}/foundation_pose/build_all_conda.sh"

python "${PROJECT_ROOT}/scripts/export_foundationpose_openvino.py"

python "${PROJECT_ROOT}/scripts/verify_foundationpose_ros_env.py"
