#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

SAM_DIR="${PROJECT_ROOT}/models/sam"
FP_WEIGHTS_DIR="${PROJECT_ROOT}/foundation_pose/weights"

mkdir -p "${SAM_DIR}" "${FP_WEIGHTS_DIR}"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "exists: ${output}"
    return
  fi
  echo "download: ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  else
    echo "wget or curl is required to download ${url}" >&2
    exit 1
  fi
}

download_foundationpose_weights() {
  if [[ -f "${FP_WEIGHTS_DIR}/2023-10-28-18-33-37/model_best.pth" && \
        -f "${FP_WEIGHTS_DIR}/2024-01-11-20-02-45/model_best.pth" ]]; then
    echo "exists: FoundationPose weights"
    return
  fi

  if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
  then
    :
  else
    echo "huggingface_hub is required for FoundationPose weights." >&2
    echo "Install it with: python3 -m pip install huggingface_hub" >&2
    echo "Official Google Drive folder: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing" >&2
    exit 1
  fi

  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="gpue/foundationpose-weights",
    local_dir=r"${FP_WEIGHTS_DIR}",
    local_dir_use_symlinks=False,
)
PY
}

download_file \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  "${SAM_DIR}/sam_vit_b_01ec64.pth"

download_foundationpose_weights

echo "Public weights are ready."
