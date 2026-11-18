#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${PROJ_ROOT}/.." &> /dev/null && pwd )

source "${PROJECT_ROOT}/scripts/foundationpose_env.sh"

PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
PYBIND11_DIR=$("${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())")

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. -Dpybind11_DIR="${PYBIND11_DIR}" -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" && \
make -j$(nproc)

cd ${PROJ_ROOT}
