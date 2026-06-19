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

# Install nvdiffrast
cd ${PROJ_ROOT}/nvdiffrast && \
"${PYTHON_EXECUTABLE}" -m pip install --no-build-isolation -e .

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
"${PYTHON_EXECUTABLE}" -m pip install -e .

cd ${PROJ_ROOT}
