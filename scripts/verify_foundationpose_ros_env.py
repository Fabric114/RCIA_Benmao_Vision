#!/usr/bin/env python3
import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "RCIA_ENERGY_TRACE_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "foundation_pose"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FOUNDATIONPOSE_DIR))


def require_import(name):
    module = importlib.import_module(name)
    print(f"{name}: OK")
    return module


def main():
    print(f"python: {sys.version}")
    require_import("rclpy")
    torch = require_import("torch")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    require_import("torchvision")
    require_import("cv2")
    require_import("ultralytics")
    require_import("segment_anything")
    require_import("pyrealsense2")
    require_import("nvdiffrast.torch")
    require_import("pytorch3d.transforms")
    require_import("mycpp.build.mycpp")
    require_import("bundlesdf.mycuda.common")
    require_import("estimater")
    require_import("scripts.foundationpose_init")
    require_import("ros2.rcia_energy_trace_ros.scripts.foundationpose_init_node")
    print("FOUNDATIONPOSE_ROS_ENV_OK")


if __name__ == "__main__":
    main()
