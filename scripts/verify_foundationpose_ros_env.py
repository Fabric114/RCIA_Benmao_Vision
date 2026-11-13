#!/usr/bin/env python3
import argparse
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

OPENVINO_VERSION_PREFIX = "2024.6.0"
DEFAULT_OV_DEVICE = "CPU"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models/yolo/best_openvino_model"


def require_import(name):
    module = importlib.import_module(name)
    print(f"{name}: OK")
    return module


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check_openvino(ov):
    if not ov.__version__.startswith(OPENVINO_VERSION_PREFIX):
        fail(f"openvino {ov.__version__}，需要 {OPENVINO_VERSION_PREFIX}")
    core = ov.Core()
    devices = core.available_devices
    print(f"openvino devices: {devices}")
    if DEFAULT_OV_DEVICE == "GPU":
        if "GPU" not in devices:
            fail("默认 ov_device 为 GPU，但 OpenVINO 没有找到 GPU 设备")
        print(f"GPU FULL_DEVICE_NAME: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")


def check_ir_files():
    from learning.ov_model import WEIGHTS_ROOT, refine_ir_path, score_ir_path

    for path in (refine_ir_path(WEIGHTS_ROOT, "fp32"), score_ir_path(WEIGHTS_ROOT, "fp32")):
        if not path.exists():
            fail(f"缺少 OpenVINO IR {path}，请运行 python scripts/export_foundationpose_openvino.py")
        print(f"OpenVINO IR: {path}")


def check_yolo_model():
    if not list(DEFAULT_YOLO_MODEL.glob("*.xml")) or not (DEFAULT_YOLO_MODEL / "metadata.yaml").exists():
        fail(f"{DEFAULT_YOLO_MODEL} 下缺少 *.xml 或 metadata.yaml，请导出 OpenVINO 格式的 YOLO 模型")
    print(f"YOLO OpenVINO model: {DEFAULT_YOLO_MODEL}")


def check_moderngl(allow_software_gl):
    from moderngl_renderer import ModernglBatchRenderer, is_software_renderer

    renderer = ModernglBatchRenderer(backend="egl")
    try:
        print(f"GL_RENDERER: {renderer.gl_renderer}")
        if is_software_renderer(renderer.gl_renderer):
            message = (
                f"EGL 使用软件渲染 ({renderer.gl_renderer})。"
                "请检查当前用户是否在 render 组（sudo usermod -aG render $USER，然后重新登录）。"
            )
            if allow_software_gl:
                print(f"WARNING: {message}")
            else:
                fail(message)
    finally:
        renderer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow_software_gl",
        action="store_true",
        help="EGL 退化为软件渲染时只打印 WARNING，不返回失败",
    )
    args = parser.parse_args()

    print(f"python: {sys.version}")
    require_import("rclpy")
    torch = require_import("torch")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    require_import("torchvision")
    require_import("cv2")
    require_import("ultralytics")
    require_import("pyrealsense2")
    require_import("nvdiffrast.torch")
    require_import("pytorch3d.transforms")
    require_import("mycpp.build.mycpp")
    require_import("bundlesdf.mycuda.common")
    require_import("estimater")
    ov = require_import("openvino")
    require_import("moderngl")
    require_import("scripts.foundationpose_init")
    require_import("ros2.rcia_energy_trace_ros.scripts.foundationpose_init_node")

    check_openvino(ov)
    check_ir_files()
    check_yolo_model()
    check_moderngl(args.allow_software_gl)
    print("FOUNDATIONPOSE_ROS_ENV_OK")


if __name__ == "__main__":
    main()
