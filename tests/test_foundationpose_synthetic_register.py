import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))

from scripts.pose_eval_metrics import pose_errors  # noqa: E402

FIXTURE_DIR = PROJECT_ROOT / "tests/data/synthetic_str3d"
MESH = str(PROJECT_ROOT / "models/object/Str3D.obj")
WEIGHTS_OK = all(
    (PROJECT_ROOT / "foundation_pose/weights" / run / "model_best.pth").exists()
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45")
)


def run_synthetic_register(cfg):
    """返回 (中心误差 m, 长轴夹角 deg, last_timing_ms)。"""
    from scripts.foundationpose_factory import build_estimator

    color = cv2.imread(str(FIXTURE_DIR / "color.png"), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(color[..., ::-1])
    depth = cv2.imread(str(FIXTURE_DIR / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    mask = depth > 0.001
    K = np.loadtxt(FIXTURE_DIR / "K.txt")
    pose_gt = np.loadtxt(FIXTURE_DIR / "pose.txt")

    bundle = build_estimator(cfg)
    pose = bundle.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    c, a = pose_errors(pose_gt, pose, bundle.model_center)
    return c, a, dict(bundle.estimator.last_timing_ms)


@unittest.skipUnless(torch.cuda.is_available() and WEIGHTS_OK, "needs CUDA + weights")
class SyntheticRegisterTest(unittest.TestCase):
    def test_baseline_cuda(self):
        from scripts.foundationpose_factory import EstimatorConfig

        c, a, timing = run_synthetic_register(EstimatorConfig(mesh_file=MESH))
        print("center_err_m", c, "axis_err_deg", a)
        print("timing_ms", timing)
        self.assertLessEqual(c, 0.005)
        self.assertLessEqual(a, 5.0)

    def test_z12_cuda(self):
        from scripts.foundationpose_factory import EstimatorConfig

        c, a, timing = run_synthetic_register(EstimatorConfig(mesh_file=MESH, symmetry_mode="z12"))
        print("z12 center_err_m", c, "axis_err_deg", a)
        print("z12 timing_ms", timing)
        self.assertLessEqual(c, 0.005)
        self.assertLessEqual(a, 5.0)


if __name__ == "__main__":
    unittest.main()
