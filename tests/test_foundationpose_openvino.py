import sys
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))
sys.path.insert(0, str(PROJECT_ROOT))

from learning.ov_model import OVModel, refine_ir_path, score_ir_path  # noqa: E402

WEIGHTS = PROJECT_ROOT / "foundation_pose/weights"
WEIGHTS_OK = all(
    (WEIGHTS / run / "model_best.pth").exists()
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45")
)


def _intel_gpu_available():
    try:
        import openvino as ov

        core = ov.Core()
        return "GPU" in core.available_devices and "Intel" in core.get_property("GPU", "FULL_DEVICE_NAME")
    except Exception:
        return False


INTEL_OV_GPU = _intel_gpu_available()


@skipUnless(WEIGHTS_OK, "needs FoundationPose weights")
class OpenVinoParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.export_foundationpose_openvino import export_all
        from scripts.foundationpose_factory import PoseRefinePredictor, ScorePredictor

        cls.tmp = tempfile.mkdtemp()
        export_all(weights_root=WEIGHTS, out_root=cls.tmp)
        cls.refine_t = PoseRefinePredictor(device="cpu").model
        cls.score_t = ScorePredictor(device="cpu").model

    def test_refine_matches_torch(self):
        ov = OVModel(refine_ir_path(self.tmp), ("trans", "rot"))
        for n in (1, 3, 20):
            A, B = torch.rand(n, 6, 160, 160), torch.rand(n, 6, 160, 160)
            ref = self.refine_t(A, B)
            out = ov(A, B)
            for k in ("trans", "rot"):
                torch.testing.assert_close(out[k], ref[k], atol=1e-3, rtol=1e-3)

    def test_score_matches_torch(self):
        ov = OVModel(score_ir_path(self.tmp), ("score_logit",))
        for n in (1, 3, 20):
            A, B = torch.rand(n, 6, 160, 160), torch.rand(n, 6, 160, 160)
            torch.testing.assert_close(
                ov(A, B, L=n)["score_logit"],
                self.score_t(A, B, L=n)["score_logit"],
                atol=1e-3,
                rtol=1e-3,
            )

    @skipUnless(INTEL_OV_GPU, "needs Intel GPU in OpenVINO")
    def test_refine_on_intel_gpu(self):
        ov = OVModel(refine_ir_path(self.tmp), ("trans", "rot"), device="GPU")
        A, B = torch.rand(3, 6, 160, 160), torch.rand(3, 6, 160, 160)
        ref = self.refine_t(A, B)
        out = ov(A, B)
        for k in ("trans", "rot"):
            torch.testing.assert_close(out[k], ref[k], atol=1e-3, rtol=1e-3)


class OpenVinoPathTest(unittest.TestCase):
    def test_missing_ir_raises_with_export_hint(self):
        with self.assertRaises(FileNotFoundError) as cm:
            OVModel("/nonexistent/refine_fp32.xml", ("trans", "rot"))
        self.assertIn("export_foundationpose_openvino", str(cm.exception))

    def test_ir_path_precision(self):
        self.assertEqual(refine_ir_path("/w").name, "refine_fp32.xml")
        self.assertEqual(refine_ir_path("/w", "int8").name, "refine_int8.xml")
        self.assertEqual(score_ir_path("/w", "int8").name, "score_int8.xml")
        with self.assertRaises(ValueError):
            refine_ir_path("/w", "fp16")


if __name__ == "__main__":
    unittest.main()
