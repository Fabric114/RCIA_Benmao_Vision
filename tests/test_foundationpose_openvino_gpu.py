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


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 6, 160, 160, generator=g), torch.rand(n, 6, 160, 160, generator=g)


@skipUnless(WEIGHTS_OK and INTEL_OV_GPU, "needs FoundationPose weights and Intel GPU in OpenVINO")
class GpuStaticShapeTest(unittest.TestCase):
    SEQ = (20, 1, 3, 20, 20)  # batch 大小来回切换，每个 N 的输入固定（seed 0）

    @classmethod
    def setUpClass(cls):
        from scripts.export_foundationpose_openvino import export_all
        from scripts.foundationpose_factory import PoseRefinePredictor, ScorePredictor

        cls.tmp = tempfile.mkdtemp()
        export_all(weights_root=WEIGHTS, out_root=cls.tmp)
        cls.refine_t = PoseRefinePredictor(device="cpu").model
        cls.score_t = ScorePredictor(device="cpu").model

    def _refine(self, precision_hint="f32"):
        return OVModel(refine_ir_path(self.tmp), ("trans", "rot"), device="GPU", precision_hint=precision_hint)

    def _score(self, precision_hint="f32"):
        return OVModel(score_ir_path(self.tmp), ("score_logit",), device="GPU", precision_hint=precision_hint)

    def test_gpu_fp32_matches_torch(self):
        ov = self._refine()
        for n in self.SEQ:
            A, B = inputs(n)
            ref, out = self.refine_t(A, B), ov(A, B)
            for k in ("trans", "rot"):
                torch.testing.assert_close(out[k], ref[k], atol=1e-3, rtol=1e-3)
        ov = self._score()
        for n in self.SEQ:
            A, B = inputs(n)
            torch.testing.assert_close(ov(A, B, L=n)["score_logit"],
                                       self.score_t(A, B, L=n)["score_logit"], atol=1e-3, rtol=1e-3)

    def test_gpu_fp16_close_to_torch(self):
        ov = self._refine(precision_hint="f16")
        for n in self.SEQ:
            A, B = inputs(n)
            ref, out = self.refine_t(A, B), ov(A, B)
            for k in ("trans", "rot"):
                torch.testing.assert_close(out[k], ref[k], atol=0.05, rtol=0.05)
        ov = self._score(precision_hint="f16")
        for n in self.SEQ:
            A, B = inputs(n)
            ref = self.score_t(A, B, L=n)["score_logit"]
            out = ov(A, B, L=n)["score_logit"]
            torch.testing.assert_close(out, ref, atol=0.2, rtol=0.2)
            self.assertEqual(int(out.argmax()), int(ref.argmax()))

    def test_gpu_static_shape_is_repeatable(self):
        A, B = inputs(20)
        for hint in ("f32", "f16"):
            ov = self._score(precision_hint=hint)
            runs = [ov(A, B, L=20)["score_logit"] for _ in range(5)]
            for r in runs[1:]:
                self.assertLessEqual(float((r - runs[0]).abs().max()), 1e-4, hint)

    def test_warmup_compiles_each_batch_size(self):
        ov = self._refine()
        ov.warmup((20, 1))
        self.assertEqual(sorted(ov.compiled_batch_sizes), [1, 20])


if __name__ == "__main__":
    unittest.main()
