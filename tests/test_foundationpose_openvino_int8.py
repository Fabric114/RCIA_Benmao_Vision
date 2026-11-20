import sys
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

import numpy as np
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
FP32_IR_OK = refine_ir_path(WEIGHTS).exists() and score_ir_path(WEIGHTS).exists()


def _nncf_available():
    try:
        import nncf  # noqa: F401

        return True
    except Exception:
        return False


def _egl_available():
    try:
        import moderngl

        ctx = moderngl.create_standalone_context(backend="egl")
        ctx.release()
        return True
    except Exception:
        return False


NNCF_OK = _nncf_available()
EGL_OK = _egl_available()


class Int8PathTest(unittest.TestCase):
    def test_missing_int8_ir_raises_with_quantize_hint(self):
        with self.assertRaises(FileNotFoundError) as cm:
            OVModel("/nonexistent/refine_int8.xml", ("trans", "rot"))
        self.assertIn("quantize_foundationpose_openvino", str(cm.exception))
        self.assertNotIn("export_foundationpose_openvino", str(cm.exception))

    def test_missing_fp32_ir_still_hints_export(self):
        with self.assertRaises(FileNotFoundError) as cm:
            OVModel("/nonexistent/refine_fp32.xml", ("trans", "rot"))
        self.assertIn("export_foundationpose_openvino", str(cm.exception))

    def test_unknown_precision_raises(self):
        from scripts.foundationpose_factory import EstimatorConfig, build_estimator

        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=str(PROJECT_ROOT / "models/object/Str3D.obj"),
                                            device="cpu", render_backend="moderngl",
                                            net_backend="openvino", ov_precision="fp16"))


class CalibSinkTest(unittest.TestCase):
    def test_reservoir_keeps_capacity_and_single_rows(self):
        from scripts.quantize_foundationpose_openvino import _RefineReservoir

        sink = _RefineReservoir(capacity=5, seed=0)
        rng = np.random.default_rng(1)
        for _ in range(4):
            A = rng.random((20, 6, 160, 160), dtype=np.float32)
            B = rng.random((20, 6, 160, 160), dtype=np.float32)
            sink.append((A, B))
        self.assertEqual(sink.seen, 80)
        self.assertEqual(len(sink), 5)
        for a, b in sink:
            self.assertEqual(a.shape, (1, 6, 160, 160))
            self.assertEqual(a.dtype, np.float32)
            self.assertEqual(b.shape, (1, 6, 160, 160))

    def test_reservoir_is_deterministic_for_seed(self):
        from scripts.quantize_foundationpose_openvino import _RefineReservoir

        def run(seed):
            sink = _RefineReservoir(capacity=3, seed=seed)
            for i in range(10):
                x = np.full((2, 6, 4, 4), float(i), dtype=np.float32)
                sink.append((x, x))
            return [float(a[0, 0, 0, 0]) for a, _ in sink]

        self.assertEqual(run(7), run(7))


@skipUnless(WEIGHTS_OK and FP32_IR_OK and NNCF_OK, "needs weights, FP32 IR and nncf")
class QuantizeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.quantize_foundationpose_openvino import count_fake_quantize, quantize_ir

        cls.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)

        def mk(n):
            return (rng.random((n, 6, 160, 160), dtype=np.float32),
                    rng.random((n, 6, 160, 160), dtype=np.float32))

        cls.refine_batches = [mk(1) for _ in range(8)]
        cls.score_batches = [mk(20) for _ in range(4)]
        quantize_ir(refine_ir_path(WEIGHTS), cls.tmp / "refine_int8.xml", cls.refine_batches)
        quantize_ir(score_ir_path(WEIGHTS), cls.tmp / "score_int8.xml", cls.score_batches)
        quantize_ir(refine_ir_path(WEIGHTS), cls.tmp / "refine_int8_conv.xml", cls.refine_batches, conv_only=True)
        cls.count_fake_quantize = staticmethod(count_fake_quantize)

    def test_int8_ir_contains_fake_quantize(self):
        full = self.count_fake_quantize(self.tmp / "refine_int8.xml")
        conv = self.count_fake_quantize(self.tmp / "refine_int8_conv.xml")
        self.assertGreater(conv, 0)
        self.assertGreater(full, conv)
        self.assertEqual(self.count_fake_quantize(refine_ir_path(WEIGHTS)), 0)

    def test_int8_models_run_with_dynamic_batch(self):
        refine = OVModel(self.tmp / "refine_int8.xml", ("trans", "rot"))
        score = OVModel(self.tmp / "score_int8.xml", ("score_logit",))
        for n in (1, 20):
            A, B = torch.rand(n, 6, 160, 160), torch.rand(n, 6, 160, 160)
            out = refine(A, B)
            self.assertEqual(tuple(out["trans"].shape), (n, 3))
            self.assertTrue(torch.isfinite(out["rot"]).all())
            s = score(A, B, L=n)["score_logit"]
            self.assertEqual(tuple(s.shape), (1, n))
            self.assertTrue(torch.isfinite(s).all())


@skipUnless(WEIGHTS_OK and EGL_OK, "needs weights and EGL")
class CalibSinkRegisterTest(unittest.TestCase):
    def test_calib_sink_records_network_inputs(self):
        from scripts.foundationpose_factory import EstimatorConfig, build_estimator

        cfg = EstimatorConfig(mesh_file=str(PROJECT_ROOT / "models/object/Str3D.obj"), device="cpu",
                              render_backend="moderngl", net_backend="torch", symmetry_mode="z12")
        bundle = build_estimator(cfg)
        bundle.estimator.refiner.calib_sink = []
        bundle.estimator.scorer.calib_sink = []
        try:
            _register_fixture(bundle.estimator, iteration=5)
            refine_sink = bundle.estimator.refiner.calib_sink
            score_sink = bundle.estimator.scorer.calib_sink
            self.assertEqual(sum(a.shape[0] for a, _ in refine_sink), 20 * 5)
            self.assertEqual([a.shape for a, _ in score_sink], [(20, 6, 160, 160)])
            self.assertEqual(refine_sink[0][0].dtype, np.float32)
        finally:
            if hasattr(bundle.estimator.glctx, "release"):
                bundle.estimator.glctx.release()


def _register_fixture(estimator, iteration):
    import cv2

    fixture = PROJECT_ROOT / "tests/data/synthetic_str3d"
    color = cv2.imread(str(fixture / "color.png"), cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(color[..., ::-1])
    depth = cv2.imread(str(fixture / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    mask = depth > 0.001
    K = np.loadtxt(fixture / "K.txt")
    return estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)


if __name__ == "__main__":
    unittest.main()
