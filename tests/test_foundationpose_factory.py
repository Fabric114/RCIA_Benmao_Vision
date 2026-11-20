import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.foundationpose_factory import EstimatorConfig, build_estimator  # noqa: E402

MESH = str(PROJECT_ROOT / "models/object/Str3D.obj")


class RenderBackendValidationTest(unittest.TestCase):
    def test_nvdiffrast_requires_cuda(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, device="cpu", render_backend="nvdiffrast"))

    def test_unknown_render_backend(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, render_backend="pyrender"))

    def test_default_render_backend_is_nvdiffrast(self):
        self.assertEqual(EstimatorConfig(mesh_file=MESH).render_backend, "nvdiffrast")


class OpenVinoPrecisionValidationTest(unittest.TestCase):
    def test_fp16_requires_gpu(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, device="cpu", render_backend="moderngl",
                                            net_backend="openvino", ov_device="CPU", ov_precision="fp16"))

    def test_num_threads_rejected_on_gpu(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, device="cpu", render_backend="moderngl",
                                            net_backend="openvino", ov_device="GPU", ov_num_threads=4))

    def test_unknown_ov_precision(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, device="cpu", render_backend="moderngl",
                                            net_backend="openvino", ov_precision="bf16"))

    def test_default_ov_settings(self):
        cfg = EstimatorConfig(mesh_file=MESH)
        self.assertEqual(cfg.ov_precision, "fp32")
        self.assertIsNone(cfg.ov_num_threads)


if __name__ == "__main__":
    unittest.main()
