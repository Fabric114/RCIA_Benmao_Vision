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


if __name__ == "__main__":
    unittest.main()
