import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.foundationpose_factory import EstimatorConfig, build_estimator, z_symmetry_tfs  # noqa: E402

MESH = str(PROJECT_ROOT / "models/object/Str3D.obj")
WEIGHTS_OK = all(
    (PROJECT_ROOT / "foundation_pose/weights" / run / "model_best.pth").exists()
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45")
)


class ZSymmetryTfsTest(unittest.TestCase):
    def test_z_symmetry_tfs_shape_and_axis(self):
        tfs = z_symmetry_tfs()
        self.assertEqual(tfs.shape, (12, 4, 4))
        for i, tf in enumerate(tfs):
            np.testing.assert_allclose(tf[:3, 2], [0, 0, 1], atol=1e-12)
            np.testing.assert_allclose(tf[:3, 3], 0, atol=1e-12)
            self.assertAlmostEqual(np.degrees(np.arctan2(tf[1, 0], tf[0, 0])) % 360, 30.0 * i, places=6)


class UnknownSymmetryModeTest(unittest.TestCase):
    def test_unknown_symmetry_mode_raises(self):
        with self.assertRaises(ValueError):
            build_estimator(EstimatorConfig(mesh_file=MESH, symmetry_mode="z6"))


@unittest.skipUnless(torch.cuda.is_available() and WEIGHTS_OK, "needs CUDA + weights")
class RotGridSizeTest(unittest.TestCase):
    def test_rot_grid_size(self):
        z12 = build_estimator(EstimatorConfig(mesh_file=MESH, symmetry_mode="z12"))
        self.assertEqual(len(z12.estimator.rot_grid), 20)
        none = build_estimator(EstimatorConfig(mesh_file=MESH))
        self.assertEqual(len(none.estimator.rot_grid), 252)


if __name__ == "__main__":
    unittest.main()
