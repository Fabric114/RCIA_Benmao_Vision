import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))

from learning.rotation_utils import rotation_6d_to_matrix, so3_exp_map  # noqa: E402

try:
    import pytorch3d.transforms as p3d
    PYTORCH3D_OK = True
except (ImportError, OSError):
    PYTORCH3D_OK = False


class RotationUtilsTest(unittest.TestCase):
    def test_zero_log_is_identity(self):
        np.testing.assert_allclose(so3_exp_map(torch.zeros(2, 3)).numpy(), np.tile(np.eye(3), (2, 1, 1)), atol=1e-7)

    def test_outputs_are_rotations(self):
        for R in (so3_exp_map(torch.randn(64, 3)), rotation_6d_to_matrix(torch.randn(64, 6))):
            np.testing.assert_allclose((R.transpose(1, 2) @ R).numpy(), np.tile(np.eye(3), (64, 1, 1)), atol=1e-5)
            np.testing.assert_allclose(torch.det(R).numpy(), 1.0, atol=1e-5)

    @unittest.skipUnless(PYTORCH3D_OK, "needs pytorch3d")
    def test_matches_pytorch3d(self):
        x = torch.randn(256, 3) * 2
        d = torch.randn(256, 6)
        torch.testing.assert_close(so3_exp_map(x), p3d.so3_exp_map(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rotation_6d_to_matrix(d), p3d.rotation_6d_to_matrix(d), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
