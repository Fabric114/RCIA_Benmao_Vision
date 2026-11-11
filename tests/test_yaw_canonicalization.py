import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.yaw_canonicalization import (  # noqa: E402
    YawCanonicalizer,
    canonicalize_yaw,
    mesh_center_from_vertices,
)

C = np.array([0.0475, 0.0475, -0.075])


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def make_pose(z_axis, x_hint, center_cam):
    z = unit(z_axis)
    hint = np.asarray(x_hint, dtype=np.float64)
    x = unit(hint - np.dot(hint, z) * z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = np.asarray(center_cam, dtype=np.float64) - R @ C
    return pose


def random_pose(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = Q
    pose[:3, 3] = [0.1, -0.05, 0.5]
    return pose


class CanonicalizeYawTest(unittest.TestCase):
    def test_upright_facing_camera_x_points_to_camera(self):
        out = canonicalize_yaw(make_pose([0, -1, 0], [1, 0, 0], [0, 0, 0.5]), C)
        np.testing.assert_allclose(out[:3, 0], [0, 0, -1], atol=1e-9)

    def test_center_and_long_axis_preserved(self):
        p = make_pose(unit([0.3, -0.8, 0.2]), [1, 0, 0], [0.05, 0.02, 0.6])
        out = canonicalize_yaw(p, C)
        np.testing.assert_allclose(out[:3, :3] @ C + out[:3, 3], p[:3, :3] @ C + p[:3, 3], atol=1e-12)
        np.testing.assert_allclose(out[:3, 2], p[:3, 2], atol=1e-12)

    def test_output_is_rotation(self):
        R = canonicalize_yaw(random_pose(seed=1), C)[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_invariant_to_input_yaw(self):
        a = canonicalize_yaw(make_pose(unit([0.3, -0.8, 0.2]), [1, 0, 0], [0, 0, 0.5]), C)
        b = canonicalize_yaw(make_pose(unit([0.3, -0.8, 0.2]), [0, 0, 1], [0, 0, 0.5]), C)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_end_on_uses_minus_y(self):
        for z in ([0, 0, 1], [0, 0, -1]):
            out = canonicalize_yaw(make_pose(z, [1, 0, 0], [0, 0, 0.5]), C)
            np.testing.assert_allclose(out[:3, 0], [0, -1, 0], atol=1e-9)

    def test_threshold_switches_at_15_degrees(self):
        def z_at(deg):
            r = np.radians(deg)
            return [np.sin(r), 0.0, -np.cos(r)]

        x16 = canonicalize_yaw(make_pose(z_at(16), [0, 1, 0], [0, 0, 0.5]), C)[:3, 0]
        x14 = canonicalize_yaw(make_pose(z_at(14), [0, 1, 0], [0, 0, 0.5]), C)[:3, 0]
        self.assertAlmostEqual(x16[1], 0.0, places=9)
        self.assertAlmostEqual(x14[1], -1.0, places=9)

    def test_mesh_center_from_vertices(self):
        v = np.array([[0, 0, -0.15], [0.095, 0.095, 0.0]])
        np.testing.assert_allclose(mesh_center_from_vertices(v), C, atol=1e-12)


class YawCanonicalizerTest(unittest.TestCase):
    def test_canonicalizer_reuses_previous_x_in_fallback_zone(self):
        p_normal = make_pose(unit([0.6, 0, -0.8]), [0, 1, 0], [0, 0, 0.5])
        p_endon = make_pose(unit([0.05, 0, -1]), [0, 1, 0], [0, 0, 0.5])
        c = YawCanonicalizer(C)
        prev_x = c.apply(p_normal)[:3, 0]
        out_x = c.apply(p_endon)[:3, 0]
        self.assertAlmostEqual(out_x[1], 0.0, places=9)
        self.assertGreater(np.dot(out_x, prev_x), 0.8)
        c.reset()
        np.testing.assert_allclose(c.apply(p_endon)[:3, 0], [0, -1, 0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
