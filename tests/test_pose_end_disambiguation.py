import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pose_end_disambiguation import (  # noqa: E402
    MIN_VALID_DEPTH_M,
    count_points_in_sphere,
    disambiguate_zone_pose,
    flip_pose_around_center,
    mesh_endpoints_from_vertices,
)

STR3D = PROJECT_ROOT / "models/object/Str3D.obj"
STR3D_BBOX_CENTER = PROJECT_ROOT / "models/object/Str3D_bbox_center.obj"


def load_obj_vertices(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            parts = line.split()
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=np.float64)


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_pose(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def bruteforce_count(depth_m, K, center_cam, radius_m):
    """Back-project every pixel of the full image and count points inside the sphere."""
    depth = np.asarray(depth_m, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    valid = np.isfinite(depth) & (depth >= MIN_VALID_DEPTH_M)
    z = np.where(valid, depth, 0.0)
    rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    x = (cols - px) * z / fx
    y = (rows - py) * z / fy
    cx, cy, cz = center_cam
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return int(np.count_nonzero(valid & (d2 <= radius_m ** 2)))


def make_intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def inverted_mesh_vertices():
    """Large cap (r=0.05) at +z and small cap (r=0.02) at -z, with a thin body between."""
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    large_ring = np.column_stack([0.05 * np.cos(angles), 0.05 * np.sin(angles), np.full(24, 0.1)])
    small_ring = np.column_stack([0.02 * np.cos(angles), 0.02 * np.sin(angles), np.full(24, -0.1)])
    body = np.column_stack([0.03 * np.cos(angles), 0.03 * np.sin(angles), np.zeros(24)])
    centers = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    return np.vstack([large_ring, small_ring, body, centers])


class MeshEndpointsTest(unittest.TestCase):
    def test_mesh_endpoints_str3d_and_centered(self):
        vertices = load_obj_vertices(STR3D)
        center, large_tip, small_tip = mesh_endpoints_from_vertices(vertices)

        expected_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        np.testing.assert_allclose(center, expected_center, atol=1e-9)
        np.testing.assert_allclose(center, [0.047503, 0.047467, -0.075], atol=1e-6)

        # 大端在 z 最小的一侧（半径更大），小端在 z 最大的一侧。
        np.testing.assert_allclose(large_tip, [center[0], center[1], -0.15], atol=1e-6)
        np.testing.assert_allclose(small_tip, [center[0], center[1], 0.0], atol=1e-6)

        centered = load_obj_vertices(STR3D_BBOX_CENTER)
        center_c, large_c, small_c = mesh_endpoints_from_vertices(centered)
        np.testing.assert_allclose(center_c, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(large_c, [0.0, 0.0, -0.075], atol=1e-6)
        np.testing.assert_allclose(small_c, [0.0, 0.0, 0.075], atol=1e-6)

        inverted = inverted_mesh_vertices()
        center_i, large_i, small_i = mesh_endpoints_from_vertices(inverted)
        np.testing.assert_allclose(center_i, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(large_i, [0.0, 0.0, 0.1], atol=1e-12)
        np.testing.assert_allclose(small_i, [0.0, 0.0, -0.1], atol=1e-12)

        # 把 Z 轴翻转后，大端应跟着翻到新的 +z 侧。
        flipped_vertices = inverted * np.array([1.0, 1.0, -1.0])
        _, large_f, small_f = mesh_endpoints_from_vertices(flipped_vertices)
        np.testing.assert_allclose(large_f, [0.0, 0.0, -0.1], atol=1e-12)
        np.testing.assert_allclose(small_f, [0.0, 0.0, 0.1], atol=1e-12)

    def test_mesh_endpoints_rejects_invalid_shapes(self):
        bad_inputs = {
            "1d": np.zeros(3),
            "two_columns": np.zeros((4, 2)),
            "four_columns": np.zeros((4, 4)),
            "empty": np.zeros((0, 3)),
            "flat_z": np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]]),
            "nan": np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]]),
            "inf": np.array([[0.0, 0.0, 0.0], [0.0, np.inf, 1.0]]),
        }
        for name, bad in bad_inputs.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    mesh_endpoints_from_vertices(bad)


class FlipPoseAroundCenterTest(unittest.TestCase):
    def test_flip_pose_around_center_invariants(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(100 + seed)
                R = random_rotation(seed)
                t = rng.uniform(-0.3, 0.3, size=3) + [0.0, 0.0, 0.5]
                pose = make_pose(R, t)
                center = rng.uniform(-0.1, 0.1, size=3)

                out = flip_pose_around_center(pose, center)
                R_out = out[:3, :3]

                # 模型中心在相机系中的位置保持不变。
                center_h = np.append(center, 1.0)
                np.testing.assert_allclose(out @ center_h, pose @ center_h, atol=1e-12)

                # 局部 X 轴不变，局部 Y、Z 轴取反。
                np.testing.assert_allclose(R_out[:, 0], R[:, 0], atol=1e-12)
                np.testing.assert_allclose(R_out[:, 1], -R[:, 1], atol=1e-12)
                np.testing.assert_allclose(R_out[:, 2], -R[:, 2], atol=1e-12)

                # 输出仍是合法旋转矩阵。
                np.testing.assert_allclose(R_out.T @ R_out, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(R_out), 1.0, places=12)
                np.testing.assert_allclose(out[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

                # 连续翻两次应回到原位姿。
                np.testing.assert_allclose(flip_pose_around_center(out, center), pose, atol=1e-12)


class CountPointsInSphereTest(unittest.TestCase):
    def assert_matches_bruteforce(self, depth, K, center_cam, radius_m):
        got = count_points_in_sphere(depth, K, np.asarray(center_cam, dtype=np.float64), radius_m)
        expected = bruteforce_count(depth, K, center_cam, radius_m)
        self.assertEqual(got, expected)
        return expected

    def test_count_points_in_sphere_matches_bruteforce_and_edge_cases(self):
        K = make_intrinsics()
        H, W = 120, 160
        total = 0
        rng = np.random.default_rng(7)

        def random_depth(lo=0.2, hi=1.0):
            depth = rng.uniform(lo, hi, size=(H, W))
            depth[rng.random((H, W)) < 0.10] = 0.0
            depth[rng.random((H, W)) < 0.05] = np.nan
            depth[rng.random((H, W)) < 0.03] = np.inf
            depth[rng.random((H, W)) < 0.03] = -np.inf
            depth[rng.random((H, W)) < 0.03] = -0.4
            return depth

        # 1. 图像中央的常规球体，深度图含 0、NaN、±inf 和负值。
        for seed in range(4):
            with self.subTest(case="interior", seed=seed):
                depth = random_depth()
                center = [0.0, 0.0, 0.5 + 0.05 * seed]
                total += self.assert_matches_bruteforce(depth, K, center, 0.14)

        # 2. 球体贴近图像边缘或角落，以及广角大偏角情形（校验切线包络不漏点）。
        edge_cases = [
            (K, [-0.24, -0.18, 0.5], 0.14),   # 左上角附近
            (K, [0.36, 0.24, 0.5], 0.14),     # 右下角附近
            (K, [-0.30, 0.0, 0.5], 0.14),     # 左边缘
            (K, [0.0, 0.25, 0.6], 0.12),      # 下边缘
            (make_intrinsics(fx=300.0, fy=300.0), [0.80, 0.0, 1.0], 0.14),  # 广角大偏角
        ]
        for i, (k_mat, center, radius) in enumerate(edge_cases):
            with self.subTest(case="image_edge", index=i):
                depth = random_depth()
                total += self.assert_matches_bruteforce(depth, k_mat, center, radius)

        # 3. 球体穿过或贴近相机光心平面（cz <= radius_m），走整幅图分支。
        for i, (cz, radius) in enumerate([(0.1, 0.14), (0.14, 0.14), (0.05, 0.14), (-0.05, 0.14)]):
            with self.subTest(case="crosses_camera_plane", index=i):
                depth = random_depth(lo=0.01, hi=0.5)
                total += self.assert_matches_bruteforce(depth, K, [0.0, 0.0, cz], radius)

        # 4. 球体完全在相机后方（cz <= -radius_m），必须返回 0。
        for i, cz in enumerate([-0.14, -0.2, -1.0]):
            with self.subTest(case="behind_camera", index=i):
                depth = random_depth(lo=0.01, hi=1.0)
                self.assertEqual(count_points_in_sphere(depth, K, np.array([0.0, 0.0, cz]), 0.14), 0)
                self.assertEqual(bruteforce_count(depth, K, [0.0, 0.0, cz], 0.14), 0)

        # 5. 球体在图像外侧，投影后无像素。
        with self.subTest(case="outside_image"):
            depth = random_depth()
            self.assertEqual(count_points_in_sphere(depth, K, np.array([5.0, 0.0, 0.5]), 0.14), 0)
            self.assertEqual(bruteforce_count(depth, K, [5.0, 0.0, 0.5], 0.14), 0)

        self.assertGreater(total, 0, "synthetic depth maps should contain points inside spheres")

    def test_count_points_in_sphere_rejects_invalid_inputs(self):
        K = make_intrinsics()
        depth = np.ones((4, 4))
        center = np.array([0.0, 0.0, 0.5])
        for bad_radius in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(radius=bad_radius):
                with self.assertRaises(ValueError):
                    count_points_in_sphere(depth, K, center, bad_radius)
        with self.assertRaises(ValueError):
            count_points_in_sphere(np.ones((2, 2, 2)), K, center, 0.1)
        with self.assertRaises(ValueError):
            count_points_in_sphere(depth, np.eye(4), center, 0.1)


class DisambiguateZonePoseTest(unittest.TestCase):
    def setUp(self):
        vertices = load_obj_vertices(STR3D)
        self.model_center, self.large_tip, self.small_tip = mesh_endpoints_from_vertices(vertices)
        self.K = make_intrinsics()
        self.H, self.W = 480, 640

        R_true = random_rotation(seed=3)
        center_cam = np.array([0.02, -0.01, 0.55])
        t_true = center_cam - R_true @ self.model_center
        self.R_true = R_true
        self.pose_true = make_pose(R_true, t_true)
        self.large_cam = R_true @ self.large_tip + t_true
        self.small_cam = R_true @ self.small_tip + t_true

    def make_depth(self, clusters):
        """Background at 1.0 m plus dense point clusters given as (camera point, count) pairs."""
        depth = np.full((self.H, self.W), 1.0, dtype=np.float64)
        rng = np.random.default_rng(11)
        for point, count in clusters:
            offsets = rng.normal(scale=0.02, size=(count, 3))
            pts = np.asarray(point) + offsets
            for x, y, z in pts:
                u = int(round(self.K[0, 0] * x / z + self.K[0, 2]))
                v = int(round(self.K[1, 1] * y / z + self.K[1, 2]))
                depth[v, u] = z
        return depth

    def test_disambiguate_zone_pose_flips_only_when_n_large_gt_n_small(self):
        depth = self.make_depth([(self.small_cam, 300), (self.large_cam, 0)])

        # 正确位姿：小端周围点多，不翻转。
        correct = disambiguate_zone_pose(
            self.pose_true, depth, self.K, self.model_center, self.large_tip, self.small_tip
        )
        self.assertFalse(correct.flipped)
        self.assertGreater(correct.n_small, correct.n_large)
        np.testing.assert_allclose(correct.pose, self.pose_true, atol=1e-12)

        # 绕模型中心翻转 180 度后的位姿：大端会落到小端的真实位置，应被翻回来。
        flipped_input = flip_pose_around_center(self.pose_true, self.model_center)
        self.assertFalse(np.allclose(flipped_input, self.pose_true))
        result = disambiguate_zone_pose(
            flipped_input, depth, self.K, self.model_center, self.large_tip, self.small_tip
        )
        self.assertTrue(result.flipped)
        self.assertGreater(result.n_large, result.n_small)
        np.testing.assert_allclose(result.pose[:3, :3], self.R_true, atol=1e-9)
        np.testing.assert_allclose(
            result.pose[:3, :3] @ self.model_center + result.pose[:3, 3],
            self.pose_true[:3, :3] @ self.model_center + self.pose_true[:3, 3],
            atol=1e-9,
        )
        np.testing.assert_allclose(result.pose, self.pose_true, atol=1e-9)

    def test_disambiguate_zone_pose_ties_do_not_flip(self):
        empty = np.zeros((self.H, self.W), dtype=np.float64)
        result = disambiguate_zone_pose(
            self.pose_true, empty, self.K, self.model_center, self.large_tip, self.small_tip
        )
        self.assertEqual((result.n_large, result.n_small), (0, 0))
        self.assertFalse(result.flipped)
        np.testing.assert_allclose(result.pose, self.pose_true, atol=1e-12)

        # 点云位于两端中点附近（距两端均小于 0.14 m），两个球体包含的点完全相同，计数必然相等。
        midpoint = (self.small_cam + self.large_cam) / 2.0
        rng = np.random.default_rng(5)
        directions = rng.normal(size=(150, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = 0.04 * rng.uniform(size=(150, 1)) ** (1.0 / 3.0)
        symmetric = np.full((self.H, self.W), 1.0, dtype=np.float64)
        for x, y, z in midpoint + directions * radii:
            u = int(round(self.K[0, 0] * x / z + self.K[0, 2]))
            v = int(round(self.K[1, 1] * y / z + self.K[1, 2]))
            symmetric[v, u] = z
        tie = disambiguate_zone_pose(
            self.pose_true, symmetric, self.K, self.model_center, self.large_tip, self.small_tip
        )
        self.assertEqual(tie.n_large, tie.n_small)
        self.assertGreater(tie.n_small, 0)
        self.assertFalse(tie.flipped)
        np.testing.assert_allclose(tie.pose, self.pose_true, atol=1e-12)

    def test_disambiguate_zone_pose_rejects_invalid_inputs(self):
        depth = self.make_depth([(self.small_cam, 50)])
        args = (self.K, self.model_center, self.large_tip, self.small_tip)

        for bad_radius in (0.0, -0.14, float("nan"), float("inf")):
            with self.subTest(radius=bad_radius):
                with self.assertRaises(ValueError):
                    disambiguate_zone_pose(self.pose_true, depth, *args, radius_m=bad_radius)

        bad_poses = {
            "3x4": self.pose_true[:3, :],
            "5x5": np.eye(5),
            "nan": np.where(np.eye(4) > 0, np.nan, 0.0),
        }
        for name, bad in bad_poses.items():
            with self.subTest(pose=name):
                with self.assertRaises(ValueError):
                    disambiguate_zone_pose(bad, depth, *args)

        bad_ks = {"4x4": np.eye(4), "3x3_nan": np.full((3, 3), np.nan)}
        for name, bad in bad_ks.items():
            with self.subTest(K=name):
                with self.assertRaises(ValueError):
                    disambiguate_zone_pose(self.pose_true, depth, bad, self.model_center,
                                           self.large_tip, self.small_tip)


if __name__ == "__main__":
    unittest.main()
