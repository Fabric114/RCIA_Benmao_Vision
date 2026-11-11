import unittest

import numpy as np

from scripts.pose_eval_metrics import pose_errors


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PoseErrorsTest(unittest.TestCase):
    def setUp(self):
        self.center = np.array([0.0475, 0.0475, -0.075])

    def about_center(self, pose, R):
        T = np.eye(4)
        T[:3, 3] = self.center
        Rh = np.eye(4)
        Rh[:3, :3] = R
        return pose @ T @ Rh @ np.linalg.inv(T)

    def test_identical_poses_have_zero_error(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.0, 0.5]
        c, a = pose_errors(pose, pose, self.center)
        self.assertAlmostEqual(c, 0.0, places=9)
        self.assertAlmostEqual(a, 0.0, places=6)

    def test_yaw_about_object_axis_is_ignored(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.0, 0.5]
        c, a = pose_errors(pose, self.about_center(pose, rot_z(np.radians(73))), self.center)
        self.assertLess(c, 1e-9)
        self.assertLess(a, 1e-6)

    def test_axis_flip_is_180_degrees(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.0, 0.5]
        c, a = pose_errors(pose, self.about_center(pose, rot_x(np.pi)), self.center)
        self.assertLess(c, 1e-9)
        self.assertAlmostEqual(a, 180.0, places=4)

    def test_translation_error_in_meters(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.0, 0.5]
        moved = pose.copy()
        moved[:3, 3] += [0.003, 0.004, 0.0]
        c, a = pose_errors(pose, moved, self.center)
        self.assertAlmostEqual(c, 0.005, places=9)
        self.assertAlmostEqual(a, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
