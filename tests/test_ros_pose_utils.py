import math
import unittest

from scripts.ros_pose_utils import (
    matrix_to_pose_components,
    normalize_quaternion,
    validate_pose_matrix,
)


class RosPoseUtilsTest(unittest.TestCase):
    def test_validate_pose_matrix_accepts_4x4_homogeneous_matrix(self):
        matrix = [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]

        validated = validate_pose_matrix(matrix)

        self.assertEqual(validated[0][3], 0.1)
        self.assertEqual(validated[1][3], -0.2)
        self.assertEqual(validated[2][3], 0.3)

    def test_matrix_to_pose_components_returns_translation_and_identity_quaternion(self):
        matrix = [
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ]

        pose = matrix_to_pose_components(matrix)

        self.assertEqual(pose["position"], {"x": 0.25, "y": -0.5, "z": 0.75})
        self.assertEqual(pose["orientation"], {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})

    def test_matrix_to_pose_components_handles_z_axis_rotation(self):
        c = math.cos(math.pi / 4.0)
        s = math.sin(math.pi / 4.0)
        matrix = [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        pose = matrix_to_pose_components(matrix)

        self.assertAlmostEqual(pose["orientation"]["x"], 0.0, places=6)
        self.assertAlmostEqual(pose["orientation"]["y"], 0.0, places=6)
        self.assertAlmostEqual(pose["orientation"]["z"], s, places=6)
        self.assertAlmostEqual(pose["orientation"]["w"], c, places=6)

    def test_normalize_quaternion_rejects_zero_norm(self):
        with self.assertRaises(ValueError):
            normalize_quaternion(0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
