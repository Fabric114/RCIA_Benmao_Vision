import math
import unittest

import numpy as np

from scripts.linemod_line2d import Line2DFeature, Line2DMatch, Line2DTemplate
from scripts.linemod_pnp_pose import estimate_pose_from_match
from scripts.linemod_pose_utils import (
    add_error,
    mean_reprojection_error,
    object_diameter,
    project_model_points,
)


def rotation_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class LinemodPnpPoseTest(unittest.TestCase):
    def test_estimate_pose_from_match_recovers_known_object_in_camera_pose(self):
        object_points = np.array(
            [
                [-0.04, -0.03, -0.02],
                [0.04, -0.03, -0.02],
                [0.04, 0.03, -0.02],
                [-0.04, 0.03, -0.02],
                [-0.04, -0.03, 0.02],
                [0.04, -0.03, 0.02],
                [0.04, 0.03, 0.02],
                [-0.04, 0.03, 0.02],
            ],
            dtype=np.float64,
        )
        k_matrix = np.array(
            [
                [620.0, 0.0, 160.0],
                [0.0, 620.0, 120.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_pose = np.eye(4, dtype=np.float64)
        expected_pose[:3, :3] = rotation_y(math.radians(12.0))
        expected_pose[:3, 3] = [0.015, -0.01, 0.55]
        image_points = project_model_points(object_points, expected_pose, k_matrix)
        match = Line2DMatch("cube", 3, x=9, y=7, similarity=99.0)
        features = tuple(
            Line2DFeature(
                x=float(point_2d[0] - match.x),
                y=float(point_2d[1] - match.y),
                orientation=0,
                model_point=tuple(point_3d.tolist()),
            )
            for point_2d, point_3d in zip(image_points, object_points)
        )
        template = Line2DTemplate("cube", 3, width=80, height=70, bbox=(0, 0, 80, 70), features=features)

        result = estimate_pose_from_match(match, template, k_matrix)

        self.assertTrue(result.success, result.reason)
        self.assertGreaterEqual(result.inlier_count, 6)
        self.assertGreater(result.object_in_camera[2, 3], 0.0)
        self.assertLess(result.mean_reprojection_error_px, 2.0)
        self.assertLess(
            mean_reprojection_error(object_points, image_points, result.object_in_camera, k_matrix),
            2.0,
        )
        self.assertLess(
            add_error(object_points, expected_pose, result.object_in_camera),
            0.1 * object_diameter(object_points),
        )

    def test_estimate_pose_from_match_rejects_insufficient_correspondences(self):
        template = Line2DTemplate(
            "obj",
            1,
            width=10,
            height=10,
            bbox=(0, 0, 10, 10),
            features=(
                Line2DFeature(1, 1, 0, model_point=(0.0, 0.0, 0.0)),
                Line2DFeature(2, 2, 0, model_point=(0.01, 0.0, 0.0)),
            ),
        )
        match = Line2DMatch("obj", 1, x=0, y=0, similarity=100.0)

        result = estimate_pose_from_match(match, template, np.eye(3))

        self.assertFalse(result.success)
        self.assertIsNone(result.object_in_camera)
        self.assertIn("correspondences", result.reason)


if __name__ == "__main__":
    unittest.main()
