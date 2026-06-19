import math
import unittest

import numpy as np

from scripts.linemod_pose_utils import (
    backproject_pixel,
    look_at_object_in_camera,
    replace_pose_translation_from_depth,
    sample_hemisphere_views,
)


class LinemodPoseUtilsTest(unittest.TestCase):
    def test_sample_hemisphere_views_returns_expected_count_and_radius(self):
        views = sample_hemisphere_views(
            radius=0.5,
            azimuth_count=4,
            elevation_degrees=[0.0, 45.0],
        )

        self.assertEqual(views.shape, (8, 3))
        np.testing.assert_allclose(np.linalg.norm(views, axis=1), 0.5, atol=1e-6)

    def test_look_at_object_pose_maps_object_origin_in_front_of_camera(self):
        pose = look_at_object_in_camera(np.array([0.0, 0.0, -0.5]))

        origin_in_camera = pose @ np.array([0.0, 0.0, 0.0, 1.0])

        np.testing.assert_allclose(origin_in_camera[:3], [0.0, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6)

    def test_backproject_pixel_uses_camera_intrinsics(self):
        k_matrix = np.array(
            [
                [600.0, 0.0, 320.0],
                [0.0, 600.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )

        point = backproject_pixel(380.0, 180.0, 0.5, k_matrix)

        np.testing.assert_allclose(point, [0.05, -0.05, 0.5], atol=1e-6)

    def test_replace_pose_translation_keeps_rotation_and_uses_depth_point(self):
        k_matrix = np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )
        template_pose = np.eye(4)
        angle = math.pi / 6.0
        template_pose[:3, :3] = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        pose = replace_pose_translation_from_depth(
            template_pose,
            center_u=420.0,
            center_v=240.0,
            depth_m=1.0,
            k_matrix=k_matrix,
        )

        np.testing.assert_allclose(pose[:3, :3], template_pose[:3, :3], atol=1e-6)
        np.testing.assert_allclose(pose[:3, 3], [0.2, 0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
