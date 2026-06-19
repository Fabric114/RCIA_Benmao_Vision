import unittest

import numpy as np

from scripts.linemod_silhouette_renderer import render_model_buffers
from scripts.m3t_tracking_health_gate import (
    TrackingHealthGateParams,
    evaluate_m3t_tracking_health,
)


class M3TTrackingHealthGateTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = [0.0, 0.0, 0.5]
        self.k_matrix = np.array(
            [[600.0, 0.0, 80.0], [0.0, 600.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        _, self.mask, _, rendered_depth, _ = render_model_buffers(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            image_size=(160, 120),
        )
        self.depth = np.zeros((120, 160), dtype=np.uint16)
        self.depth[self.mask > 0] = np.round(rendered_depth[self.mask > 0] * 1000).astype(
            np.uint16
        )
        self.color = np.full((120, 160, 3), (30, 30, 30), dtype=np.uint8)
        self.color[:, 80:] = (220, 220, 220)
        self.params = TrackingHealthGateParams(
            min_rendered_overlap_pixels=50,
            min_depth_consistency_ratio=0.90,
            min_projection_iou=0.20,
            max_projection_center_distance_ratio=0.60,
            min_projection_area_ratio=0.30,
            max_depth_residual_m=0.005,
            max_pose_jump_translation_m=0.05,
            max_pose_jump_rotation_deg=20.0,
            require_smooth_pose=True,
        )

    def test_healthy_pose_is_tracking_ok(self):
        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=self.pose,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color,
            observed_depth_image=self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertTrue(result.tracking_ok, result.reason)
        self.assertTrue(result.depth_ok)
        self.assertTrue(result.projection_ok)
        self.assertTrue(result.projection_center_ok)
        self.assertTrue(result.projection_area_ok)
        self.assertTrue(result.smoothness_ok)

    def test_projection_mismatch_is_not_tracking_ok(self):
        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=self.pose,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color,
            observed_depth_image=self.depth,
            yolo_bbox=(0, 0, 10, 10),
            params=self.params,
        )

        self.assertFalse(result.tracking_ok)
        self.assertFalse(result.projection_ok)
        self.assertIn("projection", result.reason.lower())

    def test_projection_center_mismatch_is_not_tracking_ok(self):
        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=self.pose,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color,
            observed_depth_image=self.depth,
            yolo_bbox=(120, 0, 159, 119),
            params=TrackingHealthGateParams(
                min_rendered_overlap_pixels=50,
                min_depth_consistency_ratio=0.90,
                min_projection_iou=0.0,
                max_projection_center_distance_ratio=0.40,
                min_projection_area_ratio=0.0,
                max_depth_residual_m=0.005,
            ),
        )

        self.assertFalse(result.tracking_ok)
        self.assertFalse(result.projection_center_ok)
        self.assertIn("center", result.reason.lower())

    def test_depth_mismatch_is_not_tracking_ok(self):
        shifted = self.pose.copy()
        shifted[:3, 3] = [0.0, 0.0, 0.58]

        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=shifted,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color,
            observed_depth_image=self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertFalse(result.tracking_ok)
        self.assertFalse(result.depth_ok)
        self.assertIn("depth", result.reason.lower())

    def test_black_white_mismatch_is_not_tracking_ok_after_start(self):
        colorful = np.zeros_like(self.color)
        colorful[:, :] = (0, 0, 255)

        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=self.pose,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=colorful,
            observed_depth_image=self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertFalse(result.tracking_ok)
        self.assertFalse(result.black_white_ok)
        self.assertIn("black", result.reason.lower())

    def test_pose_jump_is_reported_as_smoothness_failure(self):
        jumped = self.pose.copy()
        jumped[:3, 3] = [0.20, 0.0, 0.5]

        result = evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=jumped,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color,
            observed_depth_image=self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertFalse(result.smoothness_ok)
        self.assertGreater(result.translation_jump_m, 0.05)
        self.assertFalse(result.tracking_ok)


if __name__ == "__main__":
    unittest.main()
