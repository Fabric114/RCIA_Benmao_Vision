import unittest

import numpy as np

from scripts.linemod_silhouette_renderer import render_model_buffers
from scripts.m3t_tracking_health_gate import (
    TrackingHealthGateParams,
    evaluate_m3t_tracking_health,
)


DEFAULT_COLOR = object()


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
        self.mask_depth_m = rendered_depth
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

    def _evaluate_with_depth(self, depth, pose=None, color=DEFAULT_COLOR):
        params = TrackingHealthGateParams(
            min_rendered_overlap_pixels=80,
            min_depth_consistency_ratio=0.90,
            min_projection_iou=0.0,
            max_projection_center_distance_ratio=10.0,
            min_projection_area_ratio=0.0,
            max_depth_residual_m=0.005,
            require_smooth_pose=False,
        )
        return evaluate_m3t_tracking_health(
            vertices=self.vertices,
            faces=self.faces,
            object_in_camera=self.pose if pose is None else pose,
            previous_pose=self.pose,
            k_matrix=self.k_matrix,
            color_bgr=self.color if color is DEFAULT_COLOR else color,
            observed_depth_image=depth,
            yolo_bbox=None,
            params=params,
        )

    def _depth_for_first_mask_pixels(self, count):
        """Observed depth valid only at the first `count` rendered mask pixels."""
        depth = np.zeros((120, 160), dtype=np.uint16)
        ys, xs = np.nonzero(self.mask > 0)
        keep_y, keep_x = ys[:count], xs[:count]
        depth[keep_y, keep_x] = np.round(
            self.mask_depth_m[keep_y, keep_x] * 1000
        ).astype(np.uint16)
        return depth

    def test_zero_depth_is_unknown_not_measured_contradiction(self):
        result = self._evaluate_with_depth(np.zeros((120, 160), dtype=np.uint16))

        self.assertTrue(result.health_evaluated)
        self.assertEqual(result.valid_depth_overlap_count, 0)
        self.assertFalse(result.depth_evaluable)
        self.assertFalse(result.depth_ok)

    def test_valid_background_depth_is_measured_contradiction(self):
        background = np.full((120, 160), 2000, dtype=np.uint16)

        result = self._evaluate_with_depth(background)

        self.assertTrue(result.health_evaluated)
        self.assertGreaterEqual(result.valid_depth_overlap_count, 80)
        self.assertTrue(result.depth_evaluable)
        self.assertEqual(result.depth_ratio, 0.0)
        self.assertFalse(result.depth_ok)

    def test_depth_evaluable_boundary_is_80_valid_overlap_pixels(self):
        below = self._evaluate_with_depth(self._depth_for_first_mask_pixels(79))
        at = self._evaluate_with_depth(self._depth_for_first_mask_pixels(80))

        self.assertEqual(below.valid_depth_overlap_count, 79)
        self.assertFalse(below.depth_evaluable)
        self.assertEqual(at.valid_depth_overlap_count, 80)
        self.assertTrue(at.depth_evaluable)

    def test_out_of_view_pose_is_evaluated_but_not_depth_evaluable(self):
        outside = self.pose.copy()
        outside[:3, 3] = [5.0, 0.0, 0.5]

        result = self._evaluate_with_depth(self.depth, pose=outside)

        self.assertTrue(result.health_evaluated)
        self.assertEqual(result.rendered_pixel_count, 0)
        self.assertEqual(result.valid_depth_overlap_count, 0)
        self.assertFalse(result.depth_evaluable)

    def test_nonfinite_depth_counts_as_invalid_overlap(self):
        depth = np.full((120, 160), np.nan, dtype=np.float32)

        result = self._evaluate_with_depth(depth)

        self.assertTrue(result.health_evaluated)
        self.assertEqual(result.valid_depth_overlap_count, 0)
        self.assertFalse(result.depth_evaluable)

    def test_nonfinite_pose_is_rejected_before_rendering(self):
        from unittest import mock

        nan_pose = self.pose.copy()
        nan_pose[0, 3] = float("nan")

        with mock.patch(
            "scripts.tracking_watchdog_validation.render_model_buffers"
        ) as render:
            result = self._evaluate_with_depth(self.depth, pose=nan_pose)

        render.assert_not_called()
        self.assertFalse(result.health_evaluated)
        self.assertFalse(result.depth_evaluable)

    def test_missing_rgbd_is_not_evaluated(self):
        result = self._evaluate_with_depth(self.depth, color=None)

        self.assertFalse(result.health_evaluated)
        self.assertFalse(result.depth_evaluable)
        self.assertEqual(result.valid_depth_overlap_count, 0)


if __name__ == "__main__":
    unittest.main()
