import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.linemod_silhouette_renderer import render_mask_and_depth, render_model_buffers
from scripts.linemod_validator import (
    validate_black_white_appearance,
    validate_rendered_depth_consistency,
)
from scripts.tracking_watchdog_validation import (
    WatchdogHealth,
    WatchdogValidationParams,
    bbox_iou,
    build_watchdog_health,
    bbox_area_ratio,
    bbox_center_distance_ratio,
    mask_bbox,
    pose_jump_ok,
    projection_iou_from_mask,
    validate_candidate_pose,
    validate_m3t_pose_health,
)


class TrackingWatchdogValidationTest(unittest.TestCase):
    def test_bbox_iou_calculates_overlap_ratio(self):
        self.assertAlmostEqual(
            bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25.0 / 175.0,
        )
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(bbox_iou(None, (20, 20, 30, 30)), 0.0)

    def test_projection_iou_uses_projected_mask_bbox(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255

        self.assertEqual(mask_bbox(mask), (10.0, 10.0, 29.0, 29.0))
        self.assertGreater(projection_iou_from_mask(mask, (12, 12, 28, 28)), 0.65)
        self.assertEqual(projection_iou_from_mask(mask, (0, 0, 5, 5)), 0.0)

    def test_bbox_center_distance_ratio_normalizes_by_yolo_size(self):
        self.assertAlmostEqual(
            bbox_center_distance_ratio((10, 10, 30, 30), (10, 10, 30, 30)),
            0.0,
        )
        self.assertGreater(
            bbox_center_distance_ratio((10, 10, 30, 30), (50, 10, 70, 30)),
            1.0,
        )
        self.assertEqual(bbox_center_distance_ratio(None, (10, 10, 30, 30)), float("inf"))

    def test_bbox_area_ratio_is_symmetric(self):
        self.assertAlmostEqual(bbox_area_ratio((0, 0, 20, 20), (0, 0, 10, 10)), 0.25)
        self.assertAlmostEqual(bbox_area_ratio((0, 0, 10, 10), (0, 0, 20, 20)), 0.25)
        self.assertEqual(bbox_area_ratio(None, (0, 0, 20, 20)), 0.0)

    def test_pose_jump_ok_rejects_translation_and_rotation_jumps(self):
        previous = np.eye(4, dtype=np.float64)
        small_move = np.eye(4, dtype=np.float64)
        small_move[:3, 3] = [0.01, 0.0, 0.0]
        large_move = np.eye(4, dtype=np.float64)
        large_move[:3, 3] = [0.20, 0.0, 0.0]

        rotate_90 = np.eye(4, dtype=np.float64)
        angle = math.radians(90)
        rotate_90[:3, :3] = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        self.assertTrue(pose_jump_ok(previous, small_move, max_translation_m=0.05))
        self.assertFalse(pose_jump_ok(previous, large_move, max_translation_m=0.05))
        self.assertFalse(pose_jump_ok(previous, rotate_90, max_rotation_deg=30.0))
        self.assertTrue(pose_jump_ok(None, rotate_90))

    def test_build_watchdog_health_combines_depth_appearance_and_iou(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        depth_result = SimpleNamespace(
            success=True,
            consistency_ratio=0.78,
            rendered_pixel_count=1600,
            valid_overlap_count=1500,
            supported_pixel_count=1170,
            mean_abs_depth_residual_m=0.008,
            reason="",
        )
        appearance_result = SimpleNamespace(
            success=True,
            sampled_pixel_count=1600,
            low_saturation_ratio=0.95,
            black_white_support_ratio=0.80,
            reason="",
        )

        health = build_watchdog_health(
            yolo_bbox=(18, 18, 62, 62),
            rendered_mask=mask,
            depth_result=depth_result,
            appearance_result=appearance_result,
            previous_pose=np.eye(4),
            current_pose=np.eye(4),
            params=WatchdogValidationParams(min_rendered_overlap_pixels=80),
        )

        self.assertTrue(health.yolo_visible)
        self.assertTrue(health.pose_valid)
        self.assertTrue(health.black_white_success)
        self.assertAlmostEqual(health.depth_consistency_ratio, 0.78)
        self.assertGreater(health.projection_iou, 0.7)
        self.assertTrue(health.projection_center_success)
        self.assertTrue(health.projection_area_success)

    def test_build_watchdog_health_marks_small_projected_mask_inconclusive(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        depth_result = SimpleNamespace(
            success=False,
            consistency_ratio=0.0,
            rendered_pixel_count=9,
            valid_overlap_count=9,
            supported_pixel_count=0,
            mean_abs_depth_residual_m=None,
            reason="too few pixels",
        )
        appearance_result = SimpleNamespace(
            success=True,
            sampled_pixel_count=9,
            low_saturation_ratio=1.0,
            black_white_support_ratio=1.0,
            reason="",
        )

        health = build_watchdog_health(
            yolo_bbox=(0, 0, 8, 8),
            rendered_mask=mask,
            depth_result=depth_result,
            appearance_result=appearance_result,
            previous_pose=None,
            current_pose=np.eye(4),
            params=WatchdogValidationParams(min_rendered_overlap_pixels=80),
        )

        self.assertFalse(health.pose_valid)
        self.assertFalse(health.black_white_success)
        self.assertIn("too few rendered pixels", health.reason)

    def test_build_watchdog_health_rejects_projection_center_mismatch(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        depth_result = SimpleNamespace(
            success=True,
            consistency_ratio=0.95,
            rendered_pixel_count=1600,
            valid_overlap_count=1600,
            supported_pixel_count=1520,
            mean_abs_depth_residual_m=0.002,
            reason="",
        )
        appearance_result = SimpleNamespace(
            success=True,
            sampled_pixel_count=1600,
            low_saturation_ratio=0.95,
            black_white_support_ratio=0.80,
            reason="",
        )

        health = build_watchdog_health(
            yolo_bbox=(70, 30, 99, 70),
            rendered_mask=mask,
            depth_result=depth_result,
            appearance_result=appearance_result,
            previous_pose=np.eye(4),
            current_pose=np.eye(4),
            params=WatchdogValidationParams(
                min_rendered_overlap_pixels=80,
                min_yolo_projection_iou=0.0,
                max_projection_center_distance_ratio=0.6,
            ),
        )

        self.assertFalse(health.pose_valid)
        self.assertFalse(health.projection_center_success)
        self.assertIn("center", health.reason.lower())

    def test_build_watchdog_health_rejects_projection_area_mismatch(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[10:110, 10:110] = 255
        depth_result = SimpleNamespace(
            success=True,
            consistency_ratio=0.95,
            rendered_pixel_count=10000,
            valid_overlap_count=10000,
            supported_pixel_count=9500,
            mean_abs_depth_residual_m=0.002,
            reason="",
        )
        appearance_result = SimpleNamespace(
            success=True,
            sampled_pixel_count=10000,
            low_saturation_ratio=0.95,
            black_white_support_ratio=0.80,
            reason="",
        )

        health = build_watchdog_health(
            yolo_bbox=(40, 40, 80, 80),
            rendered_mask=mask,
            depth_result=depth_result,
            appearance_result=appearance_result,
            previous_pose=np.eye(4),
            current_pose=np.eye(4),
            params=WatchdogValidationParams(
                min_rendered_overlap_pixels=80,
                min_yolo_projection_iou=0.0,
                min_projection_area_ratio=0.30,
            ),
        )

        self.assertFalse(health.pose_valid)
        self.assertFalse(health.projection_area_success)
        self.assertIn("area", health.reason.lower())

    def test_build_watchdog_health_carries_raw_valid_depth_overlap(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        depth_result = SimpleNamespace(
            success=False,
            consistency_ratio=0.0,
            rendered_pixel_count=1600,
            valid_overlap_count=37,
            supported_pixel_count=0,
            mean_abs_depth_residual_m=None,
            reason="Need at least 80 rendered/observed depth overlap pixels, got 37",
        )
        appearance_result = SimpleNamespace(
            success=True,
            sampled_pixel_count=1600,
            low_saturation_ratio=0.95,
            black_white_support_ratio=0.80,
            reason="",
        )

        health = build_watchdog_health(
            yolo_bbox=None,
            rendered_mask=mask,
            depth_result=depth_result,
            appearance_result=appearance_result,
            previous_pose=None,
            current_pose=np.eye(4),
            params=WatchdogValidationParams(min_rendered_overlap_pixels=80),
        )

        self.assertEqual(health.valid_depth_overlap_count, 37)
        self.assertEqual(WatchdogHealth(
            yolo_visible=False,
            depth_success=False,
            depth_consistency_ratio=0.0,
            black_white_success=False,
            projection_iou=0.0,
            projection_center_distance_ratio=0.0,
            projection_area_ratio=0.0,
            projection_center_success=True,
            projection_area_success=True,
            pose_jump_ok=True,
            pose_valid=False,
            rendered_pixel_count=0,
        ).valid_depth_overlap_count, 0)


class RenderOnceWatchdogValidationTest(unittest.TestCase):
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
            self.vertices, self.faces, self.pose, self.k_matrix, image_size=(160, 120)
        )
        self.depth = np.zeros((120, 160), dtype=np.uint16)
        self.depth[self.mask > 0] = np.round(rendered_depth[self.mask > 0] * 1000).astype(np.uint16)
        self.color = np.full((120, 160, 3), (30, 30, 30), dtype=np.uint8)
        self.color[:, 80:] = (220, 220, 220)
        self.params = WatchdogValidationParams(
            min_rendered_overlap_pixels=50,
            tracking_min_depth_consistency_ratio=0.90,
            max_depth_residual_m=0.005,
            min_yolo_projection_iou=0.0,
        )

    def test_m3t_pose_health_renders_once_and_matches_reference_logic(self):
        yolo_bbox = (20, 0, 140, 119)
        previous_pose = self.pose.copy()

        with mock.patch(
            "scripts.tracking_watchdog_validation.render_mask_and_depth",
            wraps=render_mask_and_depth,
        ) as render_once, mock.patch(
            "scripts.tracking_watchdog_validation.render_model_buffers"
        ) as legacy_render:
            actual = validate_m3t_pose_health(
                self.vertices,
                self.faces,
                self.pose,
                self.k_matrix,
                self.color,
                self.depth,
                yolo_bbox=yolo_bbox,
                previous_pose=previous_pose,
                params=self.params,
            )

        render_once.assert_called_once()
        legacy_render.assert_not_called()

        _, reference_mask, _, _, _ = render_model_buffers(
            self.vertices, self.faces, self.pose, self.k_matrix, image_size=(160, 120)
        )
        reference_depth_result = validate_rendered_depth_consistency(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            self.depth,
            image_size=(160, 120),
            depth_scale=self.params.depth_scale,
            max_depth_residual_m=self.params.max_depth_residual_m,
            min_consistency_ratio=self.params.tracking_min_depth_consistency_ratio,
            min_overlap_pixels=self.params.min_rendered_overlap_pixels,
        )
        expected = build_watchdog_health(
            yolo_bbox=yolo_bbox,
            rendered_mask=reference_mask,
            depth_result=reference_depth_result,
            appearance_result=validate_black_white_appearance(self.color, reference_mask),
            previous_pose=previous_pose,
            current_pose=self.pose,
            params=self.params,
        )

        self.assertEqual(actual, expected)
        self.assertTrue(actual.depth_success)
        self.assertTrue(actual.black_white_success)

    def test_candidate_pose_renders_once_and_matches_reference_logic(self):
        yolo_bbox = (20, 0, 140, 119)

        with mock.patch(
            "scripts.tracking_watchdog_validation.render_mask_and_depth",
            wraps=render_mask_and_depth,
        ) as render_once, mock.patch(
            "scripts.tracking_watchdog_validation.render_model_buffers"
        ) as legacy_render:
            actual = validate_candidate_pose(
                self.vertices,
                self.faces,
                self.pose,
                self.k_matrix,
                self.color,
                self.depth,
                yolo_bbox,
                params=self.params,
            )

        render_once.assert_called_once()
        legacy_render.assert_not_called()

        _, reference_mask, _, _, _ = render_model_buffers(
            self.vertices, self.faces, self.pose, self.k_matrix, image_size=(160, 120)
        )
        reference_depth_result = validate_rendered_depth_consistency(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            self.depth,
            image_size=(160, 120),
            depth_scale=self.params.depth_scale,
            max_depth_residual_m=self.params.max_depth_residual_m,
            min_consistency_ratio=self.params.tracking_min_depth_consistency_ratio,
            min_overlap_pixels=self.params.min_rendered_overlap_pixels,
        )
        reference_appearance = validate_black_white_appearance(self.color, reference_mask)
        self.assertTrue(actual.success, actual.reason)
        self.assertEqual(actual.depth_consistency_ratio, reference_depth_result.consistency_ratio)
        self.assertEqual(actual.rendered_pixel_count, reference_depth_result.rendered_pixel_count)
        self.assertEqual(actual.black_white_success, reference_appearance.success)
        self.assertEqual(
            actual.projection_iou,
            projection_iou_from_mask(reference_mask, yolo_bbox),
        )


if __name__ == "__main__":
    unittest.main()
