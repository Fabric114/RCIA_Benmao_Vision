import math
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.tracking_watchdog_validation import (
    WatchdogValidationParams,
    bbox_iou,
    build_watchdog_health,
    bbox_area_ratio,
    bbox_center_distance_ratio,
    mask_bbox,
    pose_jump_ok,
    projection_iou_from_mask,
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


if __name__ == "__main__":
    unittest.main()
