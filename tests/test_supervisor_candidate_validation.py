import unittest

import numpy as np

from scripts.linemod_silhouette_renderer import render_model_buffers
from scripts.tracking_watchdog_validation import (
    WatchdogValidationParams,
    validate_candidate_pose,
)


class SupervisorCandidateValidationTest(unittest.TestCase):
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
        self.params = WatchdogValidationParams(
            min_rendered_overlap_pixels=50,
            tracking_min_depth_consistency_ratio=0.90,
            min_yolo_projection_iou=0.20,
            max_depth_residual_m=0.005,
        )

    def test_valid_synthetic_candidate_passes(self):
        result = validate_candidate_pose(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            self.color,
            self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertTrue(result.success, result.reason)
        self.assertGreaterEqual(result.depth_consistency_ratio, 0.90)
        self.assertGreater(result.projection_iou, 0.20)

    def test_shifted_depth_candidate_fails(self):
        shifted = self.pose.copy()
        shifted[:3, 3] = [0.0, 0.0, 0.58]

        result = validate_candidate_pose(
            self.vertices,
            self.faces,
            shifted,
            self.k_matrix,
            self.color,
            self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertFalse(result.success)
        self.assertIn("depth", result.reason.lower())

    def test_yolo_bbox_mismatch_rejects_candidate(self):
        result = validate_candidate_pose(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            self.color,
            self.depth,
            yolo_bbox=(0, 0, 10, 10),
            params=self.params,
        )

        self.assertFalse(result.success)
        self.assertIn("IoU", result.reason)

    def test_missing_rgbd_rejects_candidate(self):
        result = validate_candidate_pose(
            self.vertices,
            self.faces,
            self.pose,
            self.k_matrix,
            None,
            self.depth,
            yolo_bbox=(20, 0, 140, 119),
            params=self.params,
        )

        self.assertFalse(result.success)
        self.assertIn("missing", result.reason)


if __name__ == "__main__":
    unittest.main()
