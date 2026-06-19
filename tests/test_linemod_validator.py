import unittest

import numpy as np

from scripts.linemod_silhouette_renderer import render_model_buffers
from scripts.linemod_validator import (
    validate_black_white_appearance,
    validate_rendered_depth_consistency,
)


class LinemodValidatorTest(unittest.TestCase):
    def test_black_white_appearance_accepts_low_saturation_black_and_white_target(self):
        color = np.full((80, 80, 3), (30, 30, 30), dtype=np.uint8)
        color[:, 40:] = (220, 220, 220)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 10:70] = 255

        result = validate_black_white_appearance(color, mask)

        self.assertTrue(result.success, result.reason)
        self.assertGreaterEqual(result.low_saturation_ratio, 0.65)
        self.assertGreaterEqual(result.black_white_support_ratio, 0.45)

    def test_black_white_appearance_rejects_high_saturation_color_region(self):
        color = np.full((80, 80, 3), (0, 0, 255), dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 10:70] = 255

        result = validate_black_white_appearance(color, mask)

        self.assertFalse(result.success)
        self.assertLess(result.low_saturation_ratio, 0.65)

    def test_correct_synthetic_pose_passes_rendered_depth_consistency(self):
        vertices = np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]
        k_matrix = np.array(
            [
                [600.0, 0.0, 80.0],
                [0.0, 600.0, 60.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        _, mask, _, rendered_depth, _ = render_model_buffers(
            vertices,
            faces,
            pose,
            k_matrix,
            image_size=(160, 120),
        )
        observed_depth = np.zeros((120, 160), dtype=np.uint16)
        observed_depth[mask > 0] = np.round(rendered_depth[mask > 0] * 1000.0).astype(np.uint16)

        result = validate_rendered_depth_consistency(
            vertices,
            faces,
            pose,
            k_matrix,
            observed_depth,
            image_size=(160, 120),
            depth_scale=0.001,
            max_depth_residual_m=0.005,
            min_consistency_ratio=0.95,
        )

        self.assertTrue(result.success, result.reason)
        self.assertGreaterEqual(result.consistency_ratio, 0.95)

    def test_shifted_pose_fails_rendered_depth_consistency(self):
        vertices = np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        observed_pose = np.eye(4, dtype=np.float64)
        observed_pose[:3, 3] = [0.0, 0.0, 0.5]
        candidate_pose = observed_pose.copy()
        candidate_pose[:3, 3] = [0.0, 0.0, 0.58]
        k_matrix = np.array(
            [
                [600.0, 0.0, 80.0],
                [0.0, 600.0, 60.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        _, mask, _, observed_rendered_depth, _ = render_model_buffers(
            vertices,
            faces,
            observed_pose,
            k_matrix,
            image_size=(160, 120),
        )
        observed_depth = np.zeros((120, 160), dtype=np.uint16)
        observed_depth[mask > 0] = np.round(observed_rendered_depth[mask > 0] * 1000.0).astype(np.uint16)

        result = validate_rendered_depth_consistency(
            vertices,
            faces,
            candidate_pose,
            k_matrix,
            observed_depth,
            image_size=(160, 120),
            depth_scale=0.001,
            max_depth_residual_m=0.02,
            min_consistency_ratio=0.70,
        )

        self.assertFalse(result.success)
        self.assertLess(result.consistency_ratio, 0.70)


if __name__ == "__main__":
    unittest.main()
