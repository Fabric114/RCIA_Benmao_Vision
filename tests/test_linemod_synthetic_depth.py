#!/usr/bin/env python3
"""
Synthetic test for rendered depth consistency validation.

This test verifies that when we:
1. Render a mesh at a given pose to generate synthetic depth
2. Validate the same pose against that synthetic depth
The consistency ratio should be near 1.0.

If this test fails, it indicates a bug in the renderer or validator itself,
not in the real-world data.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_silhouette_renderer import load_obj_mesh, render_model_buffers
from scripts.linemod_validator import validate_rendered_depth_consistency


class TestSyntheticDepthConsistency(unittest.TestCase):
    def setUp(self):
        """Load the Str3D mesh and set up camera parameters."""
        mesh_path = PROJECT_ROOT / "models/object/Str3D.obj"
        if not mesh_path.exists():
            self.skipTest(f"Mesh file not found: {mesh_path}")
        self.vertices, self.faces = load_obj_mesh(mesh_path)

        # Standard camera intrinsics (640x480)
        self.k_matrix = np.array(
            [
                [608.30, 0.0, 320.28],
                [0.0, 608.28, 240.57],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.image_size = (640, 480)

    def test_identity_pose_consistency(self):
        """Test with identity pose (object at origin, 0.5m away)."""
        # Object directly in front of camera at 0.5m
        object_in_camera = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Render synthetic depth
        _, rendered_mask, _, rendered_depth, _ = render_model_buffers(
            self.vertices,
            self.faces,
            object_in_camera,
            self.k_matrix,
            image_size=self.image_size,
        )

        # Convert rendered depth to uint16 format (simulating real depth camera)
        # Assume depth_scale=0.001 (millimeters)
        depth_scale = 0.001
        synthetic_depth_image = (rendered_depth / depth_scale).astype(np.uint16)
        synthetic_depth_image[~np.isfinite(rendered_depth)] = 0
        synthetic_depth_image[rendered_depth <= 0] = 0

        # Validate using the same pose
        result = validate_rendered_depth_consistency(
            self.vertices,
            self.faces,
            object_in_camera,
            self.k_matrix,
            synthetic_depth_image,
            image_size=self.image_size,
            depth_scale=depth_scale,
            max_depth_residual_m=0.02,
            min_consistency_ratio=0.70,
            min_overlap_pixels=50,
        )

        # Assertions
        self.assertTrue(
            result.success,
            f"Synthetic validation should succeed. Reason: {result.reason}",
        )
        self.assertGreater(
            result.consistency_ratio,
            0.95,
            f"Consistency ratio should be >0.95 for synthetic data, got {result.consistency_ratio:.3f}",
        )
        self.assertGreater(
            result.valid_overlap_count,
            100,
            f"Should have substantial overlap, got {result.valid_overlap_count}",
        )
        if result.mean_abs_depth_residual_m is not None:
            self.assertLess(
                result.mean_abs_depth_residual_m,
                0.005,
                f"Mean residual should be <5mm for synthetic, got {result.mean_abs_depth_residual_m:.4f}m",
            )

    def test_rotated_pose_consistency(self):
        """Test with a rotated pose."""
        # Rotate 30 degrees around Y axis, translate to 0.4m
        angle = np.deg2rad(30.0)
        c, s = np.cos(angle), np.sin(angle)
        object_in_camera = np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Render synthetic depth
        _, rendered_mask, _, rendered_depth, _ = render_model_buffers(
            self.vertices,
            self.faces,
            object_in_camera,
            self.k_matrix,
            image_size=self.image_size,
        )

        # Convert to uint16
        depth_scale = 0.001
        synthetic_depth_image = (rendered_depth / depth_scale).astype(np.uint16)
        synthetic_depth_image[~np.isfinite(rendered_depth)] = 0
        synthetic_depth_image[rendered_depth <= 0] = 0

        # Validate
        result = validate_rendered_depth_consistency(
            self.vertices,
            self.faces,
            object_in_camera,
            self.k_matrix,
            synthetic_depth_image,
            image_size=self.image_size,
            depth_scale=depth_scale,
            max_depth_residual_m=0.02,
            min_consistency_ratio=0.70,
            min_overlap_pixels=50,
        )

        # Assertions
        self.assertTrue(
            result.success,
            f"Rotated synthetic validation should succeed. Reason: {result.reason}",
        )
        self.assertGreater(
            result.consistency_ratio,
            0.95,
            f"Consistency ratio should be >0.95 for rotated synthetic, got {result.consistency_ratio:.3f}",
        )

    def test_wrong_pose_should_fail(self):
        """Test that validation correctly fails with wrong pose."""
        # Render with one pose
        correct_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        _, _, _, rendered_depth, _ = render_model_buffers(
            self.vertices,
            self.faces,
            correct_pose,
            self.k_matrix,
            image_size=self.image_size,
        )

        depth_scale = 0.001
        synthetic_depth_image = (rendered_depth / depth_scale).astype(np.uint16)
        synthetic_depth_image[~np.isfinite(rendered_depth)] = 0
        synthetic_depth_image[rendered_depth <= 0] = 0

        # Validate with a significantly different pose (translate 0.1m forward)
        wrong_pose = correct_pose.copy()
        wrong_pose[2, 3] = 0.4  # Move from 0.5m to 0.4m

        result = validate_rendered_depth_consistency(
            self.vertices,
            self.faces,
            wrong_pose,
            self.k_matrix,
            synthetic_depth_image,
            image_size=self.image_size,
            depth_scale=depth_scale,
            max_depth_residual_m=0.02,
            min_consistency_ratio=0.70,
            min_overlap_pixels=50,
        )

        # Should fail due to depth mismatch
        self.assertFalse(
            result.success,
            "Validation with wrong pose should fail",
        )
        self.assertLess(
            result.consistency_ratio,
            0.70,
            f"Consistency ratio should be low for wrong pose, got {result.consistency_ratio:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
