import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.linemod_depth_features import (
    extract_depth_normal_features,
    quantize_normal_orientations,
)
from scripts.linemod_line2d import Line2DTemplate, load_line2d_templates, save_line2d_templates


class LinemodDepthFeaturesTest(unittest.TestCase):
    def test_quantize_normal_orientations_is_deterministic(self):
        normal_image = np.zeros((2, 2, 3), dtype=np.float64)
        normal_image[0, 0] = [1.0, 0.0, 0.0]
        normal_image[0, 1] = [0.0, 1.0, 0.0]
        normal_image[1, 0] = [-1.0, 0.0, 0.0]
        normal_image[1, 1] = [0.0, -1.0, 0.0]

        quantized = quantize_normal_orientations(normal_image)

        self.assertEqual(quantized.shape, (2, 2))
        self.assertEqual(int(quantized[0, 0]), 0)
        self.assertEqual(int(quantized[0, 1]), 2)
        self.assertEqual(int(quantized[1, 0]), 4)
        self.assertEqual(int(quantized[1, 1]), 6)

    def test_extract_depth_normal_features_uses_object_interior_and_round_trips_json(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.rectangle(mask, (8, 8), (31, 31), 255, -1)
        normal_image = np.full((40, 40, 3), np.nan, dtype=np.float64)
        normal_image[mask > 0] = [1.0, 0.0, 0.0]
        model_point_image = np.full((40, 40, 3), np.nan, dtype=np.float64)
        ys, xs = np.where(mask > 0)
        model_point_image[ys, xs, 0] = xs.astype(np.float64) * 0.001
        model_point_image[ys, xs, 1] = ys.astype(np.float64) * 0.001
        model_point_image[ys, xs, 2] = 0.0

        normal_features = extract_depth_normal_features(
            normal_image,
            mask,
            bbox=(8, 8, 24, 24),
            model_point_image=model_point_image,
            max_features=12,
            feature_spacing=5,
        )
        template = Line2DTemplate(
            class_id="rect",
            template_id=1,
            width=24,
            height=24,
            bbox=(8, 8, 24, 24),
            features=tuple(),
            normal_features=tuple(normal_features),
        )

        self.assertGreater(len(template.normal_features), 0)
        self.assertTrue(all(feature.model_point is not None for feature in template.normal_features))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "templates.json"
            save_line2d_templates(path, [template])
            loaded = load_line2d_templates(path)

        self.assertEqual(loaded[0].normal_features, template.normal_features)

    def test_extract_depth_normal_features_prefers_stable_interior_over_near_border(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (49, 49), 255, -1)
        normal_image = np.full((60, 60, 3), np.nan, dtype=np.float64)
        normal_image[mask > 0] = [1.0, 0.0, 0.0]
        model_point_image = np.zeros((60, 60, 3), dtype=np.float64)

        normal_features = extract_depth_normal_features(
            normal_image,
            mask,
            bbox=(10, 10, 40, 40),
            model_point_image=model_point_image,
            max_features=8,
            feature_spacing=6,
            erosion_iterations=1,
        )

        self.assertGreater(len(normal_features), 0)
        for feature in normal_features:
            absolute_x = feature.x + 10
            absolute_y = feature.y + 10
            distance_to_border = min(
                absolute_x - 10,
                absolute_y - 10,
                49 - absolute_x,
                49 - absolute_y,
            )
            self.assertGreaterEqual(distance_to_border, 6)


if __name__ == "__main__":
    unittest.main()
