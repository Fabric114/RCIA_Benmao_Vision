import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.linemod_line2d import (
    extract_line2d_template,
    load_line2d_templates,
    save_line2d_templates,
)
from scripts.linemod_silhouette_renderer import render_silhouette_with_model_points


class LinemodFeatureCorrespondenceTest(unittest.TestCase):
    def test_renderer_returns_object_space_model_points_for_mask_pixels(self):
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
        object_in_camera = np.eye(4)
        object_in_camera[:3, 3] = [0.0, 0.0, 0.5]
        k_matrix = np.array(
            [
                [600.0, 0.0, 80.0],
                [0.0, 600.0, 60.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        color, mask, model_point_image = render_silhouette_with_model_points(
            vertices,
            faces,
            object_in_camera,
            k_matrix,
            image_size=(160, 120),
        )

        self.assertEqual(color.shape, (120, 160, 3))
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(model_point_image.shape, (120, 160, 3))
        self.assertGreater(int(mask.sum()), 0)
        np.testing.assert_allclose(model_point_image[60, 80], [0.0, 0.0, 0.0], atol=1e-3)
        self.assertTrue(np.isfinite(model_point_image[mask > 0]).all())

    def test_extract_template_attaches_model_points_and_json_round_trips(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (50, 50), (255, 255, 255), -1)
        cv2.rectangle(mask, (20, 20), (50, 50), 255, -1)
        model_point_image = np.full((80, 80, 3), np.nan, dtype=np.float64)
        ys, xs = np.where(mask > 0)
        model_point_image[ys, xs, 0] = xs.astype(np.float64) * 0.001
        model_point_image[ys, xs, 1] = ys.astype(np.float64) * 0.001
        model_point_image[ys, xs, 2] = 0.0

        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=11,
            max_features=16,
            feature_spacing=4,
            model_point_image=model_point_image,
            require_model_points=True,
        )

        self.assertGreater(len(template.features), 0)
        self.assertTrue(all(feature.model_point is not None for feature in template.features))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "templates.json"
            save_line2d_templates(path, [template])
            loaded = load_line2d_templates(path)

        self.assertEqual(loaded[0].features, template.features)

    def test_old_template_json_without_model_points_still_loads(self):
        payload = {
            "version": 1,
            "templates": [
                {
                    "class_id": "rect",
                    "template_id": 1,
                    "width": 10,
                    "height": 10,
                    "bbox": [0, 0, 10, 10],
                    "features": [{"x": 1, "y": 2, "orientation": 3}],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_templates.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_line2d_templates(path)

        self.assertIsNone(loaded[0].features[0].model_point)


if __name__ == "__main__":
    unittest.main()
