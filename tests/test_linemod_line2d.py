import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.linemod_depth_features import compute_depth_normal_image, quantize_normal_orientations
from scripts.linemod_line2d import (
    extract_line2d_template,
    Line2DFeature,
    Line2DTemplate,
    load_line2d_templates,
    match_line2d_templates,
    save_line2d_templates,
)


def rectangle_image(x1, y1, x2, y2, size=(80, 80)):
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return image, mask


class LinemodLine2DTest(unittest.TestCase):
    def test_extract_line2d_template_selects_sparse_gradient_features(self):
        image, mask = rectangle_image(20, 20, 50, 50)

        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=3,
            max_features=32,
            feature_spacing=4,
        )

        self.assertEqual(template.class_id, "rect")
        self.assertEqual(template.template_id, 3)
        self.assertGreater(len(template.features), 8)
        self.assertLessEqual(len(template.features), 32)

    def test_extract_line2d_template_ignores_internal_edges_when_contour_band_is_enabled(self):
        image, mask = rectangle_image(10, 10, 69, 69)
        cv2.line(image, (40, 12), (40, 67), (0, 0, 0), 2)

        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=8,
            max_features=32,
            feature_spacing=3,
            contour_band_px=3,
        )

        self.assertGreater(len(template.features), 0)
        self.assertTrue(
            all(
                feature.x <= 4
                or feature.y <= 4
                or feature.x >= template.width - 5
                or feature.y >= template.height - 5
                for feature in template.features
            )
        )

    def test_match_line2d_templates_finds_shifted_rectangle(self):
        template_image, template_mask = rectangle_image(20, 20, 50, 50)
        template = extract_line2d_template(
            template_image,
            template_mask,
            class_id="rect",
            template_id=1,
            max_features=48,
            feature_spacing=3,
        )
        scene, _ = rectangle_image(30, 25, 60, 55)

        matches = match_line2d_templates(scene, [template], threshold=70.0)

        self.assertGreaterEqual(len(matches), 1)
        best = matches[0]
        self.assertEqual(best.class_id, "rect")
        self.assertEqual(best.template_id, 1)
        self.assertAlmostEqual(best.x, 30, delta=2)
        self.assertAlmostEqual(best.y, 25, delta=2)

    def test_save_and_load_line2d_templates_round_trips_features(self):
        image, mask = rectangle_image(20, 20, 50, 50)
        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=4,
            max_features=16,
            feature_spacing=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "line2d_templates.json"
            save_line2d_templates(path, [template])
            loaded = load_line2d_templates(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].class_id, "rect")
        self.assertEqual(loaded[0].template_id, 4)
        self.assertEqual(loaded[0].features, template.features)

    def test_depth_normal_score_contributes_to_match_similarity(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        depth = np.zeros((80, 80), dtype=np.float64)
        xs = np.arange(80, dtype=np.float64)
        depth[:, :] = 0.45 + xs[None, :] * 0.001
        k_matrix = np.array(
            [[500.0, 0.0, 40.0], [0.0, 500.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        observed_normal_orientation = int(
            quantize_normal_orientations(compute_depth_normal_image(depth, k_matrix))[20, 20]
        )
        template = Line2DTemplate(
            class_id="rect",
            template_id=9,
            width=40,
            height=40,
            bbox=(0, 0, 40, 40),
            features=(Line2DFeature(20, 20, 0),),
            normal_features=(
                Line2DFeature(20, 20, observed_normal_orientation),
                Line2DFeature(24, 20, observed_normal_orientation),
            ),
        )

        color_only = match_line2d_templates(
            image,
            [template],
            threshold=0.0,
            stride=80,
            min_magnitude=999.0,
        )[0]
        rgbd = match_line2d_templates(
            image,
            [template],
            threshold=0.0,
            stride=80,
            min_magnitude=999.0,
            depth_image=depth,
            k_matrix=k_matrix,
            line2d_score_weight=0.0,
            normal_score_weight=1.0,
        )[0]

        self.assertEqual(color_only.similarity, 0.0)
        self.assertGreater(rgbd.similarity, 0.0)

    def test_match_line2d_templates_limits_search_to_roi_boxes(self):
        template_image, template_mask = rectangle_image(20, 20, 50, 50)
        template = extract_line2d_template(
            template_image,
            template_mask,
            class_id="rect",
            template_id=2,
            max_features=48,
            feature_spacing=3,
        )
        scene, _ = rectangle_image(30, 25, 60, 55)

        rejected = match_line2d_templates(
            scene,
            [template],
            threshold=70.0,
            roi_boxes=[(0, 0, 20, 20)],
        )
        accepted = match_line2d_templates(
            scene,
            [template],
            threshold=70.0,
            roi_boxes=[(20, 15, 75, 70)],
        )

        self.assertEqual(rejected, [])
        self.assertGreaterEqual(len(accepted), 1)
        self.assertAlmostEqual(accepted[0].x, 30, delta=2)
        self.assertAlmostEqual(accepted[0].y, 25, delta=2)


if __name__ == "__main__":
    unittest.main()
