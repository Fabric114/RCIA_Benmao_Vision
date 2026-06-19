import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.linemod_depth_validation import validate_depth_support
from scripts.linemod_init_frame import load_yolo_roi_boxes, run_init_frame
from scripts.linemod_line2d import extract_line2d_template, save_line2d_templates
from scripts.linemod_template_store import TemplatePoseMetadata, save_template_metadata


def rectangle_image(x1=20, y1=20, x2=50, y2=50, size=(80, 80)):
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return image, mask


class LinemodDepthValidationTest(unittest.TestCase):
    def test_load_yolo_roi_boxes_uses_model_as_roi_provider(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def __getitem__(self, index):
                return self.value[index]

        class FakeBox:
            xyxy = [FakeTensor([10.0, 20.0, 50.0, 70.0])]
            conf = [0.91]
            cls = [0]

        class FakeBoxes:
            def __len__(self):
                return 1

            def __iter__(self):
                return iter([FakeBox()])

        class FakeResult:
            boxes = FakeBoxes()

        class FakeYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def predict(self, image_rgb, conf, verbose):
                self.last_conf = conf
                return [FakeResult()]

        fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
        image = np.zeros((80, 80, 3), dtype=np.uint8)

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            boxes = load_yolo_roi_boxes(
                image,
                "models/yolo/best.pt",
                target_class=0,
                confidence=0.7,
            )

        self.assertEqual(boxes, [(10.0, 20.0, 50.0, 70.0)])

    def test_validate_depth_support_accepts_pose_with_consistent_feature_depths(self):
        depth = np.zeros((80, 80), dtype=np.uint16)
        depth[20:51, 20:51] = 500
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]
        template = argparse.Namespace(
            features=(
                argparse.Namespace(x=10, y=10),
                argparse.Namespace(x=20, y=20),
                argparse.Namespace(x=30, y=30),
            )
        )
        match = argparse.Namespace(x=10, y=10)

        result = validate_depth_support(
            depth,
            match,
            template,
            pose,
            depth_scale=0.001,
            max_abs_z_error_m=0.03,
            min_valid_depths=3,
        )

        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.valid_depth_count, 3)
        self.assertAlmostEqual(result.observed_median_depth_m, 0.5, places=3)
        self.assertLess(result.abs_z_error_m, 0.03)

    def test_validate_depth_support_rejects_pose_without_depth_support(self):
        depth = np.zeros((80, 80), dtype=np.uint16)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]
        template = argparse.Namespace(features=(argparse.Namespace(x=10, y=10),))
        match = argparse.Namespace(x=10, y=10)

        result = validate_depth_support(
            depth,
            match,
            template,
            pose,
            depth_scale=0.001,
            min_valid_depths=1,
        )

        self.assertFalse(result.success)
        self.assertIn("valid depth", result.reason)

    def test_init_frame_does_not_write_final_pose_when_depth_rejects_all_candidates(self):
        image, mask = rectangle_image()
        model_point_image = np.zeros((80, 80, 3), dtype=np.float64)
        ys, xs = np.where(mask > 0)
        model_point_image[ys, xs, 0] = (xs.astype(np.float64) - 35.0) * 0.001
        model_point_image[ys, xs, 1] = (ys.astype(np.float64) - 35.0) * 0.001
        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=1,
            max_features=16,
            feature_spacing=4,
            model_point_image=model_point_image,
            require_model_points=True,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            color_path = tmpdir / "color.png"
            depth_path = tmpdir / "depth.png"
            camera_info_path = tmpdir / "camera_info.yaml"
            templates_path = tmpdir / "templates.json"
            poses_path = tmpdir / "poses.json"
            output_dir = tmpdir / "out"
            cv2.imwrite(str(color_path), image)
            cv2.imwrite(str(depth_path), np.zeros((80, 80), dtype=np.uint16))
            camera_info_path.write_text(
                "k: [600.0, 0.0, 40.0, 0.0, 600.0, 40.0, 0.0, 0.0, 1.0]\n",
                encoding="utf-8",
            )
            save_line2d_templates(templates_path, [template])
            save_template_metadata(
                poses_path,
                [TemplatePoseMetadata("rect", 1, pose, [35.0, 35.0], "unit-test")],
            )

            result = run_init_frame(
                argparse.Namespace(
                    templates=templates_path,
                    poses=poses_path,
                    color=color_path,
                    depth=depth_path,
                    camera_info=camera_info_path,
                    output_dir=output_dir,
                    top_k=3,
                    threshold=70.0,
                    stride=1,
                    depth_scale=0.001,
                    max_abs_z_error_m=0.03,
                    min_valid_depths=3,
                )
            )

            self.assertFalse(result["success"])
            self.assertTrue((output_dir / "candidate_scores.json").exists())
            self.assertFalse((output_dir / "final_pose.txt").exists())

    def test_init_frame_rejects_candidate_that_fails_black_white_appearance(self):
        image, mask = rectangle_image()
        image[mask > 0] = (0, 0, 255)
        model_point_image = np.zeros((80, 80, 3), dtype=np.float64)
        ys, xs = np.where(mask > 0)
        model_point_image[ys, xs, 0] = (xs.astype(np.float64) - 35.0) * 0.001
        model_point_image[ys, xs, 1] = (ys.astype(np.float64) - 35.0) * 0.001
        template = extract_line2d_template(
            image,
            mask,
            class_id="rect",
            template_id=2,
            max_features=16,
            feature_spacing=4,
            model_point_image=model_point_image,
            require_model_points=True,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            color_path = tmpdir / "color.png"
            depth_path = tmpdir / "depth.png"
            camera_info_path = tmpdir / "camera_info.yaml"
            templates_path = tmpdir / "templates.json"
            poses_path = tmpdir / "poses.json"
            output_dir = tmpdir / "out"
            depth = np.zeros((80, 80), dtype=np.uint16)
            depth[mask > 0] = 500
            cv2.imwrite(str(color_path), image)
            cv2.imwrite(str(depth_path), depth)
            camera_info_path.write_text(
                "k: [600.0, 0.0, 40.0, 0.0, 600.0, 40.0, 0.0, 0.0, 1.0]\n",
                encoding="utf-8",
            )
            save_line2d_templates(templates_path, [template])
            save_template_metadata(
                poses_path,
                [TemplatePoseMetadata("rect", 2, pose, [35.0, 35.0], "unit-test")],
            )

            result = run_init_frame(
                argparse.Namespace(
                    templates=templates_path,
                    poses=poses_path,
                    color=color_path,
                    depth=depth_path,
                    camera_info=camera_info_path,
                    output_dir=output_dir,
                    top_k=3,
                    threshold=70.0,
                    stride=1,
                    depth_scale=0.001,
                    max_abs_z_error_m=0.03,
                    min_valid_depths=3,
                    enable_black_white_validation=True,
                )
            )

            self.assertFalse(result["success"])
            self.assertFalse((output_dir / "final_pose.txt").exists())
            self.assertFalse(result["candidates"][0]["black_white_appearance"]["success"])
            self.assertTrue((output_dir / "black_white_validation_map.png").exists())


if __name__ == "__main__":
    unittest.main()
