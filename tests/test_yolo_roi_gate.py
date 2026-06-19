import unittest

import numpy as np

from scripts.yolo_roi_gate import YoloRoiGateParams, validate_yolo_roi


class YoloRoiGateTest(unittest.TestCase):
    def test_black_white_roi_with_depth_passes(self):
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[:, :50] = (25, 25, 25)
        color[:, 50:] = (220, 220, 220)
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[10:90, 10:90] = 350

        result = validate_yolo_roi(
            color_bgr=color,
            depth_image=depth,
            bbox_xyxy=(10, 10, 90, 90),
            params=YoloRoiGateParams(min_roi_area_px=100, min_valid_depth_ratio=0.8),
        )

        self.assertTrue(result.success, result.reason)
        self.assertGreaterEqual(result.low_saturation_ratio, 0.8)
        self.assertGreaterEqual(result.black_white_support, 0.4)

    def test_colorful_false_positive_fails_before_foundationpose(self):
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        color[:, :] = (0, 0, 255)
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[10:90, 10:90] = 350

        result = validate_yolo_roi(
            color_bgr=color,
            depth_image=depth,
            bbox_xyxy=(10, 10, 90, 90),
            params=YoloRoiGateParams(min_roi_area_px=100, min_valid_depth_ratio=0.8),
        )

        self.assertFalse(result.success)
        self.assertIn("saturation", result.reason)

    def test_roi_without_depth_fails_before_foundationpose(self):
        color = np.full((100, 100, 3), 220, dtype=np.uint8)
        depth = np.zeros((100, 100), dtype=np.uint16)

        result = validate_yolo_roi(
            color_bgr=color,
            depth_image=depth,
            bbox_xyxy=(10, 10, 90, 90),
            params=YoloRoiGateParams(min_roi_area_px=100, min_valid_depth_ratio=0.8),
        )

        self.assertFalse(result.success)
        self.assertIn("depth", result.reason)


if __name__ == "__main__":
    unittest.main()
