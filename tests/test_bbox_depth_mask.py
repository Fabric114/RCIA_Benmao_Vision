import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bbox_depth_mask import bbox_depth_mask  # noqa: E402


def make_scene():
    depth = np.full((120, 160), 1.0, dtype=np.float64)
    depth[40:80, 60:100] = 0.5
    return depth


class BboxDepthMaskTest(unittest.TestCase):
    def setUp(self):
        self.depth = make_scene()

    def test_selects_object_and_rejects_background(self):
        m = bbox_depth_mask(self.depth, [50, 30, 110, 90])

        self.assertTrue(m[40:80, 60:100].all())
        self.assertEqual(m.sum(), 40 * 40)

    def test_pixels_outside_bbox_never_selected(self):
        m = bbox_depth_mask(self.depth, [70, 50, 90, 70])

        self.assertFalse(m[:, :70].any())
        self.assertFalse(m[:, 90:].any())

    def test_center_hole_falls_back_to_whole_bbox(self):
        d = self.depth.copy()
        d[50:70, 70:90] = 0.0

        m = bbox_depth_mask(d, [62, 42, 98, 78])

        self.assertTrue(m[42:50, 62:98].all())
        self.assertFalse(m[50:70, 70:90].any())

    def test_no_valid_depth_returns_all_false(self):
        m = bbox_depth_mask(np.zeros_like(self.depth), [50, 30, 110, 90])

        self.assertFalse(m.any())

    def test_bbox_outside_image_is_clipped(self):
        m = bbox_depth_mask(self.depth, [-20, -20, 180, 140])

        self.assertEqual(m.shape, self.depth.shape)
        self.assertTrue(m[40:80, 60:100].all())

    def test_empty_bbox_after_clipping_returns_all_false(self):
        m = bbox_depth_mask(self.depth, [200, 200, 300, 300])

        self.assertEqual(m.shape, self.depth.shape)
        self.assertFalse(m.any())


if __name__ == "__main__":
    unittest.main()
