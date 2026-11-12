import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "foundation_pose"))


class GuessTranslationBboxTest(unittest.TestCase):
    def test_pixel_center_comes_from_bbox(self):
        from estimater import FoundationPose

        depth = np.full((120, 160), 0.5)
        mask = np.zeros_like(depth, dtype=bool)
        mask[40:80, 60:80] = True
        K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])

        c = FoundationPose.guess_translation(
            SimpleNamespace(debug=0), depth, mask, K, bbox_xyxy=[60, 40, 100, 80]
        )

        np.testing.assert_allclose(c, [0.0, 0.0, 0.5], atol=1e-9)

    def test_without_bbox_uses_mask_extent(self):
        from estimater import FoundationPose

        depth = np.full((120, 160), 0.5)
        mask = np.zeros_like(depth, dtype=bool)
        mask[40:80, 60:100] = True
        K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])

        c = FoundationPose.guess_translation(SimpleNamespace(debug=0), depth, mask, K)

        np.testing.assert_allclose(c, [-0.0025, -0.0025, 0.5], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
