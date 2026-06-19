import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.rgbd_frame_saver import save_rgbd_frame


class CameraInfo:
    width = 4
    height = 3
    distortion_model = "plumb_bob"
    d = [0.0, 0.0, 0.0, 0.0, 0.0]
    k = [100.0, 0.0, 2.0, 0.0, 100.0, 1.5, 0.0, 0.0, 1.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [100.0, 0.0, 2.0, 0.0, 0.0, 100.0, 1.5, 0.0, 0.0, 0.0, 1.0, 0.0]


class RgbdFrameSaverTest(unittest.TestCase):
    def test_save_rgbd_frame_writes_color_depth_camera_info_and_metadata(self):
        color = np.zeros((3, 4, 3), dtype=np.uint8)
        color[:, :, 1] = 255
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "frame_000000"
            paths = save_rgbd_frame(output_dir, color, depth, CameraInfo(), 123456789)

            self.assertTrue(paths["color"].exists())
            self.assertTrue(paths["depth"].exists())
            self.assertTrue(paths["camera_info"].exists())
            self.assertTrue(paths["metadata"].exists())
            loaded_depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(loaded_depth, depth)
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["stamp_ns"], 123456789)
            self.assertEqual(metadata["color_file"], "color.png")


if __name__ == "__main__":
    unittest.main()
