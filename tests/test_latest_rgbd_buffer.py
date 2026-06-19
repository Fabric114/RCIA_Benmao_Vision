import unittest

from scripts.latest_rgbd_buffer import LatestRgbdBuffer


class LatestRgbdBufferTest(unittest.TestCase):
    def test_returns_latest_color_with_nearest_depth(self):
        buffer = LatestRgbdBuffer()
        buffer.put_color(1_000, "color1")
        buffer.put_depth(1_002, "depth1")

        pair = buffer.get_latest_pair(now_ns=1_010, tolerance_ns=5, max_age_ns=20)

        self.assertIsNotNone(pair)
        self.assertEqual(pair.color_stamp_ns, 1_000)
        self.assertEqual(pair.depth_stamp_ns, 1_002)
        self.assertEqual(pair.color, "color1")
        self.assertEqual(pair.depth, "depth1")

    def test_ignores_older_color_after_newer_color(self):
        buffer = LatestRgbdBuffer()
        buffer.put_color(2_000, "new")
        buffer.put_color(1_000, "old")
        buffer.put_depth(2_000, "depth")

        pair = buffer.get_latest_pair(now_ns=2_010, tolerance_ns=5, max_age_ns=20)

        self.assertEqual(pair.color_stamp_ns, 2_000)
        self.assertEqual(pair.color, "new")

    def test_rejects_stale_latest_color(self):
        buffer = LatestRgbdBuffer()
        buffer.put_color(1_000, "color")
        buffer.put_depth(1_000, "depth")

        pair = buffer.get_latest_pair(now_ns=1_200, tolerance_ns=5, max_age_ns=100)

        self.assertIsNone(pair)

    def test_rejects_depth_outside_sync_tolerance(self):
        buffer = LatestRgbdBuffer()
        buffer.put_color(1_000, "color")
        buffer.put_depth(1_020, "depth")

        pair = buffer.get_latest_pair(now_ns=1_010, tolerance_ns=5, max_age_ns=100)

        self.assertIsNone(pair)

    def test_returns_nearest_depth_for_latest_color(self):
        buffer = LatestRgbdBuffer()
        buffer.put_color(1_000, "color")
        buffer.put_depth(990, "depth_old")
        buffer.put_depth(1_003, "depth_near")

        pair = buffer.get_latest_pair(now_ns=1_010, tolerance_ns=20, max_age_ns=100)

        self.assertEqual(pair.depth_stamp_ns, 1_003)
        self.assertEqual(pair.depth, "depth_near")

    def test_returns_historical_color_and_depth_by_stamp(self):
        buffer = LatestRgbdBuffer(color_history_size=4, depth_history_size=4)
        buffer.put_color(1_000, "color_old")
        buffer.put_depth(1_002, "depth_old")
        buffer.put_color(2_000, "color_new")
        buffer.put_depth(2_001, "depth_new")

        pair = buffer.get_pair_at(target_stamp_ns=1_000, tolerance_ns=5)

        self.assertIsNotNone(pair)
        self.assertEqual(pair.color_stamp_ns, 1_000)
        self.assertEqual(pair.depth_stamp_ns, 1_002)
        self.assertEqual(pair.color, "color_old")
        self.assertEqual(pair.depth, "depth_old")

    def test_historical_pair_returns_none_when_outside_tolerance(self):
        buffer = LatestRgbdBuffer(color_history_size=4, depth_history_size=4)
        buffer.put_color(1_000, "color")
        buffer.put_depth(1_030, "depth")

        self.assertIsNone(buffer.get_pair_at(target_stamp_ns=1_000, tolerance_ns=5))


if __name__ == "__main__":
    unittest.main()
