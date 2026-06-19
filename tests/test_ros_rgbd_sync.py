import unittest

from scripts.ros_rgbd_sync import StampedFrameBuffer, find_nearest_stamp, stamp_to_nanoseconds


class Stamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class RosRgbdSyncTest(unittest.TestCase):
    def test_stamp_to_nanoseconds_uses_ros_time_fields(self):
        self.assertEqual(stamp_to_nanoseconds(Stamp(12, 345)), 12_000_000_345)

    def test_find_nearest_stamp_accepts_exact_or_nearest_within_tolerance(self):
        stamps = [100, 130, 180]

        self.assertEqual(find_nearest_stamp(132, stamps, 5), 130)
        self.assertEqual(find_nearest_stamp(100, stamps, 0), 100)

    def test_find_nearest_stamp_rejects_frames_outside_tolerance(self):
        self.assertIsNone(find_nearest_stamp(150, [100, 180], 20))

    def test_stamped_frame_buffer_returns_nearest_frame_and_prunes_oldest(self):
        buffer = StampedFrameBuffer(max_size=2)
        buffer.put(100, "old")
        buffer.put(200, "middle")
        buffer.put(300, "new")

        self.assertIsNone(buffer.get_nearest(100, tolerance_ns=0))
        self.assertEqual(buffer.get_nearest(205, tolerance_ns=10), (200, "middle"))
        self.assertEqual(buffer.get_nearest(295, tolerance_ns=10), (300, "new"))


if __name__ == "__main__":
    unittest.main()
