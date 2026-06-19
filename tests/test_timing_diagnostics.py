import unittest

from scripts.timing_diagnostics import make_timing_event


class TimingDiagnosticsTest(unittest.TestCase):
    def test_make_timing_event_reports_frame_age_and_duration(self):
        event = make_timing_event(
            "foundationpose_register_done",
            frame_stamp_ns=1_000_000_000,
            node_now_ns=1_250_000_000,
            started_ns=1_100_000_000,
            extra={"selected_bbox": [10, 20, 30, 40]},
        )

        self.assertEqual(event["event"], "foundationpose_register_done")
        self.assertEqual(event["frame_stamp_ns"], 1_000_000_000)
        self.assertEqual(event["node_now_ns"], 1_250_000_000)
        self.assertAlmostEqual(event["age_ms"], 250.0)
        self.assertAlmostEqual(event["duration_ms"], 150.0)
        self.assertEqual(event["selected_bbox"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
