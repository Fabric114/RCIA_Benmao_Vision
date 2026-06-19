import unittest

from scripts.auto_register import AutoRegisterGate


class AutoRegisterGateTest(unittest.TestCase):
    def test_gate_waits_for_confident_stable_detections(self):
        gate = AutoRegisterGate(min_conf=0.7, stable_frames=2)

        self.assertIsNone(gate.update([(0, 0, 10, 10, 0.69, 0)]))
        self.assertIsNone(gate.update([(0, 0, 10, 10, 0.71, 0)]))

        selected = gate.update([(1, 1, 11, 11, 0.72, 0)])

        self.assertEqual(selected, (1, 1, 11, 11, 0.72, 0))

    def test_gate_resets_when_detection_drops_below_threshold(self):
        gate = AutoRegisterGate(min_conf=0.7, stable_frames=2)

        self.assertIsNone(gate.update([(0, 0, 10, 10, 0.8, 0)]))
        self.assertIsNone(gate.update([]))
        self.assertIsNone(gate.update([(0, 0, 10, 10, 0.8, 0)]))

    def test_gate_selects_highest_confidence_detection(self):
        gate = AutoRegisterGate(min_conf=0.7, stable_frames=1)

        selected = gate.update([
            (0, 0, 100, 100, 0.75, 0),
            (0, 0, 10, 10, 0.9, 0),
        ])

        self.assertEqual(selected, (0, 0, 10, 10, 0.9, 0))

    def test_reset_clears_stable_count(self):
        gate = AutoRegisterGate(min_conf=0.7, stable_frames=2)
        self.assertIsNone(gate.update([(0, 0, 10, 10, 0.8, 0)]))

        gate.reset()
        after_reset = gate.update([(0, 0, 10, 10, 0.8, 0)])

        self.assertIsNone(after_reset)


if __name__ == "__main__":
    unittest.main()
