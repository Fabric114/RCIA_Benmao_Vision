import importlib.util
from pathlib import Path
import unittest


def load_tracking_supervisor_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "ros2/rcia_energy_trace_ros/scripts/tracking_supervisor_node.py"
    )
    spec = importlib.util.spec_from_file_location("tracking_supervisor_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackingSupervisorNodeHelpersTest(unittest.TestCase):
    def test_empty_health_matches_tracking_health_result_contract(self):
        module = load_tracking_supervisor_node()

        result = module.TrackingSupervisorNode._empty_health("missing M3T pose", None)

        self.assertFalse(result.tracking_ok)
        self.assertFalse(result.depth_ok)
        self.assertFalse(result.black_white_ok)
        self.assertEqual(result.reason, "missing M3T pose")

    def test_empty_health_is_not_evaluated(self):
        module = load_tracking_supervisor_node()

        empty = module.TrackingSupervisorNode._empty_health("missing M3T pose", None)
        empty_with_yolo = module.TrackingSupervisorNode._empty_health(
            "stale M3T pose", (10, 10, 50, 50)
        )

        self.assertFalse(empty.health_evaluated)
        self.assertFalse(empty.depth_evaluable)
        self.assertEqual(empty.valid_depth_overlap_count, 0)
        self.assertFalse(empty_with_yolo.health_evaluated)
        self.assertFalse(empty_with_yolo.depth_evaluable)


if __name__ == "__main__":
    unittest.main()
