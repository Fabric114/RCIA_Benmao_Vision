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


if __name__ == "__main__":
    unittest.main()
