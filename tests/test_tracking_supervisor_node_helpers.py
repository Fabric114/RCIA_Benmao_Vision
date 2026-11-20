import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np


def load_tracking_supervisor_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "ros2/rcia_energy_trace_ros/scripts/tracking_supervisor_node.py"
    )
    spec = importlib.util.spec_from_file_location("tracking_supervisor_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pose_msg(
    px=0.0, py=0.0, pz=0.5, qx=0.0, qy=0.0, qz=0.0, qw=1.0
):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=px, y=py, z=pz),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        )
    )


def make_fake_node(module):
    """Minimal stand-in for the node: only the attributes the tested methods touch."""
    return SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=1e9),
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: None),
        lock=threading.Lock(),
        vertices=None,
        faces=None,
        previous_valid_m3t_pose=None,
        tracking_health_params=None,
        latest_m3t_pose=None,
        _empty_health=module.TrackingSupervisorNode._empty_health,
    )


def make_health_inputs():
    return {
        "now_ns": 10,
        "m3t_pose": {"stamp_ns": 10, "matrix": np.eye(4)},
        "pair": SimpleNamespace(
            color=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.zeros((4, 4), dtype=np.uint16),
        ),
        "camera_info": object(),
    }


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

    def test_pose_msg_to_matrix_accepts_finite_pose(self):
        module = load_tracking_supervisor_node()

        matrix = module.pose_msg_to_matrix(make_pose_msg(px=0.1, pz=0.5))

        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertAlmostEqual(matrix[0, 3], 0.1)
        self.assertAlmostEqual(matrix[2, 3], 0.5)

    def test_pose_msg_to_matrix_rejects_nonfinite_translation(self):
        module = load_tracking_supervisor_node()

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    module.pose_msg_to_matrix(make_pose_msg(px=bad))

    def test_pose_msg_to_matrix_rejects_nonfinite_quaternion(self):
        module = load_tracking_supervisor_node()

        with self.assertRaises(ValueError):
            module.pose_msg_to_matrix(make_pose_msg(qw=float("nan")))
        with self.assertRaises(ValueError):
            module.pose_msg_to_matrix(make_pose_msg(qx=float("inf")))

    def test_pose_msg_to_matrix_rejects_zero_quaternion(self):
        module = load_tracking_supervisor_node()

        with self.assertRaises(ValueError):
            module.pose_msg_to_matrix(make_pose_msg(qw=0.0))

    def test_invalid_m3t_pose_clears_previous_pose(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.latest_m3t_pose = {"stamp_ns": 1, "matrix": np.eye(4), "msg": None}

        module.TrackingSupervisorNode._m3t_pose_cb(
            fake, make_pose_msg(px=float("nan"))
        )

        self.assertIsNone(fake.latest_m3t_pose)

    def test_m3t_health_exception_reports_pose_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", side_effect=RuntimeError("boom")
        ), mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = module.TrackingSupervisorNode._m3t_health(
                fake, make_health_inputs(), None
            )

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        self.assertFalse(health.tracking_ok)
        self.assertIn("validation error", health.reason)

    def test_m3t_health_evaluated_reports_pose_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        evaluated = SimpleNamespace(health_evaluated=True, tracking_ok=True)
        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated
        ), mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = module.TrackingSupervisorNode._m3t_health(
                fake, make_health_inputs(), None
            )

        self.assertIs(health, evaluated)
        self.assertTrue(available)

    def test_m3t_health_missing_rgbd_reports_pose_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        inputs = make_health_inputs()
        inputs["pair"] = None

        health, available = module.TrackingSupervisorNode._m3t_health(
            fake, inputs, None
        )

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)

    def test_m3t_health_nonfinite_pose_reports_pose_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        inputs = make_health_inputs()
        nan_pose = np.eye(4)
        nan_pose[0, 3] = float("nan")
        inputs["m3t_pose"]["matrix"] = nan_pose

        with mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = module.TrackingSupervisorNode._m3t_health(
                fake, inputs, None
            )

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        self.assertFalse(health.depth_evaluable)


if __name__ == "__main__":
    unittest.main()
