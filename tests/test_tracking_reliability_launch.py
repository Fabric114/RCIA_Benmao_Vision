"""Check that full_energy_tracking.launch.py binds the reliability parameters.

The launch description is evaluated with a LaunchContext, so the parameters are
the values launch_ros would hand to the tracking_supervisor node. The node test
then constructs the real TrackingSupervisorNode under rclpy.
"""

import importlib.util
import math
import unittest
from pathlib import Path

import rclpy
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = REPO_ROOT / "ros2/rcia_energy_trace_ros/launch/full_energy_tracking.launch.py"
NODE_PATH = REPO_ROOT / "ros2/rcia_energy_trace_ros/scripts/tracking_supervisor_node.py"

NEW_ARGUMENT_DEFAULTS = {
    "filter_absence_timeout_ms": 750.0,
    "filter_conflict_timeout_ms": 625.0,
    "filter_suspect_timeout_ms": 1000.0,
    "filter_startup_timeout_ms": 5000.0,
    "filter_hard_depth_mismatch_ratio": 0.05,
    "m3t_stop_ack_timeout_ms": 2000.0,
    "tracking_pose_frame_tolerance_ms": 15.0,
    "tracking_yolo_frame_tolerance_ms": 80.0,
    "disappearance_confirm_frames": 3,
}

NEW_ARGUMENT_OVERRIDES = {
    "filter_absence_timeout_ms": 900.0,
    "filter_conflict_timeout_ms": 500.0,
    "filter_suspect_timeout_ms": 1200.0,
    "filter_startup_timeout_ms": 4000.0,
    "filter_hard_depth_mismatch_ratio": 0.08,
    "m3t_stop_ack_timeout_ms": 3000.0,
    "tracking_pose_frame_tolerance_ms": 20.0,
    "tracking_yolo_frame_tolerance_ms": 90.0,
    "disappearance_confirm_frames": 5,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_supervisor_parameters(overrides=None):
    """Return the parameters the launch file gives to tracking_supervisor.

    launch_ros keeps the node name and parameters on the Node action with no
    public accessor before execution, so the private `_Node__node_name` and
    `_Node__parameters` attributes are read here.
    """
    module = load_module(LAUNCH_PATH, "full_energy_tracking_launch_under_test")
    description = module.generate_launch_description()
    context = LaunchContext()
    for name, value in (overrides or {}).items():
        context.launch_configurations[name] = str(value)
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    supervisors = [
        entity
        for entity in description.entities
        if isinstance(entity, Node) and entity._Node__node_name == "tracking_supervisor"
    ]
    if len(supervisors) != 1:
        raise ValueError(f"expected one tracking_supervisor node, found {len(supervisors)}")
    evaluated = evaluate_parameters(context, supervisors[0]._Node__parameters)
    return dict(evaluated[0])


def ros_args_for(overrides):
    args = ["--ros-args"]
    for name, value in overrides.items():
        args += ["-p", f"{name}:={value}"]
    return args


class TrackingReliabilityLaunchTest(unittest.TestCase):
    def test_new_arguments_bind_their_defaults_to_tracking_supervisor(self):
        params = resolve_supervisor_parameters()

        for name, expected in NEW_ARGUMENT_DEFAULTS.items():
            with self.subTest(name=name):
                self.assertIn(name, params)
                self.assertEqual(params[name], expected)
                self.assertIs(type(params[name]), type(expected))

    def test_new_argument_overrides_bind_to_tracking_supervisor(self):
        params = resolve_supervisor_parameters(NEW_ARGUMENT_OVERRIDES)

        for name, expected in NEW_ARGUMENT_OVERRIDES.items():
            with self.subTest(name=name):
                self.assertEqual(params[name], expected)
                self.assertIs(type(params[name]), type(expected))

    def test_existing_arguments_still_bind(self):
        params = resolve_supervisor_parameters()

        self.assertEqual(params["stable_yolo_frames"], 3)
        self.assertEqual(params["lost_frames"], 6)
        self.assertEqual(params["out_of_view_frames"], 10)
        self.assertEqual(params["filter_startup_grace_frames"], 40)
        self.assertEqual(params["tracking_min_depth_consistency_ratio"], 0.6)
        self.assertEqual(params["max_yolo_age_ms"], 500.0)
        self.assertEqual(params["candidate_frame_tolerance_ms"], 40.0)
        self.assertIs(params["canonicalize_yaw"], True)

    def test_existing_stable_frames_argument_overrides_stable_yolo_frames(self):
        params = resolve_supervisor_parameters({"stable_frames": 4})

        self.assertEqual(params["stable_yolo_frames"], 4)

    def test_validation_rejects_negative_timeouts_tolerances_and_ratios(self):
        module = load_module(NODE_PATH, "tracking_supervisor_node_validation")
        valid = {name: 1.0 for name in module.NON_NEGATIVE_PARAMS}
        valid.update({name: 1 for name in module.POSITIVE_FRAME_COUNT_PARAMS})
        module.validate_supervisor_parameters(valid)

        for name in module.NON_NEGATIVE_PARAMS:
            for bad in (-1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=bad):
                    values = dict(valid)
                    values[name] = bad
                    with self.assertRaises(ValueError):
                        module.validate_supervisor_parameters(values)

        for name in module.POSITIVE_FRAME_COUNT_PARAMS:
            for bad in (0, -2):
                with self.subTest(name=name, value=bad):
                    values = dict(valid)
                    values[name] = bad
                    with self.assertRaises(ValueError):
                        module.validate_supervisor_parameters(values)

    def test_node_constructor_rejects_negative_timeout_parameters(self):
        module = load_module(NODE_PATH, "tracking_supervisor_node_ctor_negative")
        cases = {
            "filter_absence_timeout_ms": -1.0,
            "filter_hard_depth_mismatch_ratio": -0.01,
            "tracking_yolo_frame_tolerance_ms": -5.0,
            "disappearance_confirm_frames": 0,
        }
        for name, bad in cases.items():
            with self.subTest(name=name):
                rclpy.init(args=ros_args_for({name: bad}))
                try:
                    with self.assertRaises(ValueError) as ctx:
                        module.TrackingSupervisorNode()
                    self.assertIn(name, str(ctx.exception))
                finally:
                    rclpy.shutdown()

    def test_node_constructor_applies_launch_bound_values(self):
        module = load_module(NODE_PATH, "tracking_supervisor_node_launch_bound")
        launch_values = resolve_supervisor_parameters(NEW_ARGUMENT_OVERRIDES)
        overrides = {name: launch_values[name] for name in NEW_ARGUMENT_OVERRIDES}

        rclpy.init(args=ros_args_for(overrides))
        try:
            node = module.TrackingSupervisorNode()
            try:
                for name, expected in NEW_ARGUMENT_OVERRIDES.items():
                    self.assertEqual(node.get_parameter(name).value, expected)
                filter_params = node.tracking_state_filter.params
                self.assertEqual(filter_params.absence_timeout_ms, 900.0)
                self.assertEqual(filter_params.conflict_timeout_ms, 500.0)
                self.assertEqual(filter_params.suspect_timeout_ms, 1200.0)
                self.assertEqual(filter_params.startup_timeout_ms, 4000.0)
                self.assertEqual(filter_params.hard_depth_mismatch_ratio, 0.08)
                self.assertEqual(filter_params.min_depth_support_ratio, 0.6)
                self.assertEqual(filter_params.min_valid_depth_overlap_pixels, 80)
                self.assertEqual(node.lifecycle.params.disappearance_confirm_frames, 5)
                self.assertEqual(node.get_parameter("m3t_stop_ack_timeout_ms").value, 3000.0)
                self.assertTrue(math.isclose(node.get_parameter("tracking_yolo_frame_tolerance_ms").value, 90.0))
            finally:
                node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
