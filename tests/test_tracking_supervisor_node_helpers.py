import importlib.util
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from scripts.latest_rgbd_buffer import LatestRgbdBuffer


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
    px=0.0, py=0.0, pz=0.5, qx=0.0, qy=0.0, qz=0.0, qw=1.0, stamp_ns=None
):
    msg = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=px, y=py, z=pz),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        )
    )
    if stamp_ns is not None:
        msg.header = SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
            ),
            frame_id="camera_color_optical_frame",
        )
    return msg


DEFAULT_PARAMS = {
    "max_pose_age_ms": 250.0,
    "max_yolo_age_ms": 500.0,
    "tracking_pose_frame_tolerance_ms": 15.0,
    "tracking_yolo_frame_tolerance_ms": 80.0,
}

HELPER_METHODS = (
    "_color_cb",
    "_m3t_health",
    "_m3t_rgbd_pair",
    "_evaluate_pose_health",
    "_matching_yolo_for_frame",
    "_yolo_observation_valid",
)


def make_fake_node(module, params=None):
    """Stand-in for the node that binds the real helper methods.

    Only the attributes the tested methods read are provided. Parameters come
    from `params`, so each test controls the thresholds it depends on.
    """
    values = dict(DEFAULT_PARAMS)
    values.update(params or {})
    fake = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
        get_logger=lambda: SimpleNamespace(
            warning=lambda *a, **k: None, error=lambda *a, **k: None
        ),
        lock=threading.Lock(),
        vertices=None,
        faces=None,
        previous_valid_m3t_pose=None,
        tracking_health_params=None,
        latest_m3t_pose=None,
        session_initial_pose_stamp_ns=0,
        health_cache=None,
        last_pose_color_stamp_ns=0,
        latest_buffer=mock.Mock(),
        yolo_history=deque(maxlen=60),
        last_color_stamp_ns=None,
        camera_time_reversed=False,
        bridge=mock.Mock(
            imgmsg_to_cv2=mock.Mock(return_value=np.zeros((4, 4, 3), np.uint8))
        ),
        _empty_health=module.TrackingSupervisorNode._empty_health,
    )
    fake.latest_buffer.get_pair_at.return_value = make_pair(color_stamp_ns=10)
    for name in HELPER_METHODS:
        setattr(
            fake,
            name,
            MethodType(getattr(module.TrackingSupervisorNode, name), fake),
        )
    return fake


def make_pair(color_stamp_ns, depth_stamp_ns=None):
    return SimpleNamespace(
        color_stamp_ns=color_stamp_ns,
        depth_stamp_ns=color_stamp_ns if depth_stamp_ns is None else depth_stamp_ns,
        color=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.zeros((4, 4), dtype=np.uint16),
    )


def make_health_inputs(pose_stamp_ns=10, now_ns=10, yolo=None, yolo_history=None):
    return {
        "now_ns": now_ns,
        "m3t_pose": {"stamp_ns": pose_stamp_ns, "matrix": np.eye(4)},
        "camera_info": object(),
        "yolo": yolo,
        "yolo_history": [] if yolo_history is None else yolo_history,
    }


def make_stamped_msg(stamp_ns):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
            ),
            frame_id="camera_color_optical_frame",
        )
    )


def evaluated_result(**changes):
    values = dict(health_evaluated=True, tracking_ok=True)
    values.update(changes)
    return SimpleNamespace(**values)


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
            health, available = fake._m3t_health(make_health_inputs(), None)

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        self.assertFalse(health.tracking_ok)
        self.assertIn("validation error", health.reason)

    def test_m3t_health_evaluated_reports_pose_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        evaluated = evaluated_result()
        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated
        ), mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = fake._m3t_health(make_health_inputs(), None)

        self.assertIs(health, evaluated)
        self.assertTrue(available)

    def test_m3t_health_missing_rgbd_reports_pose_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.latest_buffer.get_pair_at.return_value = None

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health"
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = fake._m3t_health(make_health_inputs(), None)

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        evaluate.assert_not_called()

    def test_m3t_health_nonfinite_pose_reports_pose_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        inputs = make_health_inputs()
        nan_pose = np.eye(4)
        nan_pose[0, 3] = float("nan")
        inputs["m3t_pose"]["matrix"] = nan_pose

        with mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = fake._m3t_health(inputs, None)

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        self.assertFalse(health.depth_evaluable)

    def test_health_uses_pose_frame_not_latest_frame(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        historical = make_pair(color_stamp_ns=10)
        fake.latest_buffer.get_pair_at.return_value = historical

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(make_health_inputs(pose_stamp_ns=10), None)

        fake.latest_buffer.get_pair_at.assert_called_once_with(10, 15_000_000)
        self.assertIs(evaluate.call_args.kwargs["color_bgr"], historical.color)

    def test_missing_pose_pair_does_not_fall_back_to_latest(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.latest_buffer.get_pair_at.return_value = None
        # The latest-pair accessor exists and returns data. It must not be used.
        fake.latest_buffer.get_latest_pair.return_value = make_pair(color_stamp_ns=10)

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health"
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = fake._m3t_health(make_health_inputs(), None)

        self.assertFalse(available)
        self.assertFalse(health.health_evaluated)
        fake.latest_buffer.get_latest_pair.assert_not_called()
        evaluate.assert_not_called()

    def test_repeated_pose_stamp_renders_once(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            for now_ns in (10, 20, 30):
                fake._m3t_health(make_health_inputs(pose_stamp_ns=10, now_ns=now_ns), None)
            fake._m3t_health(make_health_inputs(pose_stamp_ns=40, now_ns=50), None)

        self.assertEqual(evaluate.call_count, 2)

    def test_pose_going_stale_on_a_later_tick_is_not_available(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ), mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            _, fresh = fake._m3t_health(make_health_inputs(pose_stamp_ns=0, now_ns=100_000_000), None)
            _, stale = fake._m3t_health(make_health_inputs(pose_stamp_ns=0, now_ns=400_000_000), None)

        self.assertTrue(fresh)
        self.assertFalse(stale)

    def test_pose_predating_current_session_is_rejected(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.session_initial_pose_stamp_ns = 500

        with mock.patch.object(module, "evaluate_m3t_tracking_health") as evaluate:
            health, available = fake._m3t_health(
                make_health_inputs(pose_stamp_ns=499, now_ns=499), None
            )

        self.assertFalse(available)
        self.assertIn("predates", health.reason)
        evaluate.assert_not_called()

    def test_yolo_inside_frame_tolerance_is_used_for_projection(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        bbox = (10, 10, 50, 50)
        yolo = {"stamp_ns": 10 + 79_000_000, "selected": None}

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(make_health_inputs(yolo=yolo), bbox)

        self.assertEqual(evaluate.call_args.kwargs["yolo_bbox"], bbox)

    def test_yolo_outside_frame_tolerance_is_not_used_for_projection(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        bbox = (10, 10, 50, 50)
        yolo = {"stamp_ns": 10 + 81_000_000, "selected": None}

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(make_health_inputs(yolo=yolo), bbox)

        self.assertIsNone(evaluate.call_args.kwargs["yolo_bbox"])

    def test_yolo_observation_valid_depends_on_freshness_not_pose_frame(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        # The pose frame is 300 ms away from the YOLO stamp. Far beyond the 80 ms
        # projection tolerance, but a fresh YOLO message is still valid evidence.
        fake.last_pose_color_stamp_ns = 1_000_000_000
        now_ns = 1_300_000_000
        fresh_unmatched_pose = {"now_ns": now_ns, "yolo": {"stamp_ns": 1_000_000_000}}

        self.assertTrue(fake._yolo_observation_valid(fresh_unmatched_pose))
        self.assertTrue(
            fake._yolo_observation_valid({"now_ns": now_ns, "yolo": {"stamp_ns": now_ns}})
        )
        self.assertTrue(
            fake._yolo_observation_valid(
                {"now_ns": 1_000_000_000 + 500_000_000, "yolo": {"stamp_ns": 1_000_000_000}}
            )
        )

    def test_yolo_observation_valid_rejects_stale_future_and_missing(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.last_pose_color_stamp_ns = 1_000_000_000

        stale = {"now_ns": 1_500_000_001, "yolo": {"stamp_ns": 1_000_000_000}}
        future = {"now_ns": 1_000_000_000, "yolo": {"stamp_ns": 1_100_000_000}}

        self.assertFalse(fake._yolo_observation_valid(stale))
        self.assertFalse(fake._yolo_observation_valid(future))
        self.assertFalse(fake._yolo_observation_valid({"now_ns": 1, "yolo": None}))

    def test_m3t_health_projection_still_uses_80ms_tolerance_with_history(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        bbox = [10.0, 10.0, 50.0, 50.0]
        history = [
            {"stamp_ns": 10 + 300 * 1_000_000, "selected": {"bbox_xyxy": [1.0, 1.0, 2.0, 2.0]}},
            {"stamp_ns": 10 + 79 * 1_000_000, "selected": {"bbox_xyxy": bbox}},
        ]
        inputs = make_health_inputs(
            yolo={"stamp_ns": 10 + 300 * 1_000_000, "selected": None},
            yolo_history=history,
        )

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(inputs, None)

        self.assertEqual(evaluate.call_args.kwargs["yolo_bbox"], bbox)

    def test_m3t_health_history_entry_outside_tolerance_is_not_projected(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        history = [
            {"stamp_ns": 10 + 81 * 1_000_000, "selected": {"bbox_xyxy": [1.0, 1.0, 2.0, 2.0]}},
        ]

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(make_health_inputs(yolo_history=history), [10, 10, 50, 50])

        self.assertIsNone(evaluate.call_args.kwargs["yolo_bbox"])

    def test_matching_yolo_picks_closest_entry_within_tolerance(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        far = {"stamp_ns": 10 + 60 * 1_000_000, "selected": {"bbox_xyxy": [0.0, 0.0, 1.0, 1.0]}}
        near = {"stamp_ns": 10 + 5 * 1_000_000, "selected": {"bbox_xyxy": [2.0, 2.0, 3.0, 3.0]}}
        inputs = {"yolo_history": [far, near], "yolo": near}

        found, bbox = fake._matching_yolo_for_frame(inputs, 10)

        self.assertTrue(found)
        self.assertEqual(bbox, [2.0, 2.0, 3.0, 3.0])

    def test_matching_yolo_without_detection_is_found_with_no_box(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        empty = {"stamp_ns": 10, "selected": None}

        found, bbox = fake._matching_yolo_for_frame({"yolo_history": [empty]}, 10)

        self.assertTrue(found)
        self.assertIsNone(bbox)

    def test_matching_yolo_reports_not_found_when_history_is_outside_tolerance(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        late = {"stamp_ns": 10 + 200 * 1_000_000, "selected": None}

        self.assertEqual(
            fake._matching_yolo_for_frame({"yolo_history": [late]}, 10), (False, None)
        )

    def test_missing_pair_does_not_poison_health_cache(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.latest_buffer.get_pair_at.return_value = None

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            health, available = fake._m3t_health(make_health_inputs(pose_stamp_ns=10), None)
            self.assertFalse(available)
            self.assertFalse(health.health_evaluated)
            self.assertIsNone(fake.health_cache)

            fake.latest_buffer.get_pair_at.return_value = make_pair(color_stamp_ns=10)
            health, available = fake._m3t_health(make_health_inputs(pose_stamp_ns=10), None)

        self.assertTrue(available)
        self.assertTrue(health.health_evaluated)
        self.assertEqual(evaluate.call_count, 1)

    def test_late_matching_yolo_triggers_reevaluation_of_cached_pose(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        bbox = [10.0, 10.0, 50.0, 50.0]
        match = {"stamp_ns": 10, "selected": {"bbox_xyxy": bbox}}

        with mock.patch.object(
            module, "evaluate_m3t_tracking_health", return_value=evaluated_result()
        ) as evaluate, mock.patch.object(module, "camera_info_to_matrix", return_value=np.eye(3)):
            fake._m3t_health(make_health_inputs(pose_stamp_ns=10, now_ns=10), None)
            self.assertFalse(fake.health_cache["yolo_matched"])
            self.assertEqual(evaluate.call_count, 1)

            fake._m3t_health(
                make_health_inputs(pose_stamp_ns=10, now_ns=20, yolo_history=[match]), None
            )
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args.kwargs["yolo_bbox"], bbox)
            self.assertTrue(fake.health_cache["yolo_matched"])

            fake._m3t_health(
                make_health_inputs(pose_stamp_ns=10, now_ns=30, yolo_history=[match]), None
            )
        self.assertEqual(evaluate.call_count, 2)

    def test_camera_time_reversal_clears_buffer_and_accepts_new_frame(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.latest_buffer = LatestRgbdBuffer(color_history_size=10, depth_history_size=10)
        depth = np.zeros((4, 4), dtype=np.uint16)
        fake.latest_buffer.put_depth(1_000_000_000, depth)
        fake._color_cb(make_stamped_msg(1_000_000_000))
        fake.yolo_history.append({"stamp_ns": 1_000_000_000, "selected": None})

        fake._color_cb(make_stamped_msg(500_000_000))

        self.assertTrue(fake.camera_time_reversed)
        self.assertEqual(len(fake.yolo_history), 0)
        self.assertIsNone(fake.latest_buffer.get_latest_pair(500_000_000, 15_000_000, 250_000_000))

        fake.latest_buffer.put_depth(500_000_000, depth)
        pair = fake.latest_buffer.get_latest_pair(500_000_000, 15_000_000, 250_000_000)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.color_stamp_ns, 500_000_000)
        self.assertEqual(fake.last_color_stamp_ns, 500_000_000)

    def test_forward_color_stamps_do_not_flag_reversal(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake._color_cb(make_stamped_msg(100))
        fake._color_cb(make_stamped_msg(100))
        fake._color_cb(make_stamped_msg(200))

        self.assertFalse(fake.camera_time_reversed)
        fake.latest_buffer.put_color.assert_called_with(200, mock.ANY)


class TrackingSupervisorNodeWiringTest(unittest.TestCase):
    def test_subscription_depths_and_callback_groups(self):
        import rclpy
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

        module = load_tracking_supervisor_node()
        rclpy.init()
        try:
            node = module.TrackingSupervisorNode()
            try:
                subs = {sub.callback: sub for sub in node.subscriptions}
                depths = {
                    name: subs[getattr(node, name)].qos_profile.depth
                    for name in (
                        "_color_cb",
                        "_depth_cb",
                        "_m3t_status_cb",
                        "_m3t_pose_cb",
                    )
                }

                self.assertEqual(depths["_m3t_pose_cb"], 1)
                self.assertEqual(depths["_color_cb"], 2)
                self.assertEqual(depths["_depth_cb"], 2)
                self.assertEqual(depths["_m3t_status_cb"], 20)
                for sub in node.subscriptions:
                    self.assertIsInstance(sub.callback_group, MutuallyExclusiveCallbackGroup)
                    self.assertIsNot(sub.callback_group, node.timer.callback_group)
                self.assertIsInstance(node.timer.callback_group, MutuallyExclusiveCallbackGroup)
            finally:
                node.destroy_node()
        finally:
            rclpy.shutdown()


class MultiTargetReferenceMatchingTest(unittest.TestCase):
    def test_select_detection_for_reference_keeps_tracked_target_when_neighbor_conf_higher(self):
        module = load_tracking_supervisor_node()
        top = {"class_id": 0, "confidence": 0.88, "bbox_xyxy": [220.0, 80.0, 274.0, 156.0]}
        left = {"class_id": 0, "confidence": 0.95, "bbox_xyxy": [113.0, 131.0, 202.0, 206.0]}

        matched = module.select_detection_for_reference(
            [left, top], reference_bbox=[220.0, 80.0, 274.0, 156.0]
        )

        self.assertEqual(matched, top)

    def test_select_detection_for_reference_returns_none_when_tracked_target_removed(self):
        module = load_tracking_supervisor_node()
        top = {"class_id": 0, "confidence": 0.89, "bbox_xyxy": [220.0, 80.0, 274.0, 156.0]}
        # Neighbor box matches real 22-45/23-02 geometry (~103 px center separation, IoU=0).
        matched = module.select_detection_for_reference(
            [top], reference_bbox=[113.0, 131.0, 202.0, 206.0]
        )

        self.assertIsNone(matched)

    def test_matching_yolo_for_frame_prefers_tracked_target_over_best_selected(self):
        module = load_tracking_supervisor_node()
        fake = make_fake_node(module)
        fake.lifecycle = SimpleNamespace(state=module.TRACKING, _awaiting_disappearance=False)
        fake.tracked_target_bbox = [220.0, 80.0, 274.0, 156.0]
        top = {"class_id": 0, "confidence": 0.89, "bbox_xyxy": [221.0, 81.0, 275.0, 157.0]}
        left = {"class_id": 0, "confidence": 0.92, "bbox_xyxy": [113.0, 131.0, 202.0, 206.0]}
        entry = {"stamp_ns": 10, "detections": [left, top], "selected": left}

        found, bbox = fake._matching_yolo_for_frame({"yolo_history": [entry]}, 10)

        self.assertTrue(found)
        self.assertEqual(bbox, top["bbox_xyxy"])


if __name__ == "__main__":
    unittest.main()
