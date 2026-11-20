import json
import unittest

from scripts.supervisor_messages import (
    candidate_matches_request,
    candidate_status_is_failure,
    m3t_status_belongs_to_session,
    make_init_request_json,
    parse_m3t_status_json,
)


class SupervisorMessageRoutingTest(unittest.TestCase):
    def test_make_init_request_json_contains_request_and_bbox(self):
        payload = json.loads(
            make_init_request_json(
                request_id="req-7",
                stamp_ns=123,
                target_class=0,
                bbox_xyxy=[1, 2, 3, 4],
                confidence=0.91,
                reason="initial",
            )
        )

        self.assertEqual(payload["request_id"], "req-7")
        self.assertEqual(payload["stamp_ns"], 123)
        self.assertEqual(payload["target_class"], 0)
        self.assertEqual(payload["bbox_xyxy"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(payload["confidence"], 0.91)

    def test_parse_m3t_status_json_supplies_defaults(self):
        status = parse_m3t_status_json('{"state":"tracking","tracking_started":true}')

        self.assertEqual(status["state"], "tracking")
        self.assertTrue(status["tracking_started"])
        self.assertFalse(status["have_initial_pose"])
        self.assertEqual(status["stamp_ns"], 0)

    def test_parse_m3t_status_json_reads_session_fields(self):
        status = parse_m3t_status_json(
            json.dumps(
                {
                    "state": "tracking",
                    "initial_pose_stamp_ns": 42,
                    "processed_frame_stamp_ns": 43,
                    "latest_frame_stamp_ns": 44,
                    "replayed_frames": 3,
                }
            )
        )

        self.assertEqual(status["initial_pose_stamp_ns"], 42)
        self.assertEqual(status["processed_frame_stamp_ns"], 43)
        self.assertEqual(status["latest_frame_stamp_ns"], 44)
        self.assertEqual(status["replayed_frames"], 3)

    def test_parse_m3t_status_json_defaults_session_fields_for_old_node(self):
        status = parse_m3t_status_json('{"state":"stopped","stamp_ns":9}')

        self.assertEqual(status["initial_pose_stamp_ns"], 0)
        self.assertEqual(status["processed_frame_stamp_ns"], 0)
        self.assertEqual(status["latest_frame_stamp_ns"], 0)
        self.assertEqual(status["replayed_frames"], 0)

    def test_session_check_rejects_only_explicit_mismatch(self):
        self.assertTrue(m3t_status_belongs_to_session({"initial_pose_stamp_ns": 7}, 0))
        self.assertTrue(m3t_status_belongs_to_session({"initial_pose_stamp_ns": 7}, 7))
        self.assertFalse(m3t_status_belongs_to_session({"initial_pose_stamp_ns": 6}, 7))
        # An old node reports no session stamp, so its status is accepted.
        self.assertTrue(m3t_status_belongs_to_session({"initial_pose_stamp_ns": 0}, 7))

    def test_candidate_matches_request_by_request_id(self):
        request = {"request_id": "req-1", "stamp_ns": 1000}
        status = {"request_id": "req-1", "frame_stamp_ns": 1010}

        self.assertTrue(candidate_matches_request(request, status, max_stamp_delta_ms=50.0))
        self.assertFalse(
            candidate_matches_request(
                request,
                {"request_id": "other", "frame_stamp_ns": 1010},
                max_stamp_delta_ms=50.0,
            )
        )

    def test_candidate_matches_request_by_stamp_when_id_missing(self):
        request = {"request_id": "req-1", "stamp_ns": 1_000_000_000}
        status = {"frame_stamp_ns": 1_020_000_000}

        self.assertTrue(candidate_matches_request(request, status, max_stamp_delta_ms=50.0))
        self.assertFalse(candidate_matches_request(request, status, max_stamp_delta_ms=10.0))

    def test_candidate_status_failure_is_explicit(self):
        self.assertTrue(candidate_status_is_failure({"success": False}))
        self.assertTrue(
            candidate_status_is_failure({"event": "foundationpose_candidate_failed"})
        )
        self.assertFalse(candidate_status_is_failure({"success": True}))
        self.assertFalse(
            candidate_status_is_failure({"event": "foundationpose_candidate_pose_published"})
        )


if __name__ == "__main__":
    unittest.main()
