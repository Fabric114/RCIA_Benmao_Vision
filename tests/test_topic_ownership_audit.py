import unittest

from scripts.topic_ownership_audit import audit_launch_text


class TopicOwnershipAuditTest(unittest.TestCase):
    def test_audit_accepts_supervisor_controlled_defaults(self):
        text = """
        executable="tracking_supervisor_node.py"
        "-p direct_initial_pose_enabled:=false"
        "-p request_required:=true"
        """

        result = audit_launch_text(text)

        self.assertTrue(result["success"], result["errors"])

    def test_audit_rejects_direct_foundationpose_m3t_path(self):
        text = """
        executable="tracking_supervisor_node.py"
        "-p direct_initial_pose_enabled:=true"
        "-p request_required:=true"
        """

        result = audit_launch_text(text)

        self.assertFalse(result["success"])
        self.assertTrue(any("direct_initial_pose_enabled" in error for error in result["errors"]))

    def test_audit_rejects_legacy_supervisors(self):
        text = """
        executable="tracking_supervisor_node.py"
        executable="tracking_watchdog_node.py"
        "-p direct_initial_pose_enabled:=false"
        "-p request_required:=true"
        """

        result = audit_launch_text(text)

        self.assertFalse(result["success"])
        self.assertTrue(any("tracking_watchdog_node.py" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
