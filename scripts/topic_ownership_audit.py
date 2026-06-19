#!/usr/bin/env python3
"""Audit launch defaults for supervisor-owned tracking lifecycle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


FORBIDDEN_TOKENS = [
    "enable_tracking_supervisor",
    "enable_tracking_watchdog",
    "enable_handoff_supervisor",
    "handoff_supervisor_node.py",
    "tracking_watchdog_node.py",
    "m3t_reset_pose_topic",
]

REQUIRED_TOKENS = [
    "-p direct_initial_pose_enabled:=false",
    "-p request_required:=true",
    "tracking_supervisor_node.py",
]


def _find_launch_default(text: str, name: str) -> str | None:
    pattern = re.compile(
        r"DeclareLaunchArgument\(\s*['\"]"
        + re.escape(name)
        + r"['\"]\s*,\s*default_value\s*=\s*['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def audit_launch_text(text: str) -> Dict[str, object]:
    errors: List[str] = []
    defaults: Dict[str, str | None] = {}

    for token in FORBIDDEN_TOKENS:
        if token in text:
            errors.append(f"forbidden legacy launch token: {token}")

    for token in REQUIRED_TOKENS:
        if token not in text:
            errors.append(f"missing required launch token: {token}")

    return {
        "success": not errors,
        "errors": errors,
        "defaults": defaults,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify full_energy_tracking.launch.py keeps supervisor-controlled defaults."
    )
    parser.add_argument(
        "launch_file",
        nargs="?",
        default=(
            Path(__file__).resolve().parents[1]
            / "ros2/rcia_energy_trace_ros/launch/full_energy_tracking.launch.py"
        ),
    )
    args = parser.parse_args()

    launch_file = Path(args.launch_file)
    result = audit_launch_text(launch_file.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
