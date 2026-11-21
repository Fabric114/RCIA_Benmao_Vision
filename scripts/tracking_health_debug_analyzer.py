from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingHealthDebugResult:
    should_emit: bool
    primary_cause: str
    summary: str
    evidence: tuple

    def to_dict(self):
        return {
            "should_emit": bool(self.should_emit),
            "primary_cause": self.primary_cause,
            "summary": self.summary,
            "evidence": list(self.evidence),
        }


def analyze_tracking_health(payload, previous_payload=None):
    tracking_filter = _dict(payload.get("tracking_filter"))
    m3t = _dict(payload.get("m3t"))
    previous_filter = _dict((previous_payload or {}).get("tracking_filter"))

    state = str(tracking_filter.get("reliability_state", "UNKNOWN"))
    previous_state = str(previous_filter.get("reliability_state", ""))
    action = str(payload.get("action", ""))
    request_reinit = bool(tracking_filter.get("request_reinit", False))
    stop_m3t = bool(tracking_filter.get("stop_m3t", False))
    state_changed = bool(previous_state and previous_state != state)

    cause = _primary_cause(tracking_filter, m3t, action)
    evidence = _evidence_lines(tracking_filter, m3t)
    should_emit = (
        request_reinit
        or stop_m3t
        or state_changed
        or state in {"REINIT_NEEDED", "LOST"}
        or action in {"stop_ack_timeout", "wait_for_target_disappearance"}
    )

    if state_changed:
        state_text = f"{previous_state} -> {state}"
    else:
        state_text = state

    summary_bits = [f"state={state_text}", f"action={action}", f"cause={cause}"]
    if request_reinit:
        summary_bits.append("request_reinit=true")
    if stop_m3t:
        summary_bits.append("stop_m3t=true")
    reason = str(tracking_filter.get("reason", ""))
    if reason:
        summary_bits.append(f"reason={reason}")

    return TrackingHealthDebugResult(
        should_emit=should_emit,
        primary_cause=cause,
        summary=" | ".join(summary_bits),
        evidence=tuple(evidence),
    )


def _primary_cause(tracking_filter, m3t, action):
    state = str(tracking_filter.get("reliability_state", ""))
    reason = str(tracking_filter.get("reason", "")).lower()
    external_conflict = int(tracking_filter.get("external_conflict_streak", 0) or 0)
    out_of_view = int(tracking_filter.get("out_of_view_streak", 0) or 0)
    weak_validation = int(tracking_filter.get("weak_validation_fail_streak", 0) or 0)
    m3t_bad = int(tracking_filter.get("m3t_bad_streak", 0) or 0)
    m3t_good = int(tracking_filter.get("m3t_good_streak", 0) or 0)

    if (
        action == "stop_ack_timeout"
        or "stop ack timeout" in reason
        or "stop_ack_timeout" in reason
    ):
        return "stop_ack_timeout"
    if "startup grace expired" in reason or "startup_timeout" in reason:
        return "startup_timeout"
    if "observation_unavailable" in reason:
        return "observation_unavailable"
    if "support_timeout" in reason:
        return "support_timeout"
    if (
        "out_of_view" in reason
        or "out of view" in reason
        or "target out" in reason
        or (state == "LOST" and out_of_view > 0)
    ):
        return "out_of_view"
    if state == "REINIT_NEEDED" or "external_conflict" in reason or "external conflict" in reason:
        if external_conflict == 0 and m3t_bad > 0 and "external" not in reason:
            return "m3t_bad"
        return "external_conflict"
    if weak_validation > 0 and ("weak" in reason or state == "M3T_SUSPECT"):
        return "weak_validation"
    if m3t_bad > 0 and m3t_bad >= max(m3t_good, external_conflict):
        return "m3t_bad"
    if out_of_view > 0 or state == "LOST":
        return "out_of_view"
    if m3t_good > 0 and action == "publish_trusted_pose":
        return "healthy"
    if _m3t_reason_mentions_projection(m3t):
        return "projection"
    return "unknown"


def _evidence_lines(tracking_filter, m3t):
    lines = [
        "streaks: "
        f"good={int(tracking_filter.get('m3t_good_streak', 0) or 0)}, "
        f"startup_good_total={int(tracking_filter.get('startup_good_total', 0) or 0)}, "
        f"bad={int(tracking_filter.get('m3t_bad_streak', 0) or 0)}, "
        f"external_conflict={int(tracking_filter.get('external_conflict_streak', 0) or 0)}, "
        f"out_of_view={int(tracking_filter.get('out_of_view_streak', 0) or 0)}, "
        f"weak_validation={int(tracking_filter.get('weak_validation_fail_streak', 0) or 0)}",
        "filter: "
        f"recovery_streak={int(tracking_filter.get('recovery_streak', 0) or 0)}, "
        f"suspect_elapsed_ms={float(tracking_filter.get('suspect_elapsed_ms', 0.0) or 0.0):.1f}, "
        f"evidence_kind={tracking_filter.get('evidence_kind', 'unknown')}, "
        f"observation_is_new={bool(tracking_filter.get('observation_is_new', False))}",
        "m3t: "
        f"depth={float(m3t.get('depth_consistency_ratio', m3t.get('depth_ratio', 0.0)) or 0.0):.3f}, "
        f"projection_iou={float(m3t.get('projection_iou', 0.0) or 0.0):.3f}, "
        f"black_white={bool(m3t.get('black_white_ok', m3t.get('black_white_success', False)))}, "
        f"smooth={bool(m3t.get('smoothness_ok', False))}, "
        f"rendered_px={int(m3t.get('rendered_pixel_count', 0) or 0)}, "
        f"health_evaluated={bool(m3t.get('health_evaluated', False))}, "
        f"depth_evaluable={bool(m3t.get('depth_evaluable', False))}, "
        f"valid_depth_overlap_count={int(m3t.get('valid_depth_overlap_count', 0) or 0)}",
    ]
    reason = str(m3t.get("reason", ""))
    if reason:
        lines.append(f"m3t_reason: {reason}")
    filter_reason = str(tracking_filter.get("reason", ""))
    if filter_reason:
        lines.append(f"filter_reason: {filter_reason}")
    return lines


def _m3t_reason_mentions_projection(m3t):
    return "projection" in str(m3t.get("reason", "")).lower()


def _dict(value):
    return value if isinstance(value, dict) else {}
