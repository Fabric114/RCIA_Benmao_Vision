import json


def make_init_request_json(
    request_id,
    stamp_ns,
    target_class,
    bbox_xyxy,
    confidence=1.0,
    reason="initial",
):
    payload = {
        "request_id": str(request_id),
        "stamp_ns": int(stamp_ns),
        "target_class": int(target_class) if target_class is not None else -1,
        "bbox_xyxy": [float(v) for v in bbox_xyxy[:4]],
        "confidence": float(confidence),
        "reason": str(reason),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_m3t_status_json(payload):
    data = json.loads(payload)
    return {
        "state": str(data.get("state", "")),
        "stamp_ns": int(data.get("stamp_ns", 0)),
        "tracking_started": bool(data.get("tracking_started", False)),
        "have_initial_pose": bool(data.get("have_initial_pose", False)),
        "reason": str(data.get("reason", "")),
    }


def candidate_matches_request(request, candidate_status, max_stamp_delta_ms):
    if request is None or candidate_status is None:
        return False
    request_id = request.get("request_id")
    candidate_request_id = candidate_status.get("request_id")
    if request_id and candidate_request_id:
        return str(request_id) == str(candidate_request_id)
    request_stamp_ns = int(request.get("stamp_ns", 0))
    candidate_stamp_ns = int(
        candidate_status.get("frame_stamp_ns", candidate_status.get("stamp_ns", 0))
    )
    if request_stamp_ns <= 0 or candidate_stamp_ns <= 0:
        return False
    delta_ms = abs(request_stamp_ns - candidate_stamp_ns) / 1_000_000.0
    return delta_ms <= float(max_stamp_delta_ms)


def candidate_status_is_failure(candidate_status):
    if not isinstance(candidate_status, dict):
        return False
    if candidate_status.get("success") is False:
        return True
    event = str(candidate_status.get("event", ""))
    return event.endswith("_failed") or "failed" in event.lower()
