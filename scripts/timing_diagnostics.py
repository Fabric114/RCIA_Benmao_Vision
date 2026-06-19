def make_timing_event(event, frame_stamp_ns, node_now_ns, started_ns=None, extra=None):
    frame_stamp_ns = int(frame_stamp_ns)
    node_now_ns = int(node_now_ns)
    payload = {
        "event": str(event),
        "frame_stamp_ns": frame_stamp_ns,
        "node_now_ns": node_now_ns,
        "age_ms": (node_now_ns - frame_stamp_ns) / 1_000_000.0,
    }
    if started_ns is not None:
        payload["duration_ms"] = (node_now_ns - int(started_ns)) / 1_000_000.0
    if extra:
        payload.update(extra)
    return payload
