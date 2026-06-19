import json


def detections_to_json(stamp_ns, detections):
    payload = {
        "stamp_ns": int(stamp_ns),
        "detections": [_normalize_detection(det) for det in detections],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_detection(detection):
    if isinstance(detection, dict):
        bbox = detection.get("bbox_xyxy", detection.get("bbox", [0, 0, 0, 0]))
        confidence = detection.get("confidence", detection.get("conf", 0.0))
        class_id = detection.get("class_id", detection.get("cls", -1))
    else:
        bbox = detection[:4]
        confidence = detection[4] if len(detection) > 4 else 0.0
        class_id = detection[5] if len(detection) > 5 else -1
    if class_id is None:
        class_id = -1
    return {
        "class_id": int(class_id),
        "confidence": float(confidence),
        "bbox_xyxy": [float(v) for v in bbox[:4]],
    }
