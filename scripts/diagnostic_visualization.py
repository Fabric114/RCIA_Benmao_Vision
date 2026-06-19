from pathlib import Path

import cv2
import numpy as np


def _fit_panel(image_bgr, panel_size):
    width, height = panel_size
    if image_bgr is None:
        return make_placeholder_panel("Waiting", panel_size=panel_size)
    image = image_bgr
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    fitted = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(image.shape[1] * scale))
    resized_height = max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    fitted[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return fitted


def _draw_panel_title(image_bgr, title):
    rendered = image_bgr.copy()
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        rendered,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def make_placeholder_panel(label, panel_size=(640, 480)):
    width, height = panel_size
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        rendered,
        label,
        (24, max(36, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return rendered


def compose_pipeline_view(
    yolo=None,
    sam=None,
    foundationpose=None,
    m3t=None,
    panel_size=(640, 480),
    tracking_status=None,
    panel_statuses=None,
):
    panel_statuses = panel_statuses or {}
    panels = {
        "yolo": _draw_panel_title(_fit_panel(yolo, panel_size), "YOLO live monitor"),
        "sam": _draw_panel_title(_fit_panel(sam, panel_size), "SAM last mask"),
        "foundationpose": _draw_panel_title(
            _fit_panel(foundationpose, panel_size),
            "FoundationPose last init",
        ),
        "m3t": _draw_panel_title(_fit_panel(m3t, panel_size), "M3T live tracking"),
    }
    for key, status in panel_statuses.items():
        if key not in panels or not isinstance(status, dict):
            continue
        label = status.get("label", "")
        if label:
            panels[key] = draw_panel_status(
                panels[key],
                str(label),
                stale=bool(status.get("stale", False)),
            )
    top = np.concatenate([panels["yolo"], panels["sam"]], axis=1)
    bottom = np.concatenate([panels["foundationpose"], panels["m3t"]], axis=1)
    view = np.concatenate([top, bottom], axis=0)
    if tracking_status is not None:
        view = draw_tracking_status_overlay(view, tracking_status)
    return view


def draw_panel_status(image_bgr, label, stale=False):
    rendered = image_bgr.copy()
    color = (0, 0, 220) if stale else (0, 140, 255)
    cv2.rectangle(rendered, (0, 30), (rendered.shape[1], 56), (0, 0, 0), -1)
    cv2.rectangle(rendered, (0, 30), (8, 56), color, -1)
    cv2.putText(
        rendered,
        label[:48],
        (14, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def draw_tracking_status_overlay(image_bgr, status):
    rendered = image_bgr.copy()
    height, width = rendered.shape[:2]
    bar_height = min(34, max(24, height // 12))
    trusted = bool(status.get("trusted", False))
    state = str(status.get("state", "UNKNOWN"))
    action = str(status.get("action", "none"))
    m3t_status = status.get("m3t", {}) if isinstance(status.get("m3t", {}), dict) else {}
    candidate_status = (
        status.get("candidate", {}) if isinstance(status.get("candidate", {}), dict) else {}
    )
    yolo_visible = bool(status.get("yolo_visible", m3t_status.get("yolo_visible", False)))
    depth_ratio = status.get(
        "depth_consistency_ratio",
        m3t_status.get("depth_consistency_ratio", None),
    )
    if depth_ratio is None:
        depth_text = "depth=n/a"
    else:
        depth_text = f"depth={float(depth_ratio):.2f}"
    reason = str(
        status.get(
            "reason",
            m3t_status.get("reason", candidate_status.get("reason", "")),
        )
    )
    color = (0, 160, 0) if trusted else (0, 140, 255)
    if state in {"TARGET_OUT_OF_VIEW", "LOST_TARGET_IN_VIEW", "STOPPING_M3T", "FAILED_COOLDOWN"}:
        color = (0, 0, 220)
    elif state in {"REQUESTING_CANDIDATE", "VALIDATING_CANDIDATE", "STARTING_M3T"}:
        color = (255, 120, 0)
    cv2.rectangle(rendered, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.rectangle(rendered, (0, height - bar_height), (8, height), color, -1)
    text = (
        f"supervisor: {state} | trusted={trusted} | yolo={yolo_visible} | "
        f"{depth_text} | action={action}"
    )
    if reason:
        text = f"{text} | {reason[:52]}"
    cv2.putText(
        rendered,
        text,
        (14, height - max(8, bar_height // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def draw_yolo_detections(image_bgr, detections, selected=None):
    rendered = image_bgr.copy()
    selected_bbox = tuple(selected[:4]) if selected is not None else None
    for x1, y1, x2, y2, conf, cls_id in detections:
        bbox = (int(x1), int(y1), int(x2), int(y2))
        color = (0, 255, 255) if bbox == selected_bbox else (0, 255, 0)
        thickness = 3 if bbox == selected_bbox else 2
        cv2.rectangle(rendered, bbox[:2], bbox[2:], color, thickness)
        label = f"cls={cls_id} conf={conf:.2f}"
        cv2.putText(
            rendered,
            label,
            (bbox[0], max(bbox[1] - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def overlay_mask(image_bgr, mask, color=(0, 255, 255), alpha=0.35):
    rendered = image_bgr.copy()
    if mask is None:
        return rendered
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return rendered
    overlay = np.zeros_like(rendered)
    overlay[mask_bool] = np.array(color, dtype=np.uint8)
    rendered[mask_bool] = cv2.addWeighted(
        rendered[mask_bool], 1.0 - alpha, overlay[mask_bool], alpha, 0.0
    )
    return rendered


def save_debug_image(path, image_bgr):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
