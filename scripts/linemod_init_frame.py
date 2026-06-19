#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "runtime/ultralytics"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_depth_validation import median_depth_in_window, validate_depth_support
from scripts.linemod_line2d import load_line2d_templates, match_line2d_templates
from scripts.linemod_pnp_pose import estimate_pose_from_match
from scripts.linemod_pose_utils import replace_pose_translation_from_depth
from scripts.linemod_silhouette_renderer import load_obj_mesh, render_model_buffers
from scripts.linemod_template_store import find_template_metadata, load_template_metadata
from scripts.linemod_validator import (
    validate_black_white_appearance,
    validate_rendered_depth_consistency,
)


def load_camera_matrix(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"^k:\s*(\[[^\]]+\])", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"camera_info.yaml does not contain k: {path}")
    values = ast.literal_eval(match.group(1))
    if len(values) != 9:
        raise ValueError("camera_info k field must contain 9 values")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _candidate_to_dict(
    match,
    template,
    pose_source,
    pose,
    pnp_result,
    depth_result,
    rendered_depth_result=None,
    black_white_result=None,
):
    accepted = bool(depth_result.success) if depth_result is not None else False
    if rendered_depth_result is not None:
        accepted = accepted and bool(rendered_depth_result.success)
    if black_white_result is not None:
        accepted = accepted and bool(black_white_result.success)
    reject_reasons = []
    if not pnp_result.success and pose is None:
        reject_reasons.append(pnp_result.reason)
    if depth_result is None:
        reject_reasons.append("No pose available for depth validation")
    elif not depth_result.success:
        reject_reasons.append(depth_result.reason)
    if rendered_depth_result is not None and not rendered_depth_result.success:
        reject_reasons.append(rendered_depth_result.reason)
    if black_white_result is not None and not black_white_result.success:
        reject_reasons.append(black_white_result.reason)
    return {
        "class_id": match.class_id,
        "template_id": int(match.template_id),
        "match": {
            "x": int(match.x),
            "y": int(match.y),
            "similarity": float(match.similarity),
            "line2d_score": None
            if match.line_similarity is None
            else float(match.line_similarity),
            "depth_normal_score": None
            if match.normal_similarity is None
            else float(match.normal_similarity),
            "template_width": int(template.width),
            "template_height": int(template.height),
        },
        "pose_source": pose_source,
        "pose": None if pose is None else np.asarray(pose, dtype=np.float64).tolist(),
        "pnp": {
            "success": bool(pnp_result.success),
            "inlier_count": int(pnp_result.inlier_count),
            "mean_reprojection_error_px": float(pnp_result.mean_reprojection_error_px),
            "reason": pnp_result.reason,
        },
        "depth_validation": depth_result.to_dict() if depth_result is not None else None,
        "rendered_depth_validation": (
            rendered_depth_result.to_dict() if rendered_depth_result is not None else None
        ),
        "black_white_appearance": (
            black_white_result.to_dict() if black_white_result is not None else None
        ),
        "reject_reason": "; ".join(reason for reason in reject_reasons if reason),
        "accepted": accepted,
    }


def _fallback_pose_from_depth(match, template, metadata_item, depth_image, k_matrix, depth_scale):
    center_u = float(match.x) + float(template.width) / 2.0
    center_v = float(match.y) + float(template.height) / 2.0
    depth_m = median_depth_in_window(
        depth_image,
        center_u,
        center_v,
        depth_scale=depth_scale,
        radius=5,
    )


def load_roi_boxes(path, class_name=None, min_confidence=0.0):
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = (
            payload.get("detections")
            or payload.get("boxes")
            or payload.get("rois")
            or payload.get("objects")
            or []
        )
    else:
        items = payload

    boxes = []
    for item in items:
        if isinstance(item, dict):
            if class_name is not None:
                item_class = item.get("class_name", item.get("label", item.get("class_id")))
                if str(item_class) != str(class_name):
                    continue
            confidence = float(item.get("confidence", item.get("score", 1.0)))
            if confidence < float(min_confidence):
                continue
            if "bbox_xyxy" in item:
                box = item["bbox_xyxy"]
            elif "xyxy" in item:
                box = item["xyxy"]
            elif "bbox" in item:
                box = item["bbox"]
            elif "bbox_xywh" in item:
                x, y, width, height = item["bbox_xywh"]
                box = [x, y, x + width, y + height]
            else:
                continue
        else:
            box = item
        if len(box) != 4:
            continue
        boxes.append(tuple(float(value) for value in box))
    return boxes


def load_yolo_roi_boxes(color_bgr, yolo_model, target_class=None, confidence=0.7):
    if not yolo_model:
        return []
    from ultralytics import YOLO

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    results = YOLO(str(yolo_model)).predict(
        color_rgb,
        conf=float(confidence),
        verbose=False,
    )
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0]) if box.cls is not None else None
        if target_class is not None and class_id != int(target_class):
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def _match_bbox_mask(match, template, image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, int(match.x))
    y1 = max(0, int(match.y))
    x2 = min(width, int(match.x) + int(template.width))
    y2 = min(height, int(match.y) + int(template.height))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _appearance_mask_for_candidate(
    match,
    template,
    pose,
    color_shape,
    k_matrix,
    mesh_vertices=None,
    mesh_faces=None,
):
    if mesh_vertices is not None and mesh_faces is not None and pose is not None:
        _, rendered_mask, _, _, _ = render_model_buffers(
            mesh_vertices,
            mesh_faces,
            pose,
            k_matrix,
            image_size=(color_shape[1], color_shape[0]),
        )
        if int(np.count_nonzero(rendered_mask)) > 0:
            return rendered_mask
    return _match_bbox_mask(match, template, color_shape)


def _write_roi_overlay(path, color, roi_boxes):
    overlay = color.copy()
    for x1, y1, x2, y2 in roi_boxes:
        cv2.rectangle(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 255, 255),
            2,
        )
    cv2.imwrite(str(path), overlay)


def _write_topk_overlay(path, color, candidates):
    overlay = color.copy()
    for index, candidate in enumerate(candidates[:10]):
        match = candidate["match"]
        accepted = bool(candidate["accepted"])
        color_bgr = (0, 255, 0) if accepted else (0, 0, 255)
        x1 = int(match["x"])
        y1 = int(match["y"])
        x2 = x1 + int(match["template_width"])
        y2 = y1 + int(match["template_height"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            overlay,
            f"{index}:{match['similarity']:.1f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), overlay)


def _write_projected_pose_overlay(path, color, mask):
    overlay = color.copy()
    if mask is not None and int(np.count_nonzero(mask)) > 0:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(path), overlay)


def _write_template_feature_overlay(path, color, match, template):
    overlay = color.copy()
    if match is not None and template is not None:
        for feature in template.features:
            cv2.circle(
                overlay,
                (int(match.x) + int(feature.x), int(match.y) + int(feature.y)),
                2,
                (0, 255, 255),
                -1,
            )
        for feature in template.normal_features:
            cv2.circle(
                overlay,
                (int(match.x) + int(feature.x), int(match.y) + int(feature.y)),
                2,
                (255, 0, 255),
                -1,
            )
    cv2.imwrite(str(path), overlay)


def _write_rendered_vs_observed_depth(
    path,
    color_bgr,
    rendered_depth_result,
    rendered_depth,
    rendered_mask,
    observed_depth_m,
    max_depth_residual_m=0.02,
):
    """
    Generate depth consistency heatmap overlay:
    - Green: consistent (residual <= threshold)
    - Red: inconsistent (residual > threshold)
    - Blue: only rendered depth (possible occlusion)
    - Gray: only observed depth or invalid
    """
    height, width = color_bgr.shape[:2]
    overlay = (color_bgr.astype(np.float32) * 0.4).astype(np.uint8)

    if rendered_depth_result is None or rendered_depth is None or rendered_mask is None:
        cv2.putText(
            overlay,
            "No rendered depth data available",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(path), overlay)
        return

    # Create masks for different regions
    rendered_valid = (rendered_mask > 0) & np.isfinite(rendered_depth) & (rendered_depth > 0.0)
    observed_valid = np.isfinite(observed_depth_m) & (observed_depth_m > 0.0)
    overlap = rendered_valid & observed_valid

    # Calculate residuals in overlap region
    residual = np.zeros((height, width), dtype=np.float32)
    residual[overlap] = np.abs(rendered_depth[overlap] - observed_depth_m[overlap])

    consistent = overlap & (residual <= float(max_depth_residual_m))
    inconsistent = overlap & (residual > float(max_depth_residual_m))
    only_rendered = rendered_valid & ~observed_valid

    # Color coding
    overlay[consistent] = [0, 255, 0]  # Green: consistent
    overlay[inconsistent] = [0, 0, 255]  # Red: inconsistent
    overlay[only_rendered] = [255, 128, 0]  # Blue: only rendered

    # Add text summary
    text_lines = [
        f"Consistency: {rendered_depth_result.consistency_ratio:.3f}",
        f"Overlap: {rendered_depth_result.valid_overlap_count}px",
        f"Mean residual: {rendered_depth_result.mean_abs_depth_residual_m:.4f}m",
        f"Green={int(np.count_nonzero(consistent))} Red={int(np.count_nonzero(inconsistent))}",
    ]
    for i, text in enumerate(text_lines):
        cv2.putText(
            overlay,
            text,
            (8, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), overlay)


def _black_white_validation_image(color, mask):
    diagnostic = np.zeros_like(color)
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return diagnostic
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    region = mask > 0
    low_saturation = region & (hsv[:, :, 1] <= 45)
    black = low_saturation & (hsv[:, :, 2] <= 80)
    white = low_saturation & (hsv[:, :, 2] >= 150)
    diagnostic[region] = (40, 40, 40)
    diagnostic[low_saturation] = (160, 160, 160)
    diagnostic[black] = (255, 0, 0)
    diagnostic[white] = (0, 255, 0)
    return diagnostic
    if depth_m is None:
        return None
    return replace_pose_translation_from_depth(
        metadata_item.object_in_camera,
        center_u=center_u,
        center_v=center_v,
        depth_m=depth_m,
        k_matrix=k_matrix,
    )


def run_init_frame(args):
    started_at = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_pose_file = output_dir / "final_pose.txt"
    if final_pose_file.exists():
        final_pose_file.unlink()
    color = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"Could not read color image: {args.color}")
    depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth image: {args.depth}")

    k_matrix = load_camera_matrix(args.camera_info)
    templates = load_line2d_templates(args.templates)
    metadata = load_template_metadata(args.poses)
    template_by_id = {(item.class_id, item.template_id): item for item in templates}
    mesh_file = getattr(args, "mesh_file", None)
    mesh_vertices = None
    mesh_faces = None
    if mesh_file:
        mesh_vertices, mesh_faces = load_obj_mesh(mesh_file)
    roi_boxes = load_roi_boxes(
        getattr(args, "roi_json", None),
        class_name=getattr(args, "roi_class_name", None),
        min_confidence=float(getattr(args, "roi_min_confidence", 0.0)),
    )
    roi_boxes.extend(
        load_yolo_roi_boxes(
            color,
            getattr(args, "yolo_model", None),
            target_class=getattr(args, "yolo_target_class", None),
            confidence=float(getattr(args, "yolo_conf", 0.7)),
        )
    )
    _write_roi_overlay(output_dir / "roi_overlay.png", color, roi_boxes)
    match_started_at = time.perf_counter()
    matches = match_line2d_templates(
        color,
        templates,
        threshold=float(args.threshold),
        stride=int(args.stride),
        depth_image=depth if getattr(args, "enable_depth_normal_score", False) else None,
        k_matrix=k_matrix if getattr(args, "enable_depth_normal_score", False) else None,
        depth_scale=float(args.depth_scale),
        line2d_score_weight=float(getattr(args, "line2d_score_weight", 0.6)),
        normal_score_weight=float(getattr(args, "normal_score_weight", 0.4)),
        roi_boxes=roi_boxes,
        roi_padding_px=int(getattr(args, "roi_padding_px", 0)),
    )
    match_runtime_s = time.perf_counter() - match_started_at

    candidate_scores = []
    accepted = []
    first_appearance_mask = None
    first_match = None
    first_template = None
    first_rendered_depth_result = None
    first_rendered_depth = None
    first_rendered_mask = None
    first_pose = None
    for match in matches[: int(args.top_k)]:
        template = template_by_id[(match.class_id, match.template_id)]
        if first_match is None:
            first_match = match
            first_template = template
        pnp_result = estimate_pose_from_match(match, template, k_matrix)
        pose_source = "pnp"
        pose = pnp_result.object_in_camera
        if not pnp_result.success:
            pose_source = "depth_median_fallback"
            try:
                metadata_item = find_template_metadata(metadata, match.class_id, match.template_id)
                pose = _fallback_pose_from_depth(
                    match,
                    template,
                    metadata_item,
                    depth,
                    k_matrix,
                    depth_scale=float(args.depth_scale),
                )
            except KeyError:
                pose = None

        depth_result = None
        rendered_depth_result = None
        black_white_result = None
        if pose is not None:
            depth_result = validate_depth_support(
                depth,
                match,
                template,
                pose,
                k_matrix=k_matrix,
                depth_scale=float(args.depth_scale),
                max_abs_z_error_m=float(args.max_abs_z_error_m),
                min_valid_depths=int(args.min_valid_depths),
            )
            if mesh_vertices is not None:
                rendered_depth_result = validate_rendered_depth_consistency(
                    mesh_vertices,
                    mesh_faces,
                    pose,
                    k_matrix,
                    depth,
                    image_size=(color.shape[1], color.shape[0]),
                    depth_scale=float(args.depth_scale),
                    max_depth_residual_m=float(getattr(args, "max_depth_residual_m", 0.02)),
                    min_consistency_ratio=float(getattr(args, "min_depth_consistency_ratio", 0.70)),
                    min_overlap_pixels=int(getattr(args, "min_rendered_depth_overlap_pixels", 50)),
                )
                if first_rendered_depth_result is None:
                    first_rendered_depth_result = rendered_depth_result
                    first_pose = pose
                    # Render again to get depth buffers for visualization
                    _, first_rendered_mask, _, first_rendered_depth, _ = render_model_buffers(
                        mesh_vertices,
                        mesh_faces,
                        pose,
                        k_matrix,
                        image_size=(color.shape[1], color.shape[0]),
                    )
            if bool(getattr(args, "enable_black_white_validation", True)):
                appearance_mask = _appearance_mask_for_candidate(
                    match,
                    template,
                    pose,
                    color.shape,
                    k_matrix,
                    mesh_vertices=mesh_vertices,
                    mesh_faces=mesh_faces,
                )
                if first_appearance_mask is None:
                    first_appearance_mask = appearance_mask
                black_white_result = validate_black_white_appearance(
                    color,
                    appearance_mask,
                    max_saturation=float(getattr(args, "max_saturation", 45)),
                    black_v_max=float(getattr(args, "black_v_max", 80)),
                    white_v_min=float(getattr(args, "white_v_min", 150)),
                    min_low_saturation_ratio=float(
                        getattr(args, "min_low_saturation_ratio", 0.65)
                    ),
                    min_black_white_support=float(
                        getattr(args, "min_black_white_support", 0.45)
                    ),
                )
        candidate = _candidate_to_dict(
            match,
            template,
            pose_source,
            pose,
            pnp_result,
            depth_result,
            rendered_depth_result=rendered_depth_result,
            black_white_result=black_white_result,
        )
        candidate_scores.append(candidate)
        if candidate["accepted"]:
            accepted.append((match.similarity, pose, candidate))

    accepted.sort(key=lambda item: item[0], reverse=True)
    result = {
        "success": bool(accepted),
        "selected": None,
        "candidate_count": len(candidate_scores),
        "runtime": {
            "match_seconds": match_runtime_s,
            "total_seconds": time.perf_counter() - started_at,
        },
        "candidates": candidate_scores,
    }
    if accepted:
        _, pose, selected = accepted[0]
        np.savetxt(final_pose_file, np.asarray(pose, dtype=np.float64))
        result["selected"] = selected

    _write_topk_overlay(output_dir / "topk_matches.png", color, candidate_scores)
    _write_projected_pose_overlay(
        output_dir / "projected_pose_overlay.png",
        color,
        first_appearance_mask,
    )
    _write_template_feature_overlay(
        output_dir / "template_feature_overlay.png",
        color,
        first_match,
        first_template,
    )
    # Prepare observed depth in meters for visualization
    from scripts.linemod_depth_validation import depth_image_to_meters
    observed_depth_m = depth_image_to_meters(depth, depth_scale=float(args.depth_scale))

    _write_rendered_vs_observed_depth(
        output_dir / "rendered_vs_observed_depth.png",
        color,
        first_rendered_depth_result,
        first_rendered_depth,
        first_rendered_mask,
        observed_depth_m,
        max_depth_residual_m=float(getattr(args, "max_depth_residual_m", 0.02)),
    )
    cv2.imwrite(
        str(output_dir / "black_white_validation_map.png"),
        _black_white_validation_image(color, first_appearance_mask),
    )

    (output_dir / "candidate_scores.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--color", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--camera_info", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth_scale", type=float, default=0.001)
    parser.add_argument("--max_abs_z_error_m", type=float, default=0.06)
    parser.add_argument("--min_valid_depths", type=int, default=8)
    parser.add_argument("--mesh_file", default=None)
    parser.add_argument("--max_depth_residual_m", type=float, default=0.02)
    parser.add_argument("--min_depth_consistency_ratio", type=float, default=0.70)
    parser.add_argument("--min_rendered_depth_overlap_pixels", type=int, default=50)
    parser.add_argument("--enable_depth_normal_score", action="store_true")
    parser.add_argument("--line2d_score_weight", type=float, default=0.6)
    parser.add_argument("--normal_score_weight", type=float, default=0.4)
    parser.add_argument("--roi_json", default=None)
    parser.add_argument("--roi_class_name", default=None)
    parser.add_argument("--roi_min_confidence", type=float, default=0.0)
    parser.add_argument("--roi_padding_px", type=int, default=0)
    parser.add_argument("--yolo_model", default=None)
    parser.add_argument("--yolo_target_class", type=int, default=None)
    parser.add_argument("--yolo_conf", type=float, default=0.7)
    parser.add_argument("--disable_black_white_validation", action="store_true")
    parser.add_argument("--max_saturation", type=float, default=45.0)
    parser.add_argument("--black_v_max", type=float, default=80.0)
    parser.add_argument("--white_v_min", type=float, default=150.0)
    parser.add_argument("--min_low_saturation_ratio", type=float, default=0.65)
    parser.add_argument("--min_black_white_support", type=float, default=0.45)
    parsed = parser.parse_args()
    parsed.enable_black_white_validation = not parsed.disable_black_white_validation
    result = run_init_frame(parsed)
    status = "success" if result["success"] else "failure"
    print(f"LINEMOD_INIT_FRAME status={status} candidates={result['candidate_count']}")
    if result["success"]:
        print("pose=final_pose.txt")


if __name__ == "__main__":
    main()
