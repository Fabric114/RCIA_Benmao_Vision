from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Line2DFeature:
    x: int
    y: int
    orientation: int
    model_point: tuple | None = None


@dataclass(frozen=True)
class Line2DTemplate:
    class_id: str
    template_id: int
    width: int
    height: int
    bbox: tuple
    features: tuple
    normal_features: tuple = tuple()


@dataclass(frozen=True)
class Line2DMatch:
    class_id: str
    template_id: int
    x: int
    y: int
    similarity: float
    line_similarity: float | None = None
    normal_similarity: float | None = None


def _gradient_orientation(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    orientation = np.floor(((angle + 22.5) % 360.0) / 45.0).astype(np.uint8)
    return magnitude, orientation


def _orientation_close(a, b):
    delta = abs(int(a) - int(b)) % 8
    return delta <= 1 or delta >= 7


def extract_line2d_template(
    image_bgr,
    object_mask,
    class_id,
    template_id,
    max_features=80,
    min_magnitude=20.0,
    feature_spacing=4,
    model_point_image=None,
    require_model_points=False,
    contour_band_px=3,
):
    mask = (object_mask > 0).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(mask)
    if width <= 0 or height <= 0:
        raise ValueError("object_mask must contain at least one foreground pixel")

    magnitude, orientation = _gradient_orientation(image_bgr)
    if model_point_image is not None:
        model_point_image = np.asarray(model_point_image, dtype=np.float64)
        if model_point_image.shape[:2] != object_mask.shape[:2] or model_point_image.shape[2] != 3:
            raise ValueError("model_point_image must have shape HxWx3 matching object_mask")
    feature_mask = mask
    if int(contour_band_px) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=int(contour_band_px))
        feature_mask = ((mask > 0) & (eroded == 0)).astype(np.uint8)
    ys, xs = np.where((feature_mask > 0) & (magnitude >= float(min_magnitude)))
    candidates = sorted(
        zip(xs.tolist(), ys.tolist()),
        key=lambda point: float(magnitude[point[1], point[0]]),
        reverse=True,
    )

    selected = []
    spacing_sq = int(feature_spacing) * int(feature_spacing)
    for px, py in candidates:
        if len(selected) >= int(max_features):
            break
        if any(
            (px - sx) * (px - sx) + (py - sy) * (py - sy) < spacing_sq
            for sx, sy, _ in selected
        ):
            continue
        model_point = None
        if model_point_image is not None:
            candidate_model_point = model_point_image[py, px]
            if np.all(np.isfinite(candidate_model_point)):
                model_point = tuple(float(value) for value in candidate_model_point.tolist())
            elif require_model_points:
                continue
        selected.append((px, py, model_point))

    features = tuple(
        Line2DFeature(
            x=int(px - x),
            y=int(py - y),
            orientation=int(orientation[py, px]),
            model_point=model_point,
        )
        for px, py, model_point in selected
    )
    if not features:
        raise ValueError("No gradient features found in object_mask")

    return Line2DTemplate(
        class_id=str(class_id),
        template_id=int(template_id),
        width=int(width),
        height=int(height),
        bbox=(int(x), int(y), int(width), int(height)),
        features=features,
    )


def match_line2d_templates(
    image_bgr,
    templates,
    threshold=80.0,
    min_magnitude=20.0,
    stride=1,
    depth_image=None,
    k_matrix=None,
    depth_scale=0.001,
    line2d_score_weight=0.6,
    normal_score_weight=0.4,
    roi_boxes=None,
    roi_padding_px=0,
):
    magnitude, orientation = _gradient_orientation(image_bgr)
    normal_orientation = None
    if depth_image is not None and k_matrix is not None:
        from scripts.linemod_depth_features import compute_depth_normal_image, quantize_normal_orientations

        normal_orientation = quantize_normal_orientations(
            compute_depth_normal_image(depth_image, k_matrix, depth_scale=depth_scale)
        )
    height, width = image_bgr.shape[:2]
    matches = []
    stride = max(1, int(stride))
    roi_boxes = _normalize_roi_boxes(roi_boxes, width, height, int(roi_padding_px))
    for template in templates:
        if not template.features:
            continue
        max_x = width - template.width
        max_y = height - template.height
        if max_x < 0 or max_y < 0:
            continue
        for y in range(0, max_y + 1, stride):
            for x in range(0, max_x + 1, stride):
                if roi_boxes and not _template_center_in_roi(x, y, template, roi_boxes):
                    continue
                votes = 0
                for feature in template.features:
                    px = x + feature.x
                    py = y + feature.y
                    if magnitude[py, px] < float(min_magnitude):
                        continue
                    if _orientation_close(orientation[py, px], feature.orientation):
                        votes += 1
                line_similarity = 100.0 * float(votes) / float(len(template.features))
                similarity = line_similarity
                normal_similarity = None
                if normal_orientation is not None and template.normal_features:
                    normal_votes = 0
                    normal_total = 0
                    for feature in template.normal_features:
                        px = x + feature.x
                        py = y + feature.y
                        if px < 0 or py < 0 or px >= width or py >= height:
                            continue
                        observed_orientation = int(normal_orientation[py, px])
                        if observed_orientation == 255:
                            continue
                        normal_total += 1
                        if _orientation_close(observed_orientation, feature.orientation):
                            normal_votes += 1
                    if normal_total > 0:
                        normal_similarity = 100.0 * float(normal_votes) / float(normal_total)
                        similarity = (
                            float(line2d_score_weight) * line_similarity
                            + float(normal_score_weight) * normal_similarity
                        )
                if similarity >= float(threshold):
                    matches.append(
                        Line2DMatch(
                            class_id=template.class_id,
                            template_id=template.template_id,
                            x=int(x),
                            y=int(y),
                            similarity=similarity,
                            line_similarity=line_similarity,
                            normal_similarity=normal_similarity,
                        )
                    )
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return matches


def _normalize_roi_boxes(roi_boxes, image_width, image_height, padding_px=0):
    if not roi_boxes:
        return []
    normalized = []
    padding_px = int(padding_px)
    for roi in roi_boxes:
        if isinstance(roi, dict):
            if "bbox_xyxy" in roi:
                x1, y1, x2, y2 = roi["bbox_xyxy"]
            elif "bbox" in roi:
                x1, y1, x2, y2 = roi["bbox"]
            else:
                continue
        else:
            x1, y1, x2, y2 = roi
        x1 = max(0, int(round(float(x1))) - padding_px)
        y1 = max(0, int(round(float(y1))) - padding_px)
        x2 = min(int(image_width) - 1, int(round(float(x2))) + padding_px)
        y2 = min(int(image_height) - 1, int(round(float(y2))) + padding_px)
        if x2 >= x1 and y2 >= y1:
            normalized.append((x1, y1, x2, y2))
    return normalized


def _template_center_in_roi(x, y, template, roi_boxes):
    center_x = float(x) + float(template.width) * 0.5
    center_y = float(y) + float(template.height) * 0.5
    return any(
        float(x1) <= center_x <= float(x2) and float(y1) <= center_y <= float(y2)
        for x1, y1, x2, y2 in roi_boxes
    )


def _feature_to_dict(feature):
    data = {
        "x": feature.x,
        "y": feature.y,
        "orientation": feature.orientation,
    }
    if feature.model_point is not None:
        data["model_point"] = [float(value) for value in feature.model_point]
    return data


def _feature_from_dict(data):
    model_point = data.get("model_point")
    return Line2DFeature(
        x=int(data["x"]),
        y=int(data["y"]),
        orientation=int(data["orientation"]),
        model_point=None
        if model_point is None
        else tuple(float(value) for value in model_point),
    )


def _template_to_dict(template):
    data = {
        "class_id": template.class_id,
        "template_id": template.template_id,
        "width": template.width,
        "height": template.height,
        "bbox": list(template.bbox),
        "features": [_feature_to_dict(feature) for feature in template.features],
    }
    if template.normal_features:
        data["normal_features"] = [_feature_to_dict(feature) for feature in template.normal_features]
    return data


def _template_from_dict(data):
    return Line2DTemplate(
        class_id=str(data["class_id"]),
        template_id=int(data["template_id"]),
        width=int(data["width"]),
        height=int(data["height"]),
        bbox=tuple(int(value) for value in data["bbox"]),
        features=tuple(_feature_from_dict(item) for item in data["features"]),
        normal_features=tuple(
            _feature_from_dict(item) for item in data.get("normal_features", [])
        ),
    )


def save_line2d_templates(path, templates):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "templates": [_template_to_dict(template) for template in templates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_line2d_templates(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Line2D template file version")
    return [_template_from_dict(item) for item in payload["templates"]]
