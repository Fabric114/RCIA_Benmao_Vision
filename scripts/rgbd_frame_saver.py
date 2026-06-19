import json
from pathlib import Path

import cv2
import numpy as np


def _camera_info_to_yaml(camera_info):
    lines = [
        f"width: {int(camera_info.width)}",
        f"height: {int(camera_info.height)}",
        f"distortion_model: {camera_info.distortion_model}",
        "d: [" + ", ".join(str(float(value)) for value in camera_info.d) + "]",
        "k: [" + ", ".join(str(float(value)) for value in camera_info.k) + "]",
        "r: [" + ", ".join(str(float(value)) for value in camera_info.r) + "]",
        "p: [" + ", ".join(str(float(value)) for value in camera_info.p) + "]",
        "",
    ]
    return "\n".join(lines)


def _depth_to_png_u16(depth):
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth
    depth_float = depth.astype(np.float32)
    if depth_float.size and np.nanmax(depth_float) < 20.0:
        depth_float = depth_float * 1000.0
    depth_float[(~np.isfinite(depth_float)) | (depth_float < 0)] = 0
    return np.clip(depth_float, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def save_rgbd_frame(output_dir, color_bgr, depth, camera_info, stamp_ns):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_path = output_dir / "color.png"
    depth_path = output_dir / "depth.png"
    camera_info_path = output_dir / "camera_info.yaml"
    metadata_path = output_dir / "metadata.json"

    if not cv2.imwrite(str(color_path), np.asarray(color_bgr)):
        raise RuntimeError(f"Failed to write color image: {color_path}")
    if not cv2.imwrite(str(depth_path), _depth_to_png_u16(depth)):
        raise RuntimeError(f"Failed to write depth image: {depth_path}")
    camera_info_path.write_text(_camera_info_to_yaml(camera_info), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "stamp_ns": int(stamp_ns),
                "color_file": color_path.name,
                "depth_file": depth_path.name,
                "camera_info_file": camera_info_path.name,
                "depth_units": "millimeters",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "color": color_path,
        "depth": depth_path,
        "camera_info": camera_info_path,
        "metadata": metadata_path,
    }
