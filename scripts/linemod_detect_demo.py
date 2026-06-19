#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_line2d import load_line2d_templates, match_line2d_templates
from scripts.linemod_pose_utils import replace_pose_translation_from_depth
from scripts.linemod_template_store import find_template_metadata, load_template_metadata


def camera_matrix(args):
    return np.array(
        [
            [float(args.fx), 0.0, float(args.cx)],
            [0.0, float(args.fy), float(args.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def draw_match(image_bgr, template, match, pose):
    output = image_bgr.copy()
    cv2.rectangle(
        output,
        (match.x, match.y),
        (match.x + template.width, match.y + template.height),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output,
        f"{match.class_id}:{match.template_id} score={match.similarity:.1f}",
        (match.x, max(20, match.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"t=({pose[0,3]:.3f},{pose[1,3]:.3f},{pose[2,3]:.3f})",
        (10, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def run_demo(args):
    image = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read color image: {args.color}")

    templates = load_line2d_templates(args.templates)
    metadata = load_template_metadata(args.poses)
    matches = match_line2d_templates(
        image,
        templates,
        threshold=float(args.threshold),
        stride=int(args.stride),
    )
    if not matches:
        raise RuntimeError("No Line2D matches passed the threshold")

    best = matches[0]
    template_by_id = {(item.class_id, item.template_id): item for item in templates}
    template = template_by_id[(best.class_id, best.template_id)]
    pose_metadata = find_template_metadata(metadata, best.class_id, best.template_id)
    pose = pose_metadata.object_in_camera.copy()

    if args.depth_m is not None:
        center_u = best.x + template.width / 2.0
        center_v = best.y + template.height / 2.0
        pose = replace_pose_translation_from_depth(
            pose,
            center_u=center_u,
            center_v=center_v,
            depth_m=float(args.depth_m),
            k_matrix=camera_matrix(args),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "linemod_detect_pose.txt", pose)
    cv2.imwrite(str(output_dir / "linemod_detect.png"), draw_match(image, template, best, pose))
    print(
        f"LINE2D_MATCH class_id={best.class_id} template_id={best.template_id} "
        f"score={best.similarity:.2f} x={best.x} y={best.y}"
    )
    print(f"pose={output_dir / 'linemod_detect_pose.txt'}")
    print(f"diagnostic={output_dir / 'linemod_detect.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--color", required=True)
    parser.add_argument("--output_dir", default="runtime/linemod_detect")
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth_m", type=float, default=None)
    parser.add_argument("--fx", type=float, default=606.1544849121768)
    parser.add_argument("--fy", type=float, default=606.2304160969798)
    parser.add_argument("--cx", type=float, default=320.71911360697)
    parser.add_argument("--cy", type=float, default=239.7231533262084)
    run_demo(parser.parse_args())


if __name__ == "__main__":
    main()
