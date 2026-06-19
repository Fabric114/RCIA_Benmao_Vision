#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_line2d import extract_line2d_template, save_line2d_templates
from scripts.linemod_pose_utils import look_at_object_in_camera, sample_hemisphere_views
from scripts.linemod_depth_features import extract_depth_normal_features
from scripts.linemod_silhouette_renderer import load_obj_mesh, render_model_buffers
from scripts.linemod_template_store import TemplatePoseMetadata, save_template_metadata


@dataclass(frozen=True)
class TemplatePoseSpec:
    view_index: int
    distance: float
    camera_position: np.ndarray
    inplane_degree: float
    object_in_camera: np.ndarray


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_elevations(value):
    return parse_float_list(value)


def parse_distances(args):
    distances = getattr(args, "distances", None)
    if distances:
        return parse_float_list(distances)
    return [float(args.distance)]


def parse_inplane_degrees(args):
    inplane_degrees = getattr(args, "inplane_degrees", None)
    if inplane_degrees:
        return parse_float_list(inplane_degrees)
    return [0.0]


def apply_camera_inplane_rotation(object_in_camera, inplane_degree):
    pose = np.asarray(object_in_camera, dtype=np.float64).reshape(4, 4).copy()
    angle = np.deg2rad(float(inplane_degree))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    roll_camera = np.eye(3, dtype=np.float64)
    roll_camera[:2, :2] = [[c, -s], [s, c]]
    pose[:3, :3] = roll_camera @ pose[:3, :3]
    pose[3, :] = [0.0, 0.0, 0.0, 1.0]
    return pose


def iter_template_pose_specs(args):
    view_index = 0
    for distance in parse_distances(args):
        camera_positions = sample_hemisphere_views(
            radius=float(distance),
            azimuth_count=int(args.azimuth_count),
            elevation_degrees=parse_elevations(args.elevations),
        )
        for camera_position in camera_positions:
            base_pose = look_at_object_in_camera(camera_position)
            for inplane_degree in parse_inplane_degrees(args):
                yield TemplatePoseSpec(
                    view_index=view_index,
                    distance=float(distance),
                    camera_position=camera_position,
                    inplane_degree=float(inplane_degree),
                    object_in_camera=apply_camera_inplane_rotation(base_pose, inplane_degree),
                )
                view_index += 1


def camera_matrix(args):
    return np.array(
        [
            [float(args.fx), 0.0, float(args.cx)],
            [0.0, float(args.fy), float(args.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_templates(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = load_obj_mesh(args.mesh_file)
    k_matrix = camera_matrix(args)
    templates = []
    metadata = []
    template_id = 0
    for spec in iter_template_pose_specs(args):
        object_in_camera = spec.object_in_camera
        color, mask, model_point_image, _, normal_image = render_model_buffers(
            vertices,
            faces,
            object_in_camera,
            k_matrix,
            image_size=(int(args.width), int(args.height)),
        )
        if int(mask.sum()) == 0:
            continue
        try:
            template = extract_line2d_template(
                color,
                mask,
                class_id=args.class_id,
                template_id=template_id,
                max_features=int(args.max_features),
                min_magnitude=float(args.min_magnitude),
                feature_spacing=int(args.feature_spacing),
                model_point_image=model_point_image,
                require_model_points=True,
                contour_band_px=int(getattr(args, "contour_band_px", 3)),
            )
        except ValueError:
            continue

        normal_features = extract_depth_normal_features(
            normal_image,
            mask,
            bbox=template.bbox,
            model_point_image=model_point_image,
            max_features=int(args.max_normal_features),
            feature_spacing=int(args.normal_feature_spacing),
        )
        template = template.__class__(
            class_id=template.class_id,
            template_id=template.template_id,
            width=template.width,
            height=template.height,
            bbox=template.bbox,
            features=template.features,
            normal_features=tuple(normal_features),
        )

        x, y, width, height = template.bbox
        templates.append(template)
        metadata.append(
            TemplatePoseMetadata(
                class_id=args.class_id,
                template_id=template_id,
                object_in_camera=object_in_camera,
                match_center=[x + width / 2.0, y + height / 2.0],
                source=(
                    f"view_index={spec.view_index},distance={spec.distance},"
                    f"inplane_degree={spec.inplane_degree},"
                    f"camera_position={spec.camera_position.tolist()}"
                ),
            )
        )
        if template_id < int(args.save_diagnostics):
            cv2.imwrite(str(diagnostic_dir / f"template_{template_id:04d}.png"), color)
            cv2.imwrite(str(diagnostic_dir / f"mask_{template_id:04d}.png"), mask)
        template_id += 1

    if not templates:
        raise RuntimeError("No valid Line2D templates were generated")

    templates_file = output_dir / f"{args.class_id}_line2d_templates.json"
    poses_file = output_dir / f"{args.class_id}_template_poses.json"
    save_line2d_templates(templates_file, templates)
    save_template_metadata(poses_file, metadata)
    return templates_file, poses_file, len(templates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", default="models/object/Str3D.obj")
    parser.add_argument("--output_dir", default="runtime/linemod")
    parser.add_argument("--class_id", default="Str3D")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=606.1544849121768)
    parser.add_argument("--fy", type=float, default=606.2304160969798)
    parser.add_argument("--cx", type=float, default=320.71911360697)
    parser.add_argument("--cy", type=float, default=239.7231533262084)
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument("--distances", default=None)
    parser.add_argument("--azimuth_count", type=int, default=8)
    parser.add_argument("--elevations", default="0,25,50")
    parser.add_argument("--inplane_degrees", default="0")
    parser.add_argument("--max_features", type=int, default=96)
    parser.add_argument("--min_magnitude", type=float, default=20.0)
    parser.add_argument("--feature_spacing", type=int, default=5)
    parser.add_argument("--contour_band_px", type=int, default=3)
    parser.add_argument("--max_normal_features", type=int, default=64)
    parser.add_argument("--normal_feature_spacing", type=int, default=8)
    parser.add_argument("--save_diagnostics", type=int, default=8)
    args = parser.parse_args()

    templates_file, poses_file, count = build_templates(args)
    print(f"LINE2D_TEMPLATES_BUILT count={count}")
    print(f"templates={templates_file}")
    print(f"poses={poses_file}")


if __name__ == "__main__":
    main()
