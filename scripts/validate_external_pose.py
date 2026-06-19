#!/usr/bin/env python3
"""
Validate rendered depth consistency using an external pose (e.g., from FoundationPose, M3T).

This script tests whether the renderer-validator pipeline works correctly by using
a known-good pose from another method as ground truth.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_depth_validation import depth_image_to_meters
from scripts.linemod_silhouette_renderer import load_obj_mesh, render_model_buffers
from scripts.linemod_validator import validate_rendered_depth_consistency


def load_pose_from_file(pose_file):
    """Load 4x4 pose matrix from file (txt or json)."""
    pose_file = Path(pose_file)
    if pose_file.suffix == '.json':
        with open(pose_file) as f:
            data = json.load(f)
            # Handle different JSON formats
            if 'pose' in data:
                pose = np.array(data['pose'], dtype=np.float64)
            elif 'object_in_camera' in data:
                pose = np.array(data['object_in_camera'], dtype=np.float64)
            else:
                # Assume the JSON is a 4x4 matrix directly
                pose = np.array(data, dtype=np.float64)
    else:
        # Assume text file with 4x4 matrix
        pose = np.loadtxt(pose_file, dtype=np.float64)

    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4 matrix, got {pose.shape}")

    return pose


def load_camera_matrix(camera_info_file):
    """Load camera intrinsic matrix from yaml file."""
    import re
    text = Path(camera_info_file).read_text(encoding='utf-8')
    match = re.search(r'^k:\s*(\[[^\]]+\])', text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"camera_info.yaml does not contain k: {camera_info_file}")
    values = eval(match.group(1))
    if len(values) != 9:
        raise ValueError("camera_info k field must contain 9 values")
    return np.array(values, dtype=np.float64).reshape(3, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Validate rendered depth consistency with external pose"
    )
    parser.add_argument("--pose", required=True, help="Pose file (4x4 matrix, txt or json)")
    parser.add_argument("--mesh", required=True, help="Mesh file (.obj)")
    parser.add_argument("--color", required=True, help="Color image")
    parser.add_argument("--depth", required=True, help="Depth image")
    parser.add_argument("--camera_info", required=True, help="Camera info yaml")
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Depth scale (default: 0.001)")
    parser.add_argument("--max_depth_residual_m", type=float, default=0.02, help="Max residual (default: 0.02m)")
    parser.add_argument("--min_consistency_ratio", type=float, default=0.70, help="Min ratio (default: 0.70)")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for visualizations")
    args = parser.parse_args()

    # Load inputs
    print("=" * 80)
    print("EXTERNAL POSE VALIDATION TEST")
    print("=" * 80)
    print()

    pose = load_pose_from_file(args.pose)
    print(f"Loaded pose from: {args.pose}")
    print(f"  Translation: ({pose[0,3]:.4f}, {pose[1,3]:.4f}, {pose[2,3]:.4f})")
    print(f"  Z-axis: {pose[2,3]:.4f}m")
    print()

    vertices, faces = load_obj_mesh(args.mesh)
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"  Centroid: ({vertices[:,0].mean():.4f}, {vertices[:,1].mean():.4f}, {vertices[:,2].mean():.4f})")
    print()

    k_matrix = load_camera_matrix(args.camera_info)
    print(f"Camera matrix:")
    print(f"  fx={k_matrix[0,0]:.2f}, fy={k_matrix[1,1]:.2f}")
    print(f"  cx={k_matrix[0,2]:.2f}, cy={k_matrix[1,2]:.2f}")
    print()

    color = cv2.imread(str(args.color), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)

    if color is None:
        raise RuntimeError(f"Could not read color image: {args.color}")
    if depth is None:
        raise RuntimeError(f"Could not read depth image: {args.depth}")

    print(f"Images: {color.shape[1]}x{color.shape[0]}")
    print()

    # Validate rendered depth consistency
    print("Validating rendered depth consistency...")
    print("-" * 80)

    result = validate_rendered_depth_consistency(
        vertices,
        faces,
        pose,
        k_matrix,
        depth,
        image_size=(color.shape[1], color.shape[0]),
        depth_scale=args.depth_scale,
        max_depth_residual_m=args.max_depth_residual_m,
        min_consistency_ratio=args.min_consistency_ratio,
    )

    print(f"Success: {result.success}")
    print(f"Consistency ratio: {result.consistency_ratio:.4f} (threshold: {args.min_consistency_ratio})")
    print(f"Rendered pixels: {result.rendered_pixel_count}")
    print(f"Valid overlap: {result.valid_overlap_count}")
    print(f"Supported pixels: {result.supported_pixel_count}")
    if result.mean_abs_depth_residual_m is not None:
        print(f"Mean abs residual: {result.mean_abs_depth_residual_m:.4f}m ({result.mean_abs_depth_residual_m*1000:.1f}mm)")

    if not result.success:
        print(f"Reason: {result.reason}")

    print()

    # Generate visualizations if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render
        rendered_color, rendered_mask, _, rendered_depth, _ = render_model_buffers(
            vertices, faces, pose, k_matrix, image_size=(color.shape[1], color.shape[0])
        )

        # Convert observed depth to meters
        observed_depth_m = depth_image_to_meters(depth, depth_scale=args.depth_scale)

        # Generate heatmap
        overlay = (color.astype(np.float32) * 0.4).astype(np.uint8)

        rendered_valid = (rendered_mask > 0) & np.isfinite(rendered_depth) & (rendered_depth > 0.0)
        observed_valid = np.isfinite(observed_depth_m) & (observed_depth_m > 0.0)
        overlap = rendered_valid & observed_valid

        if np.any(overlap):
            residual = np.abs(rendered_depth - observed_depth_m)
            consistent = overlap & (residual <= args.max_depth_residual_m)
            inconsistent = overlap & (residual > args.max_depth_residual_m)

            overlay[consistent] = [0, 255, 0]  # Green
            overlay[inconsistent] = [0, 0, 255]  # Red

        only_rendered = rendered_valid & ~observed_valid
        overlay[only_rendered] = [255, 128, 0]  # Blue

        cv2.imwrite(str(output_dir / "rendered_vs_observed_depth.png"), overlay)

        # Projected pose overlay
        pose_overlay = color.copy()
        if np.any(rendered_mask > 0):
            contours, _ = cv2.findContours(
                (rendered_mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(pose_overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "projected_pose_overlay.png"), pose_overlay)

        # Save result JSON
        result_json = {
            "pose_file": str(args.pose),
            "mesh_file": str(args.mesh),
            "pose": pose.tolist(),
            "validation": result.to_dict(),
        }
        (output_dir / "validation_result.json").write_text(
            json.dumps(result_json, indent=2),
            encoding='utf-8',
        )

        print(f"Visualizations saved to: {output_dir}")
        print()

    # Summary
    print("=" * 80)
    if result.success and result.consistency_ratio >= 0.95:
        print("✅ EXCELLENT: Consistency ratio > 0.95")
        print("   → Renderer/validator/mesh/depth_scale are all correct")
        print("   → If LINEMOD fails, the problem is in template matching")
    elif result.success:
        print(f"✅ PASS: Consistency ratio {result.consistency_ratio:.3f} ≥ {args.min_consistency_ratio}")
        print("   → Renderer/validator work, but may have minor issues")
    else:
        print(f"❌ FAIL: Consistency ratio {result.consistency_ratio:.3f} < {args.min_consistency_ratio}")
        print("   → Check mesh scale, depth_scale, camera intrinsics, RGB-D alignment")
    print("=" * 80)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
