#!/usr/bin/env python3
"""
Normalize mesh by centering it at the origin.

This fixes the issue where Str3D.obj has its centroid at (0.047, 0.047, -0.0677)
instead of (0, 0, 0), causing a systematic depth offset in pose estimation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.linemod_silhouette_renderer import load_obj_mesh


def normalize_mesh_to_origin(input_path, output_path, center_method="centroid"):
    """
    Normalize mesh by translating it so its center is at the origin.

    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
        center_method: "centroid" (mean of all vertices) or "bbox" (center of bounding box)
    """
    vertices, faces = load_obj_mesh(input_path)

    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"\nOriginal mesh statistics:")
    print(f"  X range: [{vertices[:, 0].min():.6f}, {vertices[:, 0].max():.6f}]")
    print(f"  Y range: [{vertices[:, 1].min():.6f}, {vertices[:, 1].max():.6f}]")
    print(f"  Z range: [{vertices[:, 2].min():.6f}, {vertices[:, 2].max():.6f}]")

    if center_method == "centroid":
        center = vertices.mean(axis=0)
        print(f"  Centroid: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    elif center_method == "bbox":
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        print(f"  BBox center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    else:
        raise ValueError(f"Unknown center_method: {center_method}")

    # Translate vertices to center at origin
    normalized_vertices = vertices - center

    print(f"\nNormalized mesh statistics:")
    print(f"  X range: [{normalized_vertices[:, 0].min():.6f}, {normalized_vertices[:, 0].max():.6f}]")
    print(f"  Y range: [{normalized_vertices[:, 1].min():.6f}, {normalized_vertices[:, 1].max():.6f}]")
    print(f"  Z range: [{normalized_vertices[:, 2].min():.6f}, {normalized_vertices[:, 2].max():.6f}]")
    new_centroid = normalized_vertices.mean(axis=0)
    print(f"  New centroid: ({new_centroid[0]:.6f}, {new_centroid[1]:.6f}, {new_centroid[2]:.6f})")

    print(f"\nTranslation applied: ({-center[0]:.6f}, {-center[1]:.6f}, {-center[2]:.6f})")

    # Write normalized OBJ file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as f:
        f.write(f"# Normalized mesh from {Path(input_path).name}\n")
        f.write(f"# Translation: ({-center[0]:.6f}, {-center[1]:.6f}, {-center[2]:.6f})\n")
        f.write(f"# Center method: {center_method}\n\n")

        # Write vertices
        for v in normalized_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (1-indexed, unchanged)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"\nNormalized mesh saved to: {output_path}")
    print(f"\n⚠️  IMPORTANT: After using normalized mesh, you MUST rebuild all templates!")
    print(f"   Old templates were built with the old mesh coordinate system.")

    return center


def main():
    parser = argparse.ArgumentParser(description="Normalize mesh to origin")
    parser.add_argument("--input", required=True, help="Input OBJ file")
    parser.add_argument("--output", required=True, help="Output OBJ file")
    parser.add_argument(
        "--center_method",
        choices=["centroid", "bbox"],
        default="centroid",
        help="Method to compute center (default: centroid)",
    )
    args = parser.parse_args()

    normalize_mesh_to_origin(args.input, args.output, args.center_method)


if __name__ == "__main__":
    main()
