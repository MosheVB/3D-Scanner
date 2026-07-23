"""Axis-aligned bounding box helpers (geometry in meters, reports in mm)."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def axis_aligned_bbox_mm(
    geometry: o3d.geometry.Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) in millimeters for a point cloud or mesh."""
    if isinstance(geometry, o3d.geometry.PointCloud):
        pts = np.asarray(geometry.points)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        pts = np.asarray(geometry.vertices)
    else:
        raise TypeError(f"Unsupported geometry: {type(geometry)}")
    if pts.size == 0:
        raise ValueError("Geometry has no points.")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return lo * 1000.0, hi * 1000.0


def bbox_extents_mm(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return hi - lo


def format_bbox_report(
    label: str,
    lo: np.ndarray,
    hi: np.ndarray,
) -> str:
    ext = bbox_extents_mm(lo, hi)
    sorted_ext = np.sort(ext)[::-1]
    lines = [
        f"{label} axis-aligned bounding box (mm):",
        f"  X: {ext[0]:.1f}  (min {lo[0]:.1f}, max {hi[0]:.1f})",
        f"  Y: {ext[1]:.1f}  (min {lo[1]:.1f}, max {hi[1]:.1f})",
        f"  Z: {ext[2]:.1f}  (min {lo[2]:.1f}, max {hi[2]:.1f})",
        f"  extents (WxHxD): {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm",
        f"  sorted (L×M×S): {sorted_ext[0]:.1f} × {sorted_ext[1]:.1f} × {sorted_ext[2]:.1f} mm",
    ]
    return "\n".join(lines)


def parse_expected_mm(spec: str) -> np.ndarray:
    """Parse comma-separated length,width,thickness in mm (any order)."""
    parts = [float(p.strip()) for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values in mm, e.g. 136,35,9.25")
    return np.array(sorted(parts, reverse=True), dtype=np.float64)


def compare_to_expected(
    lo: np.ndarray,
    hi: np.ndarray,
    expected_mm: np.ndarray,
    *,
    tolerance_pct: float = 15.0,
) -> str:
    """Compare sorted bbox extents to sorted expected L×M×S (mm)."""
    measured = np.sort(bbox_extents_mm(lo, hi))[::-1]
    expected = np.sort(expected_mm)[::-1]
    lines = ["Validation vs expected (sorted L×M×S, mm):"]
    ok = True
    for i, label in enumerate(("L", "M", "S")):
        exp = expected[i]
        got = measured[i]
        if exp <= 0:
            err_pct = 0.0
        else:
            err_pct = abs(got - exp) / exp * 100.0
        pass_axis = err_pct <= tolerance_pct
        ok = ok and pass_axis
        status = "PASS" if pass_axis else "FAIL"
        lines.append(
            f"  {label}: measured {got:.1f} vs expected {exp:.1f}  "
            f"({err_pct:+.1f}% err)  [{status}]"
        )
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'} (±{tolerance_pct:.0f}% per axis)")
    return "\n".join(lines)
