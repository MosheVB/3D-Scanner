"""CLI helpers to report scan dimensions from saved sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from scanner.bbox import (
    axis_aligned_bbox_mm,
    compare_to_expected,
    format_bbox_report,
    parse_expected_mm,
)
from scanner.session import ScanSession, load_session


def _load_geometry(session_path: Path, source: str) -> o3d.geometry.Geometry:
    root = session_path if session_path.is_dir() else session_path.parent
    ply = root / "scan_object.ply"
    obj = root / "scan_mesh.obj"
    if source == "mesh" and obj.is_file():
        mesh = o3d.io.read_triangle_mesh(str(obj))
        if mesh.is_empty():
            raise ValueError(f"Empty mesh: {obj}")
        return mesh
    if source in ("cloud", "mesh") and ply.is_file():
        cloud = o3d.io.read_point_cloud(str(ply))
        if len(cloud.points) == 0:
            raise ValueError(f"Empty point cloud: {ply}")
        return cloud
    if source == "mesh" and not obj.is_file():
        raise FileNotFoundError(
            f"No scan_mesh.obj in {root}. Run: python -m scanner process {root}"
        )
    raise FileNotFoundError(
        f"No scan_object.ply in {root}. Run: python -m scanner process {root}"
    )


def measure_fused_cloud(
    session: ScanSession, *, use_icp: bool = True
) -> o3d.geometry.PointCloud:
    from scanner.fusion import fuse_point_clouds
    from scanner.process import session_to_point_clouds

    clouds = session_to_point_clouds(session)
    if not clouds:
        raise ValueError("Session has no valid depth frames.")
    if len(clouds) == 1:
        return clouds[0]
    return fuse_point_clouds(clouds, use_icp=use_icp)


def measure_session(
    session_path: Path,
    *,
    source: str = "cloud",
    reprocess: bool = False,
    use_icp: bool = True,
    expected_mm: str | None = None,
    tolerance_pct: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Print bbox dimensions; return (min_mm, max_mm)."""
    root = session_path.resolve()
    session = load_session(root)

    if reprocess or (
        source == "mesh"
        and not (root / "scan_mesh.obj").is_file()
        and len(session.frames) >= 2
    ):
        from scanner.process import process_session

        process_session(root, use_icp=use_icp)

    if source == "frames":
        cloud = measure_fused_cloud(session, use_icp=use_icp)
        lo, hi = axis_aligned_bbox_mm(cloud)
        print(format_bbox_report("Fused frames (raw, no segmentation)", lo, hi))
        if expected_mm:
            print(
                compare_to_expected(
                    lo,
                    hi,
                    parse_expected_mm(expected_mm),
                    tolerance_pct=tolerance_pct,
                )
            )
        return lo, hi

    geom = _load_geometry(root, source)
    lo, hi = axis_aligned_bbox_mm(geom)
    label = "Segmented point cloud" if source == "cloud" else "Mesh"
    print(format_bbox_report(label, lo, hi))
    if expected_mm:
        print(
            compare_to_expected(
                lo,
                hi,
                parse_expected_mm(expected_mm),
                tolerance_pct=tolerance_pct,
            )
        )
    return lo, hi
