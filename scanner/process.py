"""Reconstruct mesh from a saved scan session."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import open3d as o3d

from scanner.fusion import fuse_point_clouds
from scanner.rgbd_odometry import fuse_rgbd_sequence
from scanner.bbox import axis_aligned_bbox_mm, format_bbox_report
from scanner.mesh import export_mesh, pointcloud_to_mesh
from scanner.pointcloud import rgbd_to_pointcloud
from scanner.segment import isolate_object, keep_largest_cluster
from scanner.session import ScanSession, load_frame_rgb_depth, load_session, session_root
from scanner.session_mask import load_session_mask, mask_depth_for_session
from scanner.turntable_fusion import cap_open_faces, fuse_turntable_session


def session_to_point_clouds(session: ScanSession) -> List[o3d.geometry.PointCloud]:
    root = session_root(session)
    mask_spec = load_session_mask(root)
    clouds: List[o3d.geometry.PointCloud] = []
    for record in session.frames:
        rgb, depth = load_frame_rgb_depth(session, record)
        depth = mask_depth_for_session(root, depth, spec=mask_spec)
        cloud = rgbd_to_pointcloud(rgb, depth, session.intrinsics)
        if len(cloud.points) > 0:
            clouds.append(cloud)
    return clouds


def process_session(
    session_path: Path,
    *,
    output_mesh: Optional[Path] = None,
    save_pointcloud: bool = True,
    use_icp: bool = False,
    use_odometry: bool = False,
    use_turntable_pose: bool = True,
    refine_pivot_xz: bool = False,
    use_dot_track: bool = True,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    session = load_session(session_path)
    root = session_root(session)
    if len(session.frames) < 2:
        raise ValueError("Need at least 2 captured frames to reconstruct.")

    print(f"Processing {len(session.frames)} frames from {root.name} ({session.mode} mode)...")
    mask_spec = load_session_mask(root)

    if session.mode == "turntable" and use_turntable_pose:
        print(
            "Fusing frames (platter dot tracking + depth prefilter 100-400 mm)..."
        )
        fused = fuse_turntable_session(
            root,
            save_voxel_preview=True,
            refine_pivot_xz=refine_pivot_xz,
            use_dot_track=use_dot_track,
        )
        if len(fused.points) == 0:
            raise ValueError("Turntable fusion produced an empty point cloud.")
        lo_f, hi_f = axis_aligned_bbox_mm(fused)
        print(f"  Fused cloud: {len(fused.points)} points")
        print(format_bbox_report("Fused cloud", lo_f, hi_f))
    elif use_odometry:
        print("Fusing frames (RGB-D odometry)...")
        frames: list[tuple] = []
        for record in session.frames:
            rgb, depth = load_frame_rgb_depth(session, record)
            depth = mask_depth_for_session(root, depth, spec=mask_spec)
            frames.append((rgb, depth))
        fused, poses, ok = fuse_rgbd_sequence(frames, session.intrinsics)
        n_ok = sum(ok)
        print(f"  Odometry: {n_ok}/{max(1, len(ok))} steps succeeded")
        if len(fused.points) == 0:
            raise ValueError("Odometry fusion produced an empty point cloud.")
    else:
        clouds = session_to_point_clouds(session)
        if use_icp:
            print("Fusing frames (ICP registration)...")
        else:
            print("Fusing frames (concatenate, no ICP)...")
        fused = fuse_point_clouds(clouds, use_icp=use_icp)

    if session.mode == "turntable" and use_turntable_pose:
        # Turntable fusion already removed the platter via an axis-aware cylinder
        # crop; RANSAC plane removal would delete a flat face of a boxy object.
        print("Segmenting object (largest cluster + outlier removal)...")
        object_cloud = keep_largest_cluster(fused)
        if len(object_cloud.points) > 40:
            object_cloud, _ = object_cloud.remove_statistical_outlier(
                nb_neighbors=24, std_ratio=1.6
            )
    else:
        print("Segmenting object (plane removal + largest cluster)...")
        object_cloud = isolate_object(fused)

    if save_pointcloud:
        ply_path = root / "scan_object.ply"
        o3d.io.write_point_cloud(str(ply_path), object_cloud, print_progress=False)
        print(f"Saved point cloud: {ply_path}")

    print("Building mesh (Poisson reconstruction)...")
    if session.mode == "turntable" and use_turntable_pose:
        # Cap the flat open top (reflective center returns no depth) and the
        # occluded bottom so Poisson closes the box instead of ballooning. The
        # saved/measured cloud keeps only real captured points.
        mesh_cloud = cap_open_faces(object_cloud)
        mesh = pointcloud_to_mesh(
            mesh_cloud,
            depth=8,
            density_quantile=0.02,
            keep_largest=True,
            smooth_iters=14,
            clip_to_cloud_max_dist=0.012,
        )
    else:
        mesh = pointcloud_to_mesh(object_cloud)

    mesh_path = output_mesh or (root / "scan_mesh.obj")
    export_mesh(mesh, mesh_path)
    print(f"Saved mesh: {mesh_path}")

    lo, hi = axis_aligned_bbox_mm(object_cloud)
    print(format_bbox_report("Object (segmented cloud)", lo, hi))
    return object_cloud, mesh
