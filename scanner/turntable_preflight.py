"""Pre-fusion pose sanity check: frame 0 vs ~180 deg in object frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from scanner.config import DEPTH_STRIDE
from scanner.depth_preprocess import preprocess_depth_frame
from scanner.pointcloud import rgbd_to_pointcloud
from scanner.session import ScanSession, load_frame_rgb_depth
from scanner.session_mask import mask_depth_for_session
from scanner.turntable_pose import TurntableGeometry, transform_cam_to_object

PREFLIGHT_FRAME_B = 12
PREFLIGHT_VOXEL_M = 0.002
PREFLIGHT_MAX_CORR_M = 0.015
PREFLIGHT_MIN_FITNESS = 0.12
PREFLIGHT_MAX_RMSE_M = 0.020
PREFLIGHT_MIN_POINTS = 30
PREFLIGHT_STRIDE = 1


class TurntablePreflightError(RuntimeError):
    """Raised when frame-0 / frame-12 clouds fail to overlap in object frame."""


def _fusion_helpers():
    from scanner import turntable_fusion as tf

    return tf


def _preflight_partner_index(
    n_frames: int,
    frame_rotations_deg: list[float] | None,
    *,
    target_deg: float = 30.0,
) -> int | None:
    """Frame index ~target_deg from frame 0.

    Opaque objects show different surfaces as they rotate, so a near-180 deg
    partner sees the *opposite* faces and barely overlaps even when perfectly
    aligned. A partner ~30 deg away still shares a face, making it a valid
    alignment check.
    """
    if n_frames < 2:
        return None
    if frame_rotations_deg and len(frame_rotations_deg) >= 2:
        rot0 = float(frame_rotations_deg[0])
        best_i: int | None = None
        best_err = 1e9
        for i, r in enumerate(frame_rotations_deg):
            if i == 0:
                continue
            delta = abs(float(r) - rot0) % 360.0
            if delta > 180.0:
                delta = 360.0 - delta
            err = abs(delta - target_deg)
            if err < best_err:
                best_err = err
                best_i = i
        if best_i is not None and best_err <= 20.0:
            return best_i
    if 1 < n_frames:
        return min(2, n_frames - 1)
    return None


def _frame_cloud_object_frame(
    session: ScanSession,
    root: Path,
    frame_idx: int,
    geom: TurntableGeometry,
    *,
    meta,
    mask_spec,
    include,
    bg_state,
    preprocess_depth: bool,
    frame_rotations_deg: list[float] | None,
    stride: int,
) -> o3d.geometry.PointCloud:
    tf = _fusion_helpers()
    rec = session.frames[frame_idx]
    rgb, depth = load_frame_rgb_depth(session, rec)
    if preprocess_depth:
        depth = preprocess_depth_frame(depth, session.intrinsics)
    if mask_spec is not None and include is not None:
        depth = mask_depth_for_session(root, depth, spec=mask_spec, include=include)
    else:
        mask = tf._frame_mask(
            rgb,
            depth,
            session.intrinsics,
            include,
            bg_state=bg_state,
            frame_idx=frame_idx,
        )
        depth = depth.copy()
        depth[~mask] = 0
    if not np.any(depth > 0):
        return o3d.geometry.PointCloud()
    n_frames = len(session.frames)
    tilt, rot = tf.frame_pose(rec, frame_idx, meta)
    if frame_rotations_deg is not None and frame_idx < len(frame_rotations_deg):
        rot = float(frame_rotations_deg[frame_idx])
    else:
        rot = tf.rotation_deg_for_frame(rec, frame_idx, n_frames, meta)
    cloud = rgbd_to_pointcloud(rgb, depth, session.intrinsics, stride=stride)
    if len(cloud.points) == 0:
        return cloud
    cloud.transform(transform_cam_to_object(geom, tilt_deg=tilt, rotation_deg=rot))
    return cloud


def _overlap_metrics(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
    *,
    max_corr_m: float = PREFLIGHT_MAX_CORR_M,
) -> tuple[float, float]:
    a = cloud_a.voxel_down_sample(PREFLIGHT_VOXEL_M)
    b = cloud_b.voxel_down_sample(PREFLIGHT_VOXEL_M)
    if len(a.points) < PREFLIGHT_MIN_POINTS or len(b.points) < PREFLIGHT_MIN_POINTS:
        return 0.0, float("inf")
    eye = np.eye(4)
    ev_ab = o3d.pipelines.registration.evaluate_registration(
        a, b, max_corr_m, eye
    )
    ev_ba = o3d.pipelines.registration.evaluate_registration(
        b, a, max_corr_m, eye
    )
    fitness = max(float(ev_ab.fitness), float(ev_ba.fitness))
    rmse = min(float(ev_ab.inlier_rmse), float(ev_ba.inlier_rmse))
    return fitness, rmse


def _paint_cloud(cloud: o3d.geometry.PointCloud, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(cloud)
    out.paint_uniform_color(list(color))
    return out


def _show_preflight_window(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
    *,
    frame_b: int,
) -> None:
    import os

    if os.environ.get("SCANNER_PREFLIGHT_NO_GUI", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    vis_a = _paint_cloud(cloud_a, (1.0, 0.35, 0.35))
    vis_b = _paint_cloud(cloud_b, (0.35, 0.55, 1.0))
    try:
        o3d.visualization.draw_geometries(
            [vis_a, vis_b],
            window_name=f"Preflight: frame 0 (red) vs frame {frame_b} (blue)",
            width=1280,
            height=720,
        )
    except Exception:
        pass


def run_preflight_frame0_12(
    session: ScanSession,
    root: Path,
    geom: TurntableGeometry,
    *,
    meta,
    mask_spec,
    include,
    bg_state,
    preprocess_depth: bool,
    frame_rotations_deg: list[float] | None,
    stride: int = DEPTH_STRIDE,
    log=print,
) -> None:
    """Project frame 0 and ~180 deg partner into object frame; abort if misaligned."""
    n_frames = len(session.frames)
    frame_b = _preflight_partner_index(n_frames, frame_rotations_deg)
    if frame_b is None:
        log("  Preflight: skipped (need frame 12 or a ~180 deg pair)")
        return

    log(f"  Preflight: frame 0 vs frame {frame_b} (object frame)...")
    cloud0 = _frame_cloud_object_frame(
        session,
        root,
        0,
        geom,
        meta=meta,
        mask_spec=mask_spec,
        include=include,
        bg_state=bg_state,
        preprocess_depth=preprocess_depth,
        frame_rotations_deg=frame_rotations_deg,
        stride=stride,
    )
    cloud_b = _frame_cloud_object_frame(
        session,
        root,
        frame_b,
        geom,
        meta=meta,
        mask_spec=mask_spec,
        include=include,
        bg_state=bg_state,
        preprocess_depth=preprocess_depth,
        frame_rotations_deg=frame_rotations_deg,
        stride=stride,
    )
    if len(cloud0.points) < PREFLIGHT_MIN_POINTS or len(cloud_b.points) < PREFLIGHT_MIN_POINTS:
        log("  Preflight: skipped (too few points in one or both clouds)")
        return

    fitness, rmse = _overlap_metrics(cloud0, cloud_b)
    out_dir = root / "mesh_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "preflight_frame0_12.ply"
    combined = _paint_cloud(cloud0, (1.0, 0.35, 0.35)) + _paint_cloud(
        cloud_b, (0.35, 0.55, 1.0)
    )
    o3d.io.write_point_cloud(str(ply_path), combined, print_progress=False)
    log(f"  Preflight: saved {ply_path.name} (fitness={fitness:.3f}, rmse={rmse * 1000:.1f} mm)")

    _show_preflight_window(cloud0, cloud_b, frame_b=frame_b)

    ok = fitness >= PREFLIGHT_MIN_FITNESS and rmse <= PREFLIGHT_MAX_RMSE_M
    if ok:
        log("  Preflight: PASS")
        return

    pivot = geom.object_pivot_3d if geom.object_pivot_3d is not None else geom.pivot_m
    tf = _fusion_helpers()
    rec0 = session.frames[0]
    rec_b = session.frames[frame_b]
    tilt0, rot0 = tf.frame_pose(rec0, 0, meta)
    tilt_b, rot_b = tf.frame_pose(rec_b, frame_b, meta)
    if frame_rotations_deg is not None:
        if len(frame_rotations_deg) > 0:
            rot0 = float(frame_rotations_deg[0])
        if frame_b < len(frame_rotations_deg):
            rot_b = float(frame_rotations_deg[frame_b])

    hub_mm = [round(x * 1000, 1) for x in geom.pivot_m]
    obj_mm = (
        [round(x * 1000, 1) for x in geom.object_pivot_3d]
        if geom.object_pivot_3d is not None
        else None
    )
    log(
        "  Preflight: FAIL - frame clouds do not overlap in object frame.\n"
        f"    pivot_m mm={hub_mm}\n"
        f"    object_pivot_3d mm={obj_mm}\n"
        f"    spin pivot used mm={[round(x * 1000, 1) for x in pivot]}\n"
        f"    frame 0: tilt={tilt0:.2f} deg  rotation={rot0:.2f} deg\n"
        f"    frame {frame_b}: tilt={tilt_b:.2f} deg  rotation={rot_b:.2f} deg\n"
        f"    overlap fitness={fitness:.3f} (need >={PREFLIGHT_MIN_FITNESS})  "
        f"rmse={rmse * 1000:.1f} mm (need <={PREFLIGHT_MAX_RMSE_M * 1000:.1f} mm)\n"
        f"    inspect: {ply_path}"
    )
    raise TurntablePreflightError(
        f"Preflight failed: frame 0 vs {frame_b} misaligned "
        f"(fitness={fitness:.3f}, rmse={rmse * 1000:.1f} mm). "
        f"See {ply_path}"
    )
