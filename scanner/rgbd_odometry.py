"""RGB-D frame-to-frame odometry (Open3D) for turntable and handheld fusion."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from scanner.config import (
    MAX_DEPTH_MM,
    ODOMETRY_DEPTH_DIFF_MAX,
    ODOMETRY_DEPTH_MAX_M,
    ODOMETRY_DEPTH_MIN_M,
    ODOMETRY_DEPTH_STRIDE,
    VOXEL_SIZE_M,
)
from scanner.pointcloud import intrinsics_matrix, rgbd_to_pointcloud


def pinhole_intrinsic(
    intrinsics: Sequence[Sequence[float]],
    width: int,
    height: int,
    *,
    stride: int = 1,
) -> o3d.camera.PinholeCameraIntrinsic:
    k = intrinsics_matrix(intrinsics)
    s = max(1, int(stride))
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(width // s),
        height=int(height // s),
        fx=float(k[0, 0]) / s,
        fy=float(k[1, 1]) / s,
        cx=float(k[0, 2]) / s,
        cy=float(k[1, 2]) / s,
    )


def rgbd_from_arrays(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    *,
    mask: np.ndarray | None = None,
    stride: int = 1,
    depth_min_m: float = ODOMETRY_DEPTH_MIN_M,
    depth_max_m: float = ODOMETRY_DEPTH_MAX_M,
) -> o3d.geometry.RGBDImage:
    """Build an Open3D RGBDImage (RGB uint8, depth meters)."""
    depth = depth_mm.astype(np.float32)
    if mask is not None:
        depth = depth.copy()
        depth[~mask] = 0

    s = max(1, int(stride))
    rgb = rgb_bgr[:, :, ::-1]
    if s > 1:
        rgb = rgb[::s, ::s]
        depth = depth[::s, ::s]

    depth_m = depth / 1000.0
    depth_m[(depth <= 0) | (depth > MAX_DEPTH_MM)] = 0.0
    depth_m[(depth_m < depth_min_m) | (depth_m > depth_max_m)] = 0.0

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8))),
        o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32))),
        depth_scale=1.0,
        depth_trunc=float(depth_max_m),
        convert_rgb_to_intensity=False,
    )


def pose_step_metrics(transform: np.ndarray) -> tuple[float, float]:
    """Return (translation_mm, rotation_deg) from a 4x4 rigid motion step."""
    t = np.asarray(transform, dtype=np.float64)
    trans_mm = float(np.linalg.norm(t[:3, 3]) * 1000.0)
    r = t[:3, :3]
    cos_angle = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    return trans_mm, rot_deg


def odometry_feed_arrays(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    *,
    stride: int = ODOMETRY_DEPTH_STRIDE,
    depth_min_m: float = ODOMETRY_DEPTH_MIN_M,
    depth_max_m: float = ODOMETRY_DEPTH_MAX_M,
) -> tuple[np.ndarray, np.ndarray]:
    """RGB (BGR) and depth (mm, 0=invalid) at the resolution fed to odometry."""
    s = max(1, int(stride))
    rgb = rgb_bgr[::s, ::s]
    depth = depth_mm[::s, ::s].astype(np.float32)
    depth_m = depth / 1000.0
    invalid = (
        (depth <= 0)
        | (depth > MAX_DEPTH_MM)
        | (depth_m < depth_min_m)
        | (depth_m > depth_max_m)
    )
    depth = depth.copy()
    depth[invalid] = 0.0
    return rgb, depth


def _odometry_option() -> o3d.pipelines.odometry.OdometryOption:
    opt = o3d.pipelines.odometry.OdometryOption()
    opt.depth_min = float(ODOMETRY_DEPTH_MIN_M)
    opt.depth_max = float(ODOMETRY_DEPTH_MAX_M)
    opt.depth_diff_max = float(ODOMETRY_DEPTH_DIFF_MAX)
    return opt


def compute_rgbd_odometry_step(
    source: o3d.geometry.RGBDImage,
    target: o3d.geometry.RGBDImage,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    *,
    init: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, dict]:
    """Estimate rigid motion aligning *source* frame into *target* frame."""
    if init is None:
        init = np.eye(4, dtype=np.float64)
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source,
        target,
        intrinsic,
        init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        _odometry_option(),
    )
    return bool(success), np.asarray(trans, dtype=np.float64), info


def chain_odometry_poses(
    rgbd_pairs: Sequence[o3d.geometry.RGBDImage],
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
) -> tuple[list[np.ndarray], list[bool]]:
    """Cumulative poses mapping each camera frame into frame-0 coordinates."""
    if not rgbd_pairs:
        return [], []
    poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    ok_flags: list[bool] = []
    prev = rgbd_pairs[0]
    for rgbd in rgbd_pairs[1:]:
        success, step, _ = compute_rgbd_odometry_step(rgbd, prev, intrinsic)
        ok_flags.append(success)
        if success:
            poses.append(poses[-1] @ step)
        else:
            poses.append(poses[-1].copy())
        prev = rgbd
    return poses, ok_flags


def fuse_rgbd_sequence(
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: Sequence[Sequence[float]],
    *,
    masks: Sequence[np.ndarray | None] | None = None,
    stride: int = ODOMETRY_DEPTH_STRIDE,
    min_depth_mm: int = 0,
    max_depth_mm: int = MAX_DEPTH_MM,
    voxel_m: float = VOXEL_SIZE_M,
) -> tuple[o3d.geometry.PointCloud, list[np.ndarray], list[bool]]:
    """Fuse RGB-D frames using chained odometry into frame-0 coordinates."""
    if not frames:
        return o3d.geometry.PointCloud(), [], []

    h, w = frames[0][1].shape[:2]
    intrinsic = pinhole_intrinsic(intrinsics, w, h, stride=stride)
    rgbd_list: list[o3d.geometry.RGBDImage] = []
    for i, (rgb, depth) in enumerate(frames):
        mask = None if masks is None else masks[i]
        rgbd_list.append(
            rgbd_from_arrays(rgb, depth, intrinsic, mask=mask, stride=stride)
        )

    poses, ok_flags = chain_odometry_poses(rgbd_list, intrinsic)
    merged = o3d.geometry.PointCloud()
    for i, (rgb, depth) in enumerate(frames):
        mask = None if masks is None else masks[i]
        cloud = rgbd_to_pointcloud(
            rgb,
            depth,
            intrinsics,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            stride=stride,
        )
        if mask is not None and len(cloud.points) > 0:
            # Drop points outside mask after unprojection.
            k = intrinsics_matrix(intrinsics)
            pts = np.asarray(cloud.points)
            fx, fy = k[0, 0], k[1, 1]
            cx, cy = k[0, 2], k[1, 2]
            z = pts[:, 2]
            u = np.rint(pts[:, 0] * fx / z + cx).astype(np.int32)
            v = np.rint(-pts[:, 1] * fy / z + cy).astype(np.int32)
            mh, mw = depth.shape
            keep = (u >= 0) & (u < mw) & (v >= 0) & (v < mh) & mask[v, u]
            if np.any(keep):
                filtered = o3d.geometry.PointCloud()
                filtered.points = o3d.utility.Vector3dVector(pts[keep])
                filtered.colors = o3d.utility.Vector3dVector(
                    np.asarray(cloud.colors)[keep]
                )
                cloud = filtered
            else:
                continue
        if len(cloud.points) == 0:
            continue
        cloud.transform(poses[i])
        merged += cloud

    if len(merged.points) > 0:
        merged = merged.voxel_down_sample(voxel_m)
        merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_m * 4, max_nn=30
            )
        )
    return merged, poses, ok_flags
