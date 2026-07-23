"""Turntable kinematics in the fixed camera frame.

The BLE turntable reports absolute tilt and incremental rotation. Axes are
estimated once at the reference pose (usually tilt=0) from the plate plane in
depth, then stored in ``turntable_geometry.json``. They are fixed in camera
coordinates — so a tilt motor that is physically left-right vs up-down is
whatever plane normal / tangent the calibration observes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TurntableGeometry:
    """Axes and pivot in camera frame (meters, Y-up matching pointcloud.py)."""

    version: int = 1
    pivot_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.35])
    spin_axis: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    tilt_axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    tilt_ref_deg: float = 0.0
    rot_ref_deg: float = 0.0
    method: str = "plane_fit"
    object_pivot_3d: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurntableGeometry:
        raw_obj_pivot = data.get("object_pivot_3d")
        return cls(
            version=int(data.get("version", 1)),
            pivot_m=list(data.get("pivot_m") or [0.0, 0.0, 0.35]),
            spin_axis=list(data.get("spin_axis") or [0.0, 1.0, 0.0]),
            tilt_axis=list(data.get("tilt_axis") or [1.0, 0.0, 0.0]),
            tilt_ref_deg=float(data.get("tilt_ref_deg", 0.0)),
            rot_ref_deg=float(data.get("rot_ref_deg", 0.0)),
            method=str(data.get("method", "plane_fit")),
            object_pivot_3d=(
                list(raw_obj_pivot) if raw_obj_pivot is not None else None
            ),
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(np.float64)
    return (v / n).astype(np.float64)


def rodrigues_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    kx, ky, kz = axis
    k = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return eye + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def _unproject_points(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 3,
) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_mm.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    uu, vv = np.meshgrid(xs, ys)
    z_mm = depth_mm[vv, uu]
    keep = mask[vv, uu] & (z_mm > 0)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)
    z = z_mm[keep].astype(np.float64) / 1000.0
    u = uu[keep].astype(np.float64)
    v = vv[keep].astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def geometry_from_platter_hub(
    pivot_m: np.ndarray | Sequence[float],
    *,
    tilt_ref_deg: float = 0.0,
    rot_ref_deg: float = 0.0,
) -> TurntableGeometry:
    """Fixed vertical spin axis through measured platter hub (dot tracking)."""
    return TurntableGeometry(
        pivot_m=list(np.asarray(pivot_m, dtype=np.float64).ravel()[:3]),
        spin_axis=[0.0, 1.0, 0.0],
        tilt_axis=[1.0, 0.0, 0.0],
        tilt_ref_deg=tilt_ref_deg,
        rot_ref_deg=rot_ref_deg,
        method="platter_dot_track",
    )


def analytical_geometry_from_frame0(
    depth_mm: np.ndarray,
    include_mask: np.ndarray,
    intrinsics: np.ndarray | list[list[float]],
    *,
    tilt_ref_deg: float = 0.0,
    rot_ref_deg: float = 0.0,
) -> TurntableGeometry:
    """Fixed camera: vertical spin through frame-0 depth centroid (no plane fit)."""
    pivot = object_center_from_seed(depth_mm, include_mask, intrinsics)
    return TurntableGeometry(
        pivot_m=pivot.tolist(),
        spin_axis=[0.0, 1.0, 0.0],
        tilt_axis=[1.0, 0.0, 0.0],
        tilt_ref_deg=tilt_ref_deg,
        rot_ref_deg=rot_ref_deg,
        method="analytical_frame0_centroid",
    )


def estimate_geometry_from_seed(
    depth_mm: np.ndarray,
    include_mask: np.ndarray,
    intrinsics: np.ndarray | list[list[float]],
    *,
    tilt_ref_deg: float = 0.0,
    rot_ref_deg: float = 0.0,
) -> TurntableGeometry:
    """Fit turntable plate from lower image band (ring), not object front face."""
    k = np.array(intrinsics, dtype=np.float64)
    h, w = depth_mm.shape
    # Turntable ring / table: lower portion of frame inside mask.
    row_cut = int(h * 0.52)
    plate_region = include_mask.copy()
    plate_region[:row_cut, :] = False
    plate_region &= depth_mm > 0
    if int(plate_region.sum()) < 120:
        plate_region = include_mask & (depth_mm > 0)

    pts = _unproject_points(depth_mm, plate_region, k, stride=2)
    if pts.shape[0] < 120:
        pts = _unproject_points(depth_mm, include_mask & (depth_mm > 0), k, stride=3)
    if pts.shape[0] < 80:
        raise ValueError(f"Too few points for turntable geometry ({pts.shape[0]}).")

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _unit(vh[2])

    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Rotation axis is platform normal; prefer near-vertical in camera frame.
    if abs(float(normal[1])) < 0.45:
        spin = _unit(cam_up * 0.65 + normal * 0.35)
    else:
        spin = _unit(normal)
        if spin[1] < 0:
            spin = -spin

    tilt = _unit(np.cross(spin, cam_fwd))
    if float(np.linalg.norm(tilt)) < 0.2:
        tilt = _unit(np.cross(spin, cam_up))
    if float(np.linalg.norm(tilt)) < 0.2:
        tilt = _unit(np.array([1.0, 0.0, 0.0]))

    pivot = centroid.copy()
    pivot[1] = float(np.percentile(pts[:, 1], 20))

    return TurntableGeometry(
        pivot_m=pivot.tolist(),
        spin_axis=spin.tolist(),
        tilt_axis=tilt.tolist(),
        tilt_ref_deg=tilt_ref_deg,
        rot_ref_deg=rot_ref_deg,
        method="plane_fit",
    )


def masked_depth_centroid_3d(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray | list[list[float]],
    *,
    stride: int = 2,
    min_points: int = 30,
) -> np.ndarray | None:
    """Mean 3D position (meters, Y-up) of masked valid depth pixels."""
    k = np.array(intrinsics, dtype=np.float64)
    pts = _unproject_points(depth_mm, mask & (depth_mm > 0), k, stride=stride)
    if pts.shape[0] < min_points:
        return None
    return np.mean(pts, axis=0).astype(np.float64)


def object_center_from_seed(
    depth_mm: np.ndarray,
    include_mask: np.ndarray,
    intrinsics: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Object centroid in camera frame (meters) for voxel grid centering."""
    k = np.array(intrinsics, dtype=np.float64)
    pts = _unproject_points(depth_mm, include_mask & (depth_mm > 0), k, stride=3)
    if pts.shape[0] < 30:
        raise ValueError("Too few points to locate object center.")
    return np.median(pts, axis=0).astype(np.float64)


def object_extent_from_seed(
    depth_mm: np.ndarray,
    include_mask: np.ndarray,
    intrinsics: np.ndarray | list[list[float]],
    *,
    margin: float = 1.35,
    min_m: float = 0.28,
    max_m: float = 0.70,
) -> float:
    k = np.array(intrinsics, dtype=np.float64)
    pts = _unproject_points(depth_mm, include_mask & (depth_mm > 0), k, stride=3)
    if pts.shape[0] < 50:
        return min_m
    span = float(np.max(np.ptp(pts, axis=0)))
    return float(np.clip(span * margin, min_m, max_m))


def rotation_cam_to_object(
    geom: TurntableGeometry,
    *,
    tilt_deg: float,
    rotation_deg: float,
) -> np.ndarray:
    """3x3 R such that p_obj = R @ p_cam + t (column vectors)."""
    pivot = np.array(geom.pivot_m, dtype=np.float64)
    spin_ref = _unit(np.array(geom.spin_axis, dtype=np.float64))
    tilt_ax = _unit(np.array(geom.tilt_axis, dtype=np.float64))

    dt = np.deg2rad(tilt_deg - geom.tilt_ref_deg)
    dr = np.deg2rad(rotation_deg - geom.rot_ref_deg)

    r_tilt = rodrigues_matrix(tilt_ax, dt)
    spin_cur = _unit(r_tilt @ spin_ref)
    r_spin = rodrigues_matrix(spin_cur, dr)

    # Object moved forward: p_cam = r_spin @ r_tilt @ (p_obj - pivot) + pivot
    r_fwd = r_spin @ r_tilt
    return r_fwd.T


def transform_cam_to_object(
    geom: TurntableGeometry,
    *,
    tilt_deg: float,
    rotation_deg: float,
) -> np.ndarray:
    """4x4 T mapping camera-frame points to object reference frame."""
    pivot = np.array(geom.pivot_m, dtype=np.float64)
    r = rotation_cam_to_object(geom, tilt_deg=tilt_deg, rotation_deg=rotation_deg)
    t = pivot - r @ pivot
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def save_turntable_geometry(path: Path, geom: TurntableGeometry) -> None:
    path.write_text(
        json.dumps(geom.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_turntable_geometry(path: Path) -> TurntableGeometry | None:
    if not path.is_file():
        return None
    return TurntableGeometry.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
