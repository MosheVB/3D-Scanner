"""OBB face parallelism metrics and rotation-pivot X/Z refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from scanner.turntable_pose import TurntableGeometry


@dataclass
class FaceParallelismReport:
    """Angular errors (deg) between opposite vertical face pairs."""

    pair_a_deg: float
    pair_b_deg: float
    max_deg: float
    vertical_normals: list[list[float]]

    @property
    def ok(self) -> bool:
        return self.max_deg < 3.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(np.float64)
    return (v / n).astype(np.float64)


def _obb_face_normals(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """Six face normals from the oriented bounding box (rows, unit vectors)."""
    if len(cloud.points) < 50:
        return np.eye(3)
    obb = cloud.get_oriented_bounding_box()
    r = np.asarray(obb.R, dtype=np.float64)
    return np.stack([_unit(r[:, i]) for i in range(3)], axis=0)


def _vertical_face_pairs(
    normals: np.ndarray,
    spin_axis: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Return two opposite pairs of vertical face normals (perpendicular to spin axis)."""
    spin = _unit(np.asarray(spin_axis if spin_axis is not None else [0.0, 1.0, 0.0]))
    # Each axis contributes +n and -n faces; keep normals mostly horizontal.
    candidates: list[np.ndarray] = []
    for n in normals:
        candidates.append(_unit(n))
        candidates.append(_unit(-n))
    vertical = [n for n in candidates if abs(float(np.dot(n, spin))) < 0.45]
    if len(vertical) < 4:
        # Fallback: pick the four normals farthest from spin axis.
        scored = sorted(
            candidates,
            key=lambda n: abs(float(np.dot(n, spin))),
        )
        vertical = scored[:4]
    if len(vertical) < 2:
        return None

    used: set[int] = set()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, n1 in enumerate(vertical):
        if i in used:
            continue
        best_j = -1
        best_score = -1.0
        for j, n2 in enumerate(vertical):
            if j == i or j in used:
                continue
            score = float(np.dot(n1, n2))
            if score < best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            pairs.append((n1, vertical[best_j]))
            used.add(i)
            used.add(best_j)
        if len(pairs) == 2:
            break
    if len(pairs) < 2:
        return None
    return pairs[0], pairs[1]


def opposite_face_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle (deg) between face normals that should be anti-parallel on a box."""
    cos_a = float(np.clip(np.dot(_unit(n1), _unit(-n2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def measure_face_parallelism(
    cloud: o3d.geometry.PointCloud,
    *,
    spin_axis: np.ndarray | None = None,
) -> FaceParallelismReport:
    """Measure how parallel opposite vertical OBB faces are."""
    normals = _obb_face_normals(cloud)
    pairs = _vertical_face_pairs(normals, spin_axis=spin_axis)
    if pairs is None:
        return FaceParallelismReport(
            pair_a_deg=90.0,
            pair_b_deg=90.0,
            max_deg=90.0,
            vertical_normals=[],
        )
    (a1, a2), (b1, b2) = pairs
    err_a = opposite_face_angle_deg(a1, a2)
    err_b = opposite_face_angle_deg(b1, b2)
    verts = [a1.tolist(), a2.tolist(), b1.tolist(), b2.tolist()]
    return FaceParallelismReport(
        pair_a_deg=err_a,
        pair_b_deg=err_b,
        max_deg=max(err_a, err_b),
        vertical_normals=verts,
    )


def pivot_with_xz_offset(geom: TurntableGeometry, dx_m: float, dz_m: float) -> TurntableGeometry:
    pivot = list(geom.pivot_m)
    pivot[0] = float(pivot[0]) + float(dx_m)
    pivot[2] = float(pivot[2]) + float(dz_m)
    return TurntableGeometry(
        version=geom.version,
        pivot_m=pivot,
        spin_axis=list(geom.spin_axis),
        tilt_axis=list(geom.tilt_axis),
        tilt_ref_deg=geom.tilt_ref_deg,
        rot_ref_deg=geom.rot_ref_deg,
        method=geom.method,
        object_pivot_3d=geom.object_pivot_3d,
    )


def refine_pivot_xz_offset(
    geom: TurntableGeometry,
    fuse_fn,
    *,
    search_range_m: float = 0.015,
    step_m: float = 0.005,
    fine_range_m: float = 0.0,
    fine_step_m: float = 0.002,
    log=print,
) -> tuple[TurntableGeometry, FaceParallelismReport]:
    """Grid-search pivot X/Z offset until opposite vertical faces are most parallel.

    *fuse_fn* receives an adjusted :class:`TurntableGeometry` and returns a fused
  point cloud (camera / object frame, pre-segmentation).
    """
    best_geom = geom
    best_report = measure_face_parallelism(
        fuse_fn(geom),
        spin_axis=np.asarray(geom.spin_axis, dtype=np.float64),
    )
    best_score = best_report.max_deg
    log(
        f"  Face parallelism (initial): "
        f"pairA={best_report.pair_a_deg:.2f} deg  "
        f"pairB={best_report.pair_b_deg:.2f} deg"
    )

    steps = int(round(search_range_m / step_m))
    for ix in range(-steps, steps + 1):
        for iz in range(-steps, steps + 1):
            dx = ix * step_m
            dz = iz * step_m
            if abs(dx) < 1e-9 and abs(dz) < 1e-9:
                continue
            trial = pivot_with_xz_offset(geom, dx, dz)
            cloud = fuse_fn(trial)
            report = measure_face_parallelism(
                cloud,
                spin_axis=np.asarray(trial.spin_axis, dtype=np.float64),
            )
            if report.max_deg < best_score:
                best_score = report.max_deg
                best_geom = trial
                best_report = report

    # Fine search around the coarse winner (optional).
    if fine_range_m > 0.0 and fine_step_m > 0.0:
        fine_steps = int(round(fine_range_m / fine_step_m))
        center = np.asarray(best_geom.pivot_m, dtype=np.float64)
        base = np.asarray(geom.pivot_m, dtype=np.float64)
        for ix in range(-fine_steps, fine_steps + 1):
            for iz in range(-fine_steps, fine_steps + 1):
                dx = (center[0] - base[0]) + ix * fine_step_m
                dz = (center[2] - base[2]) + iz * fine_step_m
                trial = pivot_with_xz_offset(geom, dx, dz)
                cloud = fuse_fn(trial)
                report = measure_face_parallelism(
                    cloud,
                    spin_axis=np.asarray(trial.spin_axis, dtype=np.float64),
                )
                if report.max_deg < best_score:
                    best_score = report.max_deg
                    best_geom = trial
                    best_report = report

    if best_geom is not geom:
        p0 = np.asarray(geom.pivot_m)
        p1 = np.asarray(best_geom.pivot_m)
        log(
            f"  Pivot X/Z refined: "
            f"dx={(p1[0]-p0[0])*1000:.1f} mm  dz={(p1[2]-p0[2])*1000:.1f} mm  "
            f"max face error {best_report.max_deg:.2f} deg"
        )
    else:
        log(f"  Pivot X/Z unchanged (best max face error {best_report.max_deg:.2f} deg)")

    if not best_report.ok:
        log(
            "  WARNING: opposite vertical faces are still not parallel - "
            "rotation axis offset may need a wider search or mask."
        )
    return best_geom, best_report
