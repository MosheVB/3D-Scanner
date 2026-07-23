"""Fast 2D masks for live object-only depth integration."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from scanner.config import LIVE_DEPTH_BAND_MM


def depth_band_mask(
    depth_mm: np.ndarray,
    *,
    center_mm: float,
    band_mm: float = LIVE_DEPTH_BAND_MM,
) -> np.ndarray:
    """True where depth is valid and within ±band_mm of center_mm."""
    z = depth_mm.astype(np.float32)
    valid = z > 0
    lo = max(0.0, center_mm - band_mm)
    hi = center_mm + band_mm
    return valid & (z >= lo) & (z <= hi)


def connected_component_at_seed(
    mask: np.ndarray,
    seed_u: int,
    seed_v: int,
) -> np.ndarray:
    """Keep only the connected component that contains the seed pixel."""
    h, w = mask.shape[:2]
    u = int(np.clip(seed_u, 0, w - 1))
    v = int(np.clip(seed_v, 0, h - 1))
    if not mask[v, u]:
        return np.zeros_like(mask, dtype=bool)

    labels = np.zeros((h, w), dtype=np.int32)
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), labels, connectivity=8)
    if n <= 1:
        return mask
    seed_label = labels[v, u]
    if seed_label <= 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == seed_label


def object_mask_from_click(
    depth_mm: np.ndarray,
    seed_u: int,
    seed_v: int,
    *,
    center_mm: float,
    band_mm: float = LIVE_DEPTH_BAND_MM,
) -> np.ndarray:
    """Depth band around the click, then largest connected blob at the seed."""
    band = depth_band_mask(depth_mm, center_mm=center_mm, band_mm=band_mm)
    return connected_component_at_seed(band, seed_u, seed_v)


def depth_discontinuity_mask(
    depth_mm: np.ndarray,
    *,
    gradient_thresh_mm: float = 8.0,
) -> np.ndarray:
    """Pixels where local depth gradient exceeds threshold (object boundaries)."""
    z = depth_mm.astype(np.float32)
    valid = z > 0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return valid & (grad > gradient_thresh_mm)


def _search_region_mask(
    h: int,
    w: int,
    seed_u: int,
    seed_v: int,
    *,
    cluster_mask: np.ndarray,
    min_radius_px: int = 40,
    expand_ratio: float = 0.35,
) -> np.ndarray:
    """Rectangular ROI around seed cluster for assisted segmentation."""
    ys, xs = np.where(cluster_mask)
    if ys.size == 0:
        region = np.zeros((h, w), dtype=bool)
        r = min_radius_px
        u0, v0 = seed_u, seed_v
        region[
            max(0, v0 - r) : min(h, v0 + r + 1),
            max(0, u0 - r) : min(w, u0 + r + 1),
        ] = True
        return region
    u0, u1 = int(xs.min()), int(xs.max())
    v0, v1 = int(ys.min()), int(ys.max())
    du = int((u1 - u0 + 1) * expand_ratio) + min_radius_px
    dv = int((v1 - v0 + 1) * expand_ratio) + min_radius_px
    region = np.zeros((h, w), dtype=bool)
    region[
        max(0, v0 - dv) : min(h, v1 + dv + 1),
        max(0, u0 - du) : min(w, u1 + du + 1),
    ] = True
    return region


def _region_grow_bounded(
    candidate: np.ndarray,
    seed_mask: np.ndarray,
    seed_u: int,
    seed_v: int,
    *,
    max_iters: int = 30,
) -> np.ndarray:
    """Morphological grow from seed within candidate, blocked by barriers."""
    result = seed_mask & candidate
    if not np.any(result):
        return connected_component_at_seed(candidate, seed_u, seed_v)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = result.astype(np.uint8)
    cand = candidate.astype(np.uint8)
    prev_count = 0
    for _ in range(max_iters):
        work = cv2.dilate(work, kernel, iterations=1)
        work = cv2.bitwise_and(work, cand)
        count = int(np.count_nonzero(work))
        if count == prev_count:
            break
        prev_count = count
    return connected_component_at_seed(work.astype(bool), seed_u, seed_v)


def object_mask_from_seeds(
    depth_mm: np.ndarray,
    seeds: Sequence[tuple[int, int]],
    *,
    center_mm: float,
    band_mm: float = LIVE_DEPTH_BAND_MM,
) -> np.ndarray:
    """Union of depth-cluster blobs at each seed pixel."""
    if not seeds:
        return np.zeros(depth_mm.shape[:2], dtype=bool)
    combined = np.zeros(depth_mm.shape[:2], dtype=bool)
    for seed_u, seed_v in seeds:
        combined |= object_mask_from_click(
            depth_mm, seed_u, seed_v, center_mm=center_mm, band_mm=band_mm
        )
    return combined


def assisted_object_mask_from_click(
    rgb_bgr: np.ndarray | None,
    depth_mm: np.ndarray,
    seed_u: int,
    seed_v: int,
    *,
    center_mm: float,
    band_mm: float = LIVE_DEPTH_BAND_MM,
    canny_low: int = 40,
    canny_high: int = 120,
    depth_grad_thresh_mm: float = 8.0,
) -> np.ndarray:
    """Assisted mask: depth cluster + Canny edges + depth discontinuity in search ROI."""
    seed_cluster = object_mask_from_click(
        depth_mm, seed_u, seed_v, center_mm=center_mm, band_mm=band_mm
    )
    if not np.any(seed_cluster):
        return seed_cluster

    h, w = depth_mm.shape[:2]
    search = _search_region_mask(h, w, seed_u, seed_v, cluster_mask=seed_cluster)
    band = depth_band_mask(depth_mm, center_mm=center_mm, band_mm=band_mm) & search

    if rgb_bgr is not None and rgb_bgr.size > 0:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
    else:
        edges = np.zeros((h, w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_barrier = cv2.dilate(edges, kernel, iterations=1) > 0
    depth_jump = depth_discontinuity_mask(
        depth_mm, gradient_thresh_mm=depth_grad_thresh_mm
    )
    barrier = (edge_barrier | depth_jump) & search
    grown = _region_grow_bounded(band & ~barrier, seed_cluster, seed_u, seed_v)

    if np.count_nonzero(grown) < max(20, np.count_nonzero(seed_cluster) * 0.4):
        return seed_cluster
    return grown


def morph_cleanup_mask(
    mask: np.ndarray,
    *,
    open_iters: int = 1,
    close_iters: int = 2,
) -> np.ndarray:
    """Drop speckles (open) and fill small holes (close)."""
    if not np.any(mask):
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = mask.astype(np.uint8)
    if open_iters > 0:
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if close_iters > 0:
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    return work.astype(bool)


def _ransac_plane_inliers(
    points_m: np.ndarray,
    *,
    distance_m: float = 0.008,
    iterations: int = 200,
    min_inliers: int = 50,
) -> np.ndarray:
    """Boolean inlier mask for the dominant plane (table / turntable)."""
    n = points_m.shape[0]
    out = np.zeros(n, dtype=bool)
    if n < min_inliers:
        return out

    rng = np.random.default_rng(0)
    best_inliers = out
    best_count = 0
    dist_thresh = float(distance_m)

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points_m[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len
        d = -float(np.dot(normal, p0))
        dist = np.abs(points_m @ normal + d)
        inliers = dist < dist_thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    return best_inliers


def _depth_statistical_outlier_mask(
    z_mm: np.ndarray,
    *,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Keep pixels whose depth is within std_ratio σ of the mask median."""
    if z_mm.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(z_mm))
    std = float(np.std(z_mm))
    if std < 1e-3:
        return np.ones(z_mm.shape[0], dtype=bool)
    return np.abs(z_mm - med) <= std_ratio * std


def detect_turntable_platform_mask(
    depth_mm: np.ndarray,
    rgb_bgr: np.ndarray | None,
    intrinsics: Sequence[Sequence[float]],
    *,
    seed_u: int | None = None,
    seed_v: int | None = None,
    row_start_frac: float = 0.46,
    dark_gray_max: int = 90,
    plane_dist_mm: float = 5.0,
) -> np.ndarray:
    """Pixels belonging to the turntable platform (exclude from object scan).

    Revopoint-style rigs treat the plate as a separate static surface: dark matte
    disk in the lower frame plus a dominant plane in depth. This mask is True on
    the turntable so callers can subtract it from the object include mask.
    """
    h, w = depth_mm.shape[:2]
    su = int(w // 2 if seed_u is None else np.clip(seed_u, 0, w - 1))
    sv = int(h // 2 if seed_v is None else np.clip(seed_v, 0, h - 1))

    row0 = max(int(h * row_start_frac), sv + 8)
    lower = np.zeros((h, w), dtype=bool)
    lower[row0:, :] = True

    plate = np.zeros((h, w), dtype=bool)

    if rgb_bgr is not None and rgb_bgr.size > 0:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        plate |= lower & (gray < dark_gray_max)

    k = np.array(intrinsics, dtype=np.float64)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    ys, xs = np.where(lower & (depth_mm > 0))
    if ys.size >= 120:
        z_mm = depth_mm[ys, xs].astype(np.float64)
        z = z_mm / 1000.0
        u = xs.astype(np.float64)
        v = ys.astype(np.float64)
        x = (u - cx) * z / fx
        y = -(v - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        inliers = _ransac_plane_inliers(
            pts, distance_m=plane_dist_mm / 1000.0, min_inliers=80
        )
        if int(np.count_nonzero(inliers)) >= 80:
            plate[ys[inliers], xs[inliers]] = True

    seed_z = float(depth_mm[sv, su])
    if seed_z > 0:
        # Turntable ring is usually nearer than the object face or a separate slab below it.
        zf = depth_mm.astype(np.float32)
        near_plate = lower & (zf > 0) & (zf < seed_z + 8.0)
        far_plate = lower & (zf > 0) & (zf > seed_z + 35.0)
        plate |= near_plate | far_plate

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    plate_u8 = plate.astype(np.uint8) * 255
    plate_u8 = cv2.morphologyEx(plate_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    plate_u8 = cv2.dilate(plate_u8, kernel, iterations=1)
    plate = plate_u8 > 0

    # Keep the disk under the object, not random lower blobs.
    bottom_seed_v = min(h - 4, max(row0 + 4, int(h * 0.88)))
    plate_blob = connected_component_at_seed(plate, su, bottom_seed_v)
    if np.count_nonzero(plate_blob) >= 200:
        plate = plate_blob

    return morph_cleanup_mask(plate, open_iters=1, close_iters=1)


def exclude_turntable_from_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    rgb_bgr: np.ndarray | None,
    intrinsics: Sequence[Sequence[float]],
    *,
    seed_u: int | None = None,
    seed_v: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (object_mask, turntable_exclusion_mask) with platform removed."""
    h, w = depth_mm.shape[:2]
    su = w // 2 if seed_u is None else seed_u
    sv = h // 2 if seed_v is None else seed_v
    turntable = detect_turntable_platform_mask(
        depth_mm, rgb_bgr, intrinsics, seed_u=su, seed_v=sv
    )
    cleaned = mask & ~turntable
    if not np.any(cleaned):
        cleaned = connected_component_at_seed(mask, su, sv)
        cleaned &= ~turntable
    if not np.any(cleaned):
        cleaned = mask & ~turntable
    return morph_cleanup_mask(cleaned), turntable


def reject_background_in_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    seed_uvs: Sequence[tuple[int, int]],
    *,
    plane_dist_mm: float = 8.0,
    behind_margin_mm: float = 12.0,
) -> np.ndarray:
    """Inside-cube background rejection: plane table + seed-connected cluster + cleanup."""
    if not np.any(mask) or not seed_uvs:
        return mask

    h, w = depth_mm.shape[:2]
    ys, xs = np.where(mask)
    z_mm = depth_mm[ys, xs].astype(np.float64)
    valid = z_mm > 0
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool)

    ys = ys[valid]
    xs = xs[valid]
    z_mm = z_mm[valid]

    seed_depths = []
    for su, sv in seed_uvs:
        su = int(np.clip(su, 0, w - 1))
        sv = int(np.clip(sv, 0, h - 1))
        z = float(depth_mm[sv, su])
        if z > 0:
            seed_depths.append(z)
    if not seed_depths:
        return morph_cleanup_mask(mask)
    seed_z = float(np.median(seed_depths))

    k = np.array(intrinsics, dtype=np.float64)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    z = z_mm / 1000.0
    u = xs.astype(np.float64)
    v = ys.astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    keep = np.ones(pts.shape[0], dtype=bool)
    if pts.shape[0] >= 80:
        plane_inliers = _ransac_plane_inliers(
            pts, distance_m=plane_dist_mm / 1000.0, min_inliers=40
        )
        if np.count_nonzero(plane_inliers) >= 40:
            # Drop flat table / turntable behind or under the seed depth.
            behind = z_mm > (seed_z - behind_margin_mm)
            keep &= ~(plane_inliers & behind)

    keep &= _depth_statistical_outlier_mask(z_mm, std_ratio=2.5)

    cleaned = np.zeros((h, w), dtype=bool)
    cleaned[ys[keep], xs[keep]] = True

    combined = np.zeros((h, w), dtype=bool)
    for su, sv in seed_uvs:
        blob = connected_component_at_seed(cleaned, su, sv)
        combined |= blob

    if not np.any(combined):
        combined = connected_component_at_seed(cleaned, seed_uvs[0][0], seed_uvs[0][1])
    if not np.any(combined):
        combined = connected_component_at_seed(mask, seed_uvs[0][0], seed_uvs[0][1])

    return morph_cleanup_mask(combined)


def assisted_object_mask_from_seeds(
    rgb_bgr: np.ndarray | None,
    depth_mm: np.ndarray,
    seeds: Sequence[tuple[int, int]],
    *,
    center_mm: float,
    band_mm: float = LIVE_DEPTH_BAND_MM,
    canny_low: int = 40,
    canny_high: int = 120,
    depth_grad_thresh_mm: float = 8.0,
) -> np.ndarray:
    """Union assisted masks from multiple positive seeds."""
    if not seeds:
        return np.zeros(depth_mm.shape[:2], dtype=bool)
    combined = np.zeros(depth_mm.shape[:2], dtype=bool)
    for seed_u, seed_v in seeds:
        combined |= assisted_object_mask_from_click(
            rgb_bgr,
            depth_mm,
            seed_u,
            seed_v,
            center_mm=center_mm,
            band_mm=band_mm,
            canny_low=canny_low,
            canny_high=canny_high,
            depth_grad_thresh_mm=depth_grad_thresh_mm,
        )
    return combined
