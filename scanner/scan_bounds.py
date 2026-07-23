"""Dynamic 3D bounding-cube tracking for web scan segmentation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, Sequence

BoundsPhase = Literal[
    "none",
    "image_contour",
    "image_obb",
    "image_ellipse",
    "image_square",
    "depth_cube",
]
ShapeSource = Literal["haiku", "depth", "clicks"]
ImageOverlayPhase = Literal["image_contour", "image_obb", "image_ellipse", "image_square"]

IMAGE_RECT_PADDING_PX = 24
IMAGE_SINGLE_CLICK_PAD_PX = 28
CONTOUR_COLOR_BGR = (0, 255, 255)  # yellow on BGR

import cv2
import numpy as np

from scanner.config import (
    CLICK_PATCH_RADIUS,
    CLICK_SEARCH_MAX_RADIUS,
    LIVE_DEPTH_BAND_MM,
)
from scanner.object_mask import (
    assisted_object_mask_from_click,
    assisted_object_mask_from_seeds,
    morph_cleanup_mask,
    object_mask_from_click,
    object_mask_from_seeds,
    reject_background_in_mask,
)
from scanner.pointcloud import intrinsics_matrix


def sample_depth_at_click(depth_mm: np.ndarray, u: int, v: int) -> float | None:
    """Depth near (u,v): growing patch median (same strategy as capture.py)."""
    h, w = depth_mm.shape[:2]
    u0 = int(np.clip(u, 0, w - 1))
    v0 = int(np.clip(v, 0, h - 1))

    for radius in range(CLICK_PATCH_RADIUS, CLICK_SEARCH_MAX_RADIUS + 1, 2):
        r = int(radius)
        v_lo, v_hi = max(0, v0 - r), min(h, v0 + r + 1)
        u_lo, u_hi = max(0, u0 - r), min(w, u0 + r + 1)
        patch = depth_mm[v_lo:v_hi, u_lo:u_hi]
        ys, xs = np.where(patch > 0)
        if ys.size == 0:
            continue
        valid = patch[patch > 0]
        if valid.size >= 3:
            return float(np.median(valid))

    return None


def adaptive_depth_band_mm(
    center_mm: float,
    *,
    floor_mm: float = 35.0,
    ceiling_mm: float = 120.0,
    ratio: float = 0.12,
) -> float:
    """Depth band scaled to object distance (±ratio of Z, clamped)."""
    return float(np.clip(center_mm * ratio, floor_mm, ceiling_mm))


def uv_from_display_click(
    client_x: float,
    client_y: float,
    *,
    rect_left: float,
    rect_top: float,
    rect_width: float,
    rect_height: float,
    natural_width: int,
    natural_height: int,
) -> tuple[int, int] | None:
    """Map a browser click to image (u,v) with object-fit:contain letterboxing."""
    nw = max(1, int(natural_width))
    nh = max(1, int(natural_height))
    scale = min(rect_width / nw, rect_height / nh)
    disp_w = nw * scale
    disp_h = nh * scale
    offset_x = rect_left + (rect_width - disp_w) * 0.5
    offset_y = rect_top + (rect_height - disp_h) * 0.5
    x = client_x - offset_x
    y = client_y - offset_y
    if x < 0 or y < 0 or x > disp_w or y > disp_h:
        return None
    u = int(round(x / scale))
    v = int(round(y / scale))
    return int(np.clip(u, 0, nw - 1)), int(np.clip(v, 0, nh - 1))


@dataclass
class ScanBoundsSettings:
    margin_mm: float = 15.0
    expand_margin_mm: float = 20.0
    outside_uv_tolerance_px: float = 8.0
    classify_inside_margin_mm: float = 3.0
    smooth_alpha: float = 0.3
    default_half_extent_mm: float = 40.0
    min_half_extent_mm: float = 15.0
    max_half_extent_mm: float = 120.0
    search_expand_ratio: float = 0.15
    shrink_alpha: float = 0.12
    depth_band_mm: float = LIVE_DEPTH_BAND_MM
    edge_canny_low: int = 40
    edge_canny_high: int = 120
    expand_hold_frames: int = 20


@dataclass
class ScanBounds:
    """Axis-aligned bounding cube in camera frame (meters)."""

    center_m: np.ndarray
    half_extents_m: np.ndarray
    depth_min_mm: float
    depth_max_mm: float
    initialized: bool = True

    def contains_points_m(self, points_m: np.ndarray, *, margin_m: float = 0.0) -> np.ndarray:
        if points_m.size == 0:
            return np.array([], dtype=bool)
        lo = self.center_m - self.half_extents_m - margin_m
        hi = self.center_m + self.half_extents_m + margin_m
        return np.all((points_m >= lo) & (points_m <= hi), axis=1)

    def expanded(self, ratio: float) -> ScanBounds:
        scale = 1.0 + ratio
        return ScanBounds(
            center_m=self.center_m.copy(),
            half_extents_m=self.half_extents_m * scale,
            depth_min_mm=self.depth_min_mm,
            depth_max_mm=self.depth_max_mm,
            initialized=True,
        )

    def to_status_dict(self) -> dict[str, Any]:
        c = self.center_m
        h = self.half_extents_m
        wx = round(float(h[0]) * 2000, 1)
        hy = round(float(h[1]) * 2000, 1)
        dz = round(float(h[2]) * 2000, 1)
        return {
            "center_mm": {
                "x": round(float(c[0]) * 1000, 1),
                "y": round(float(c[1]) * 1000, 1),
                "z": round(float(c[2]) * 1000, 1),
            },
            "half_extents_mm": {
                "x": round(float(h[0]) * 1000, 1),
                "y": round(float(h[1]) * 1000, 1),
                "z": round(float(h[2]) * 1000, 1),
            },
            "dimensions_mm": {"w": wx, "h": hy, "d": dz},
            "depth_min_mm": round(self.depth_min_mm, 1),
            "depth_max_mm": round(self.depth_max_mm, 1),
            "initialized": self.initialized,
        }


def unproject_uv_to_m(
    u: float,
    v: float,
    depth_mm: float,
    intrinsics: Sequence[Sequence[float]],
) -> np.ndarray:
    k = intrinsics_matrix(intrinsics)
    z = depth_mm / 1000.0
    x = (u - k[0, 2]) * z / k[0, 0]
    y = -(v - k[1, 2]) * z / k[1, 1]
    return np.array([x, y, z], dtype=np.float64)


def _clamp_half_extents_m(
    half_m: np.ndarray,
    *,
    min_mm: float,
    max_mm: float,
) -> np.ndarray:
    lo = min_mm / 1000.0
    hi = max_mm / 1000.0
    return np.clip(half_m, lo, hi)


def _aabb_from_points_m(points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = points_m.min(axis=0)
    hi = points_m.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half, 1e-4)
    return center, half


def _aabb_from_points_m_robust(
    points_m: np.ndarray,
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    if points_m.shape[0] < 8:
        return _aabb_from_points_m(points_m)
    lo = np.percentile(points_m, lo_pct, axis=0)
    hi = np.percentile(points_m, hi_pct, axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half, 1e-4)
    return center, half


def _uv_rect_from_clicks(
    clicks: Sequence[tuple[int, int]],
    h: int,
    w: int,
    *,
    padding_px: int = IMAGE_RECT_PADDING_PX,
    depth_mm: np.ndarray | None = None,
    intrinsics: Sequence[Sequence[float]] | None = None,
) -> tuple[int, int, int, int]:
    """Inclusive (u0, v0, u1, v1) axis-aligned hull of include clicks (not forced square)."""
    if not clicks:
        return 0, 0, w - 1, h - 1
    pad = padding_px if len(clicks) > 1 else IMAGE_SINGLE_CLICK_PAD_PX
    us = [c[0] for c in clicks]
    vs = [c[1] for c in clicks]
    u0 = max(0, min(us) - pad)
    v0 = max(0, min(vs) - pad)
    u1 = min(w - 1, max(us) + pad)
    v1 = min(h - 1, max(vs) + pad)
    if depth_mm is not None and intrinsics is not None and len(clicks) >= 1:
        sil = _depth_silhouette_bounds_uv(
            depth_mm, clicks[0], intrinsics, base_rect=(u0, v0, u1, v1)
        )
        if sil is not None:
            su0, sv0, su1, sv1 = sil
            u0, v0 = min(u0, su0), min(v0, sv0)
            u1, v1 = max(u1, su1), max(v1, sv1)
    return u0, v0, u1, v1


def _depth_silhouette_bounds_uv(
    depth_mm: np.ndarray,
    seed_uv: tuple[int, int],
    intrinsics: Sequence[Sequence[float]],
    *,
    base_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Expand click hull using depth cluster extent at the seed."""
    u, v = seed_uv
    d = sample_depth_at_click(depth_mm, u, v)
    if d is None:
        return None
    band_mm = adaptive_depth_band_mm(d)
    cluster = object_mask_from_click(depth_mm, u, v, center_mm=d, band_mm=band_mm)
    if not np.any(cluster):
        return None
    ys, xs = np.where(cluster)
    u0, v0, u1, v1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pad = IMAGE_SINGLE_CLICK_PAD_PX
    h, w = depth_mm.shape[:2]
    return (
        max(0, min(base_rect[0], u0 - pad)),
        max(0, min(base_rect[1], v0 - pad)),
        min(w - 1, max(base_rect[2], u1 + pad)),
        min(h - 1, max(base_rect[3], v1 + pad)),
    )


def _hull_mask_from_clicks(
    clicks: Sequence[tuple[int, int]],
    h: int,
    w: int,
    *,
    rect_uv: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Boolean mask: convex hull of clicks, or AABB rect when < 3 clicks."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not clicks:
        mask[:, :] = 1
        return mask.astype(bool)
    if len(clicks) < 3:
        if rect_uv is None:
            rect_uv = _uv_rect_from_clicks(clicks, h, w)
        u0, v0, u1, v1 = rect_uv
        mask[v0 : v1 + 1, u0 : u1 + 1] = 1
        return mask.astype(bool)
    pts = np.array([[c[0], c[1]] for c in clicks], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def _contour_from_depth_edges(
    rgb_bgr: np.ndarray | None,
    depth_mm: np.ndarray,
    region_mask: np.ndarray,
    *,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Depth silhouette + Canny edges within region → refined mask and contour UVs."""
    h, w = depth_mm.shape[:2]
    if not np.any(region_mask):
        return region_mask, []

    work = region_mask.copy()
    z = depth_mm.astype(np.float32)
    valid = (z > 0) & work
    if np.any(valid):
        med = float(np.median(z[valid]))
        band = adaptive_depth_band_mm(med)
        sil = valid & (z >= med - band) & (z <= med + band)
        if np.count_nonzero(sil) >= 20:
            work = sil

    if rgb_bgr is not None and rgb_bgr.size > 0:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
        depth_edge = (np.sqrt(gx * gx + gy * gy) > 8.0) & valid
        edge_band = (edges > 0) | depth_edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_band = cv2.dilate(edge_band.astype(np.uint8), kernel, iterations=1).astype(bool)
        grown = cv2.dilate(work.astype(np.uint8), kernel, iterations=2).astype(bool)
        trimmed = grown & ~edge_band
        if np.count_nonzero(trimmed) >= max(20, np.count_nonzero(work) * 0.25):
            work = trimmed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = cv2.morphologyEx(work.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    work = work.astype(bool)

    contours, _ = cv2.findContours(
        work.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return work, []
    largest = max(contours, key=cv2.contourArea)
    contour_uv = [(int(p[0][0]), int(p[0][1])) for p in largest]
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, cv2.FILLED)
    return filled.astype(bool), contour_uv


def build_adaptive_shape_mask(
    rgb_bgr: np.ndarray | None,
    depth_mm: np.ndarray,
    include_uvs: Sequence[tuple[int, int]],
    intrinsics: Sequence[Sequence[float]],
    settings: ScanBoundsSettings,
    *,
    haiku_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], ShapeSource]:
    """Adaptive non-square 2D region: Haiku mask, depth contour, or click hull."""
    h, w = depth_mm.shape[:2]
    if haiku_mask is not None and np.any(haiku_mask):
        mask = haiku_mask.copy()
        if exclude_mask is not None and np.any(exclude_mask):
            mask = mask & ~exclude_mask
        from scanner.mask_haiku import contour_points_from_mask

        return mask, contour_points_from_mask(mask), "haiku"

    rect = _uv_rect_from_clicks(
        include_uvs, h, w, depth_mm=depth_mm, intrinsics=intrinsics
    )
    hull = _hull_mask_from_clicks(include_uvs, h, w, rect_uv=rect)

    seed_depths = [
        d
        for u, v in include_uvs
        if (d := sample_depth_at_click(depth_mm, u, v)) is not None
    ]
    mask: np.ndarray | None = None
    contour: list[tuple[int, int]] = []
    if seed_depths:
        center_mm = float(np.median(seed_depths))
        band_mm = adaptive_depth_band_mm(center_mm)
        seeds_mask = object_mask_from_seeds(
            depth_mm,
            include_uvs,
            center_mm=center_mm,
            band_mm=band_mm,
        )
        depth_mask = morph_cleanup_mask(seeds_mask & hull)
        if np.count_nonzero(depth_mask) >= 20:
            mask = depth_mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = [(int(p[0][0]), int(p[0][1])) for p in largest]

    if mask is None or not contour:
        mask, contour = _contour_from_depth_edges(
            rgb_bgr,
            depth_mm,
            hull,
            canny_low=int(settings.edge_canny_low),
            canny_high=int(settings.edge_canny_high),
        )
    if exclude_mask is not None and np.any(exclude_mask):
        mask = mask & ~exclude_mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour = [(int(p[0][0]), int(p[0][1])) for p in largest]
    shape_src: ShapeSource = "depth" if np.count_nonzero(mask) >= 20 else "clicks"
    if shape_src == "clicks":
        contour = [
            (rect[0], rect[1]),
            (rect[2], rect[1]),
            (rect[2], rect[3]),
            (rect[0], rect[3]),
        ]
    return mask, contour, shape_src


def _depth_points_in_uv_rect(
    depth_mm: np.ndarray,
    rect_uv: tuple[int, int, int, int],
    intrinsics: Sequence[Sequence[float]],
    *,
    stride: int = 2,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> np.ndarray:
    """Unproject valid depth samples inside a 2D UV rectangle."""
    u0, v0, u1, v1 = rect_uv
    h, w = depth_mm.shape[:2]
    u0, v0 = max(0, u0), max(0, v0)
    u1, v1 = min(w - 1, u1), min(h - 1, v1)
    if u1 < u0 or v1 < v0:
        return np.zeros((0, 3), dtype=np.float64)
    region = depth_mm[v0 : v1 + 1 : stride, u0 : u1 + 1 : stride]
    ys, xs = np.where(region > 0)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xs_g = xs.astype(np.int64) * stride + u0
    ys_g = ys.astype(np.int64) * stride + v0
    z_mm = depth_mm[ys_g, xs_g].astype(np.float64)
    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    z = z_mm / 1000.0
    u = xs_g.astype(np.float64)
    v = ys_g.astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] >= 8:
        lo = np.percentile(pts, lo_pct, axis=0)
        hi = np.percentile(pts, hi_pct, axis=0)
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        trimmed = pts[inside]
        if trimmed.shape[0] >= 4:
            pts = trimmed
    return pts


def _unproject_clicks_to_points_m(
    click_uvs: Sequence[tuple[int, int]],
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
) -> np.ndarray:
    pts: list[np.ndarray] = []
    for u, v in click_uvs:
        d = sample_depth_at_click(depth_mm, u, v)
        if d is not None:
            pts.append(unproject_uv_to_m(u, v, d, intrinsics))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(pts, axis=0)


def _half_extents_respecting_points(
    half_m: np.ndarray,
    center_m: np.ndarray,
    points_m: np.ndarray,
    settings: ScanBoundsSettings,
    *,
    margin_m: float | None = None,
) -> np.ndarray:
    """Clamp half-extents but never exclude required include-click points."""
    margin = (
        settings.expand_margin_mm / 1000.0 if margin_m is None else margin_m
    )
    lo_min = settings.min_half_extent_mm / 1000.0
    hi_max = settings.max_half_extent_mm / 1000.0
    half = np.maximum(half_m, lo_min)
    if points_m.shape[0] > 0:
        required = np.max(np.abs(points_m - center_m), axis=0) + margin
        half = np.maximum(half, required)
    for i in range(3):
        if half[i] <= hi_max:
            continue
        if points_m.shape[0] == 0:
            half[i] = hi_max
            continue
        need_i = float(np.max(np.abs(points_m[:, i] - center_m[i]) + margin))
        half[i] = max(hi_max, need_i)
    return half


def _bounds_from_points_union(
    points_m: np.ndarray,
    settings: ScanBoundsSettings,
    *,
    margin_mm: float | None = None,
) -> ScanBounds | None:
    if points_m.shape[0] < 1:
        return None
    margin_m = (
        settings.expand_margin_mm / 1000.0
        if margin_mm is None
        else margin_mm / 1000.0
    )
    center, half = _aabb_from_points_m_robust(points_m, lo_pct=5.0, hi_pct=95.0)
    half = half + margin_m
    half = _half_extents_respecting_points(
        half, center, points_m, settings, margin_m=margin_m
    )
    z_mm = points_m[:, 2] * 1000.0
    z_spread = float(np.percentile(z_mm, 95) - np.percentile(z_mm, 5))
    z_margin = max(settings.margin_mm, z_spread * 0.15 + 6.0)
    depth_min = max(0.0, float(np.percentile(z_mm, 5) - z_margin))
    depth_max = float(np.percentile(z_mm, 95) + z_margin)
    return ScanBounds(
        center_m=center,
        half_extents_m=half,
        depth_min_mm=depth_min,
        depth_max_mm=depth_max,
    )


def _all_clicks_inside_bounds(
    bounds: ScanBounds,
    click_uvs: Sequence[tuple[int, int]],
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    margin_mm: float,
) -> bool:
    margin_m = margin_mm / 1000.0
    checked = 0
    for u, v in click_uvs:
        d = sample_depth_at_click(depth_mm, u, v)
        if d is None:
            continue
        checked += 1
        pt = unproject_uv_to_m(u, v, d, intrinsics)
        if not bool(
            bounds.contains_points_m(pt.reshape(1, 3), margin_m=margin_m)[0]
        ):
            return False
    return checked > 0


def _expand_bounds_to_include_point(
    bounds: ScanBounds,
    point_m: np.ndarray,
    settings: ScanBoundsSettings,
) -> tuple[ScanBounds, np.ndarray]:
    margin_m = settings.expand_margin_mm / 1000.0
    lo_old = bounds.center_m - bounds.half_extents_m
    hi_old = bounds.center_m + bounds.half_extents_m
    lo_new = np.minimum(lo_old, point_m - margin_m)
    hi_new = np.maximum(hi_old, point_m + margin_m)
    center = 0.5 * (lo_new + hi_new)
    half = 0.5 * (hi_new - lo_new)
    delta_mm = (half - bounds.half_extents_m) * 1000.0
    half = _half_extents_respecting_points(
        half,
        center,
        point_m.reshape(1, 3),
        settings,
        margin_m=margin_m,
    )
    z_lo = (center[2] - half[2]) * 1000.0
    z_hi = (center[2] + half[2]) * 1000.0
    margin = settings.depth_band_mm * 0.5
    depth_min = min(bounds.depth_min_mm, max(0.0, z_lo - margin))
    depth_max = max(bounds.depth_max_mm, z_hi + margin)
    return ScanBounds(
        center_m=center,
        half_extents_m=half,
        depth_min_mm=depth_min,
        depth_max_mm=depth_max,
        initialized=True,
    ), delta_mm


def _click_outside_projected_aabb(
    click_uv: tuple[int, int],
    bounds: ScanBounds,
    intrinsics: Sequence[Sequence[float]],
    *,
    h: int,
    w: int,
    tolerance_px: float,
) -> bool:
    """True when the click lies outside the cube's image projection (±tolerance)."""
    us, vs = _project_cube_corners(bounds, intrinsics, h=h, w=w)
    if us.size == 0:
        return True
    u, v = click_uv
    tol = float(tolerance_px)
    return bool(
        u < float(us.min()) - tol
        or u > float(us.max()) + tol
        or v < float(vs.min()) - tol
        or v > float(vs.max()) + tol
    )


_CUBE_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)


def _cube_corners_m(bounds: ScanBounds) -> list[np.ndarray]:
    c = bounds.center_m
    h = bounds.half_extents_m
    signs = (
        (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1),
    )
    return [c + h * np.array(s, dtype=np.float64) for s in signs]


def _project_point_m(
    point_m: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    z = float(point_m[2])
    if z <= 1e-6:
        return None
    u = int(round(point_m[0] * fx / z + cx))
    v = int(round(-point_m[1] * fy / z + cy))
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    return u, v


def _draw_cube_wireframe(
    out: np.ndarray,
    bounds: ScanBounds,
    intrinsics: Sequence[Sequence[float]],
    *,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    h, w = out.shape[:2]
    corners = _cube_corners_m(bounds)
    projected: list[tuple[int, int] | None] = [
        _project_point_m(pt, intrinsics, h=h, w=w) for pt in corners
    ]
    for i, j in _CUBE_EDGES:
        pi, pj = projected[i], projected[j]
        if pi is not None and pj is not None:
            cv2.line(out, pi, pj, color, thickness, cv2.LINE_AA)


def _draw_mask_contour(
    out: np.ndarray,
    mask: np.ndarray | None,
    *,
    color: tuple[int, int, int],
) -> None:
    if mask is None or not np.any(mask):
        return
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)


def _draw_contour_polyline(
    out: np.ndarray,
    contour_uv: Sequence[tuple[int, int]] | None,
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if not contour_uv or len(contour_uv) < 2:
        return
    pts = np.array(contour_uv, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _contour_circularity(contour_uv: Sequence[tuple[int, int]]) -> float:
    """4π·area/perimeter² — 1.0 for a circle."""
    if len(contour_uv) < 3:
        return 0.0
    pts = np.array(contour_uv, dtype=np.int32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(pts))
    peri = float(cv2.arcLength(pts, True))
    if peri < 1e-3:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _obb_corners_from_points(
    points: Sequence[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Minimum-area rotated rectangle corners (u, v)."""
    if len(points) < 3:
        return None
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(arr)
    if rect[1][0] < 1.0 or rect[1][1] < 1.0:
        return None
    box = cv2.boxPoints(rect)
    return [(int(round(p[0])), int(round(p[1]))) for p in box]


def _fit_ellipse_params(
    contour_uv: Sequence[tuple[int, int]],
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """((cx, cy), (w, h), angle_deg) or None."""
    if len(contour_uv) < 5:
        return None
    pts = np.array(contour_uv, dtype=np.float32).reshape(-1, 1, 2)
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    if w < 2.0 or h < 2.0:
        return None
    return ((float(cx), float(cy)), (float(w), float(h)), float(angle))


def _pca_angle_deg(points: Sequence[tuple[int, int]]) -> float:
    """Principal axis angle (degrees) for click hull — used to align OBB."""
    if len(points) < 2:
        return 0.0
    arr = np.array(points, dtype=np.float64)
    centered = arr - arr.mean(axis=0)
    if centered.shape[0] < 2:
        return 0.0
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.degrees(np.arctan2(major[1], major[0])))


def pick_image_overlay(
    contour_uv: Sequence[tuple[int, int]],
    shape_source: ShapeSource,
    include_uvs: Sequence[tuple[int, int]],
    *,
    mask: np.ndarray | None = None,
) -> tuple[ImageOverlayPhase, list[tuple[int, int]], list[tuple[int, int]] | None, tuple | None]:
    """Heuristic 2D overlay: Haiku polygon → contour; round → ellipse; 4+ clicks → OBB."""
    contour = list(contour_uv)
    if mask is not None and np.any(mask) and len(contour) < 3:
        from scanner.mask_haiku import contour_points_from_mask

        contour = contour_points_from_mask(mask)

    obb: list[tuple[int, int]] | None = None
    ellipse: tuple | None = None

    if shape_source == "haiku" and len(contour) >= 3:
        return "image_contour", contour, None, None

    circ = _contour_circularity(contour) if len(contour) >= 5 else 0.0
    if circ >= 0.72 and len(contour) >= 5:
        ellipse = _fit_ellipse_params(contour)
        if ellipse is not None:
            return "image_ellipse", contour, None, ellipse

    click_pts = list(include_uvs)
    if len(click_pts) >= 4:
        obb = _obb_corners_from_points(click_pts)
        if obb is not None:
            return "image_obb", contour if len(contour) >= 3 else obb, obb, None

    if len(contour) >= 3:
        obb = _obb_corners_from_points(contour)
        if obb is not None and len(click_pts) >= 3:
            return "image_obb", contour, obb, None
        return "image_contour", contour, obb, None

    if len(click_pts) >= 3:
        hull = cv2.convexHull(np.array(click_pts, dtype=np.int32))
        contour = [(int(p[0][0]), int(p[0][1])) for p in hull]
        obb = _obb_corners_from_points(contour)
        if obb is not None:
            return "image_obb", contour, obb, None
        return "image_contour", contour, None, None

    return "image_square", contour, None, None


def _draw_obb(
    out: np.ndarray,
    corners: Sequence[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(corners) < 4:
        return
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_ellipse_overlay(
    out: np.ndarray,
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    center, axes, angle = ellipse
    cv2.ellipse(
        out,
        (int(round(center[0])), int(round(center[1]))),
        (int(round(axes[0] * 0.5)), int(round(axes[1] * 0.5))),
        angle,
        0,
        360,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _apply_background_rejection(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    seed_uvs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not np.any(mask) or not seed_uvs:
        return mask
    return reject_background_in_mask(depth_mm, mask, intrinsics, seed_uvs)


def _smooth_bounds(
    prev: ScanBounds,
    measured: ScanBounds,
    *,
    alpha: float,
    shrink_alpha: float,
    prevent_shrink: bool = False,
) -> ScanBounds:
    center = (1.0 - alpha) * prev.center_m + alpha * measured.center_m

    new_half = np.zeros(3, dtype=np.float64)
    for i in range(3):
        p = prev.half_extents_m[i]
        m = measured.half_extents_m[i]
        if m > p:
            new_half[i] = (1.0 - alpha) * p + alpha * m
        elif prevent_shrink:
            new_half[i] = p
        else:
            new_half[i] = (1.0 - shrink_alpha) * p + shrink_alpha * m

    depth_min = prev.depth_min_mm
    depth_max = prev.depth_max_mm
    if measured.depth_min_mm < prev.depth_min_mm:
        depth_min = measured.depth_min_mm
    elif measured.depth_min_mm > prev.depth_min_mm:
        depth_min = (1.0 - shrink_alpha) * prev.depth_min_mm + shrink_alpha * measured.depth_min_mm
    if measured.depth_max_mm > prev.depth_max_mm:
        depth_max = measured.depth_max_mm
    elif measured.depth_max_mm < prev.depth_max_mm:
        depth_max = (1.0 - shrink_alpha) * prev.depth_max_mm + shrink_alpha * measured.depth_max_mm

    return ScanBounds(
        center_m=center,
        half_extents_m=new_half,
        depth_min_mm=depth_min,
        depth_max_mm=depth_max,
        initialized=True,
    )


def _project_cube_corners(
    bounds: ScanBounds,
    intrinsics: Sequence[Sequence[float]],
    *,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) arrays for AABB corners projected to the image."""
    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    c = bounds.center_m
    h_ext = bounds.half_extents_m
    corners: list[tuple[float, float]] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                p = c + h_ext * np.array([sx, sy, sz])
                z = p[2]
                if z <= 1e-6:
                    continue
                u = p[0] * fx / z + cx
                v = -p[1] * fy / z + cy
                corners.append((u, v))
    if not corners:
        return np.array([]), np.array([])
    uv = np.array(corners, dtype=np.float64)
    us = np.clip(uv[:, 0], 0, w - 1)
    vs = np.clip(uv[:, 1], 0, h - 1)
    return us, vs


def _draw_image_rect(
    out: np.ndarray,
    rect_uv: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    u0, v0, u1, v1 = rect_uv
    cv2.rectangle(out, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)


def draw_bounds_overlay(
    bgr: np.ndarray,
    bounds: ScanBounds | None,
    intrinsics: Sequence[Sequence[float]] | None,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    assist_mask: np.ndarray | None = None,
    expand_flash: bool = False,
    phase: BoundsPhase = "depth_cube",
    image_rect_uv: tuple[int, int, int, int] | None = None,
    contour_uv: Sequence[tuple[int, int]] | None = None,
    obb_corners: Sequence[tuple[int, int]] | None = None,
    ellipse_params: tuple[tuple[float, float], tuple[float, float], float] | None = None,
) -> np.ndarray:
    """Draw oriented 2D shape (contour/OBB/ellipse) and optional 3D wireframe."""
    out = bgr.copy()
    _draw_mask_contour(out, assist_mask, color=(80, 255, 120))
    wire_color = (0, 220, 255) if expand_flash else color
    thickness = 5 if expand_flash else 3
    image_phases = ("image_contour", "image_obb", "image_ellipse", "image_square")

    if phase in image_phases:
        if phase == "image_ellipse" and ellipse_params is not None:
            _draw_ellipse_overlay(out, ellipse_params, color=CONTOUR_COLOR_BGR, thickness=2)
            if contour_uv and len(contour_uv) >= 3:
                _draw_contour_polyline(
                    out, contour_uv, color=CONTOUR_COLOR_BGR, thickness=1
                )
        elif phase == "image_obb" and obb_corners and len(obb_corners) >= 4:
            _draw_obb(out, obb_corners, color=CONTOUR_COLOR_BGR, thickness=2)
            if contour_uv and len(contour_uv) >= 3:
                _draw_contour_polyline(
                    out, contour_uv, color=CONTOUR_COLOR_BGR, thickness=1
                )
        elif contour_uv and len(contour_uv) >= 2:
            _draw_contour_polyline(out, contour_uv, color=CONTOUR_COLOR_BGR, thickness=2)
        elif assist_mask is not None:
            _draw_mask_contour(out, assist_mask, color=CONTOUR_COLOR_BGR)
        elif phase == "image_square" and image_rect_uv is not None:
            _draw_image_rect(out, image_rect_uv, color=wire_color, thickness=thickness)
        return out

    if contour_uv and len(contour_uv) >= 2:
        _draw_contour_polyline(out, contour_uv, color=CONTOUR_COLOR_BGR, thickness=2)
    if bounds is None or intrinsics is None or not bounds.initialized:
        if image_rect_uv is not None and not contour_uv:
            _draw_image_rect(out, image_rect_uv, color=wire_color, thickness=thickness)
        return out
    if phase == "depth_cube":
        _draw_cube_wireframe(out, bounds, intrinsics, color=wire_color, thickness=thickness)
    return out


def _edge_roi_mask(
    rgb_bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    """Lightweight silhouette hint: Canny edges inside dilated seed region."""
    if not np.any(seed_mask):
        return seed_mask
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region = cv2.dilate(seed_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    edge_band = cv2.dilate(edges, kernel, iterations=1).astype(bool)
    trimmed = seed_mask & (~edge_band | region)
    return trimmed if np.count_nonzero(trimmed) >= np.count_nonzero(seed_mask) * 0.25 else seed_mask




def _cluster_mask_in_bounds(
    depth_mm: np.ndarray,
    seed_uv: tuple[int, int],
    bounds: ScanBounds,
    intrinsics: Sequence[Sequence[float]],
    *,
    band_mm: float,
    margin_m: float,
) -> np.ndarray:
    """Full-resolution mask: depth cluster pixels inside 3D AABB."""
    h, w = depth_mm.shape
    focus_mm = float(bounds.center_m[2] * 1000.0)
    cluster = object_mask_from_click(
        depth_mm,
        seed_uv[0],
        seed_uv[1],
        center_mm=focus_mm,
        band_mm=band_mm,
    )
    if not np.any(cluster):
        return cluster
    ys, xs = np.where(cluster)
    z_mm = depth_mm[ys, xs].astype(np.float64)
    valid = z_mm > 0
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool)
    ys = ys[valid]
    xs = xs[valid]
    z_mm = z_mm[valid]
    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    z = z_mm / 1000.0
    u = xs.astype(np.float64)
    v = ys.astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    inside = bounds.contains_points_m(pts, margin_m=margin_m)
    out = np.zeros((h, w), dtype=bool)
    out[ys[inside], xs[inside]] = True
    return out


def _cluster_masks_in_bounds(
    depth_mm: np.ndarray,
    seed_uvs: Sequence[tuple[int, int]],
    bounds: ScanBounds,
    intrinsics: Sequence[Sequence[float]],
    *,
    band_mm: float,
    margin_m: float,
) -> np.ndarray:
    """Union depth-cluster masks from multiple seeds, clipped to 3D AABB."""
    if not seed_uvs:
        return np.zeros(depth_mm.shape[:2], dtype=bool)
    combined = np.zeros(depth_mm.shape[:2], dtype=bool)
    for seed_uv in seed_uvs:
        combined |= _cluster_mask_in_bounds(
            depth_mm,
            seed_uv,
            bounds,
            intrinsics,
            band_mm=band_mm,
            margin_m=margin_m,
        )
    return combined

def _mask_pixels_to_points_m(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    stride: int = 4,
) -> np.ndarray:
    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    h, w = depth_mm.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    uu, vv = np.meshgrid(xs, ys)
    m = mask[vv, uu]
    z_mm = depth_mm[vv, uu]
    valid = m & (z_mm > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    z = z_mm[valid].astype(np.float64) / 1000.0
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


class BoundsTracker:
    """Session-only 3D cube: init from clicks, track per frame, filter voxels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = ScanBoundsSettings()
        self._bounds: ScanBounds | None = None
        self._seed_uv: tuple[int, int] | None = None
        self._click_uvs: list[tuple[int, int]] = []
        self._frame_count = 0
        self._points_in_bounds = 0
        self._tracking = False
        self._intrinsics: Sequence[Sequence[float]] | None = None
        self._last_rgb: np.ndarray | None = None
        self._assist_mask: np.ndarray | None = None
        self._assist_succeeded = False
        self._detected_pixels = 0
        self._last_click_mode = "none"
        self._refine_seeds: list[tuple[int, int]] = []
        self._exclude_uvs: list[tuple[int, int]] = []
        self._exclude_mask_cached: np.ndarray | None = None
        self._last_expand_delta_mm: dict[str, float] | None = None
        self._expand_flash_frames = 0
        self._expand_hold_frames = 0
        self._phase: BoundsPhase = "none"
        self._image_rect_uv: tuple[int, int, int, int] | None = None
        self._all_clicks_inside: bool | None = None
        self._contour_points: list[tuple[int, int]] = []
        self._obb_corners: list[tuple[int, int]] = []
        self._ellipse_params: tuple[tuple[float, float], tuple[float, float], float] | None = None
        self._shape_source: ShapeSource = "clicks"
        self._haiku_pending = False

    def _update_image_overlay(
        self,
        contour: Sequence[tuple[int, int]],
        shape_source: ShapeSource,
        *,
        mask: np.ndarray | None = None,
        force_2d_phase: bool = False,
    ) -> None:
        """Pick contour / OBB / ellipse display from shape heuristics."""
        with self._lock:
            click_uvs = list(self._click_uvs)
            had_cube = self._bounds is not None and self._phase == "depth_cube"
        img_phase, contour_out, obb, ellipse = pick_image_overlay(
            contour, shape_source, click_uvs, mask=mask
        )
        with self._lock:
            self._contour_points = list(contour_out)
            self._obb_corners = list(obb) if obb else []
            self._ellipse_params = ellipse
            if force_2d_phase or not had_cube:
                self._phase = img_phase

    @property
    def obb_corners(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._obb_corners)

    @property
    def ellipse_params(self) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        with self._lock:
            return self._ellipse_params

    def _all_seeds_unlocked(self) -> list[tuple[int, int]]:
        seeds: list[tuple[int, int]] = []
        if self._seed_uv is not None:
            seeds.append(self._seed_uv)
        for uv in self._refine_seeds:
            if uv not in seeds:
                seeds.append(uv)
        return seeds

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._bounds is not None and self._bounds.initialized

    @property
    def bounds(self) -> ScanBounds | None:
        with self._lock:
            if self._bounds is None:
                return None
            return self._bounds

    def settings_dict(self) -> dict[str, Any]:
        with self._lock:
            s = self._settings
            return {
                "margin_mm": s.margin_mm,
                "expand_margin_mm": s.expand_margin_mm,
                "smooth_alpha": s.smooth_alpha,
                "default_half_extent_mm": s.default_half_extent_mm,
                "min_half_extent_mm": s.min_half_extent_mm,
                "max_half_extent_mm": s.max_half_extent_mm,
                "search_expand_ratio": s.search_expand_ratio,
                "shrink_alpha": s.shrink_alpha,
                "depth_band_mm": s.depth_band_mm,
            }

    def patch_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = set(self.settings_dict().keys())
        with self._lock:
            for key, val in updates.items():
                if key not in allowed:
                    continue
                setattr(self._settings, key, float(val))
            return self.settings_dict()

    def clear(self) -> None:
        with self._lock:
            self._bounds = None
            self._seed_uv = None
            self._click_uvs.clear()
            self._frame_count = 0
            self._points_in_bounds = 0
            self._tracking = False
            self._intrinsics = None
            self._last_rgb = None
            self._assist_mask = None
            self._assist_succeeded = False
            self._detected_pixels = 0
            self._last_click_mode = "none"
            self._refine_seeds.clear()
            self._exclude_uvs.clear()
            self._exclude_mask_cached = None
            self._last_expand_delta_mm = None
            self._expand_flash_frames = 0
            self._expand_hold_frames = 0
            self._phase = "none"
            self._image_rect_uv = None
            self._all_clicks_inside = None
            self._contour_points = []
            self._obb_corners = []
            self._ellipse_params = None
            self._shape_source = "clicks"
            self._haiku_pending = False

    @property
    def contour_points(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._contour_points)

    @property
    def shape_source(self) -> ShapeSource:
        with self._lock:
            return self._shape_source

    def set_haiku_pending(self, pending: bool) -> None:
        with self._lock:
            self._haiku_pending = pending

    @property
    def haiku_pending(self) -> bool:
        with self._lock:
            return self._haiku_pending

    @property
    def phase(self) -> BoundsPhase:
        with self._lock:
            return self._phase

    @property
    def image_rect_uv(self) -> tuple[int, int, int, int] | None:
        with self._lock:
            return self._image_rect_uv

    def _recompute_bounds_from_mask(
        self,
        mask: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> None:
        """Tighten 3D AABB from remaining mask inliers (smart exclude shrink)."""
        with self._lock:
            click_uvs = list(self._click_uvs)
            settings = self._settings
        if not click_uvs or not np.any(mask):
            return
        inlier_pts = _mask_pixels_to_points_m(depth_mm, mask, intrinsics, stride=3)
        click_pts = _unproject_clicks_to_points_m(click_uvs, depth_mm, intrinsics)
        pieces: list[np.ndarray] = []
        if inlier_pts.shape[0] >= 4:
            pieces.append(inlier_pts)
        if click_pts.shape[0] > 0:
            pieces.append(click_pts)
        if not pieces:
            return
        all_pts = np.vstack(pieces)
        new_bounds = _bounds_from_points_union(all_pts, settings)
        if new_bounds is None:
            return
        with self._lock:
            self._bounds = new_bounds
            self._phase = "depth_cube"
            self._intrinsics = intrinsics
            self._all_clicks_inside = _all_clicks_inside_bounds(
                new_bounds,
                click_uvs,
                depth_mm,
                intrinsics,
                margin_mm=settings.expand_margin_mm,
            )

    def _reconcile_bounds_with_clicks(
        self,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
        *,
        cluster_pts_m: np.ndarray | None = None,
        region_mask: np.ndarray | None = None,
    ) -> None:
        """Rebuild 3D cube from contour/rect depth + all unprojected include clicks."""
        with self._lock:
            click_uvs = list(self._click_uvs)
            settings = self._settings
            current = self._bounds
            h, w = depth_mm.shape[:2]
            self._image_rect_uv = _uv_rect_from_clicks(
                click_uvs, h, w, depth_mm=depth_mm, intrinsics=intrinsics
            )

        if not click_uvs:
            return

        rect = self._image_rect_uv
        assert rect is not None
        pieces: list[np.ndarray] = []
        if region_mask is not None and np.any(region_mask):
            mask_pts = _mask_pixels_to_points_m(
                depth_mm, region_mask, intrinsics, stride=3
            )
            if mask_pts.shape[0] > 0:
                pieces.append(mask_pts)
        else:
            rect_pts = _depth_points_in_uv_rect(depth_mm, rect, intrinsics)
            if rect_pts.shape[0] > 0:
                pieces.append(rect_pts)
        click_pts = _unproject_clicks_to_points_m(click_uvs, depth_mm, intrinsics)
        if click_pts.shape[0] > 0:
            pieces.append(click_pts)
        if cluster_pts_m is not None and cluster_pts_m.shape[0] > 0:
            pieces.append(cluster_pts_m)
        if current is not None:
            lo = current.center_m - current.half_extents_m
            hi = current.center_m + current.half_extents_m
            pieces.append(np.stack([lo, hi], axis=0))

        if not pieces:
            with self._lock:
                self._update_image_overlay(
                    self._contour_points,
                    self._shape_source,
                    mask=region_mask,
                    force_2d_phase=True,
                )
            return

        all_pts = np.vstack(pieces)
        new_bounds = _bounds_from_points_union(all_pts, settings)
        if new_bounds is None:
            with self._lock:
                self._update_image_overlay(
                    self._contour_points,
                    self._shape_source,
                    mask=region_mask,
                    force_2d_phase=True,
                )
            return

        with self._lock:
            self._bounds = new_bounds
            self._phase = "depth_cube"
            self._intrinsics = intrinsics
            self._all_clicks_inside = _all_clicks_inside_bounds(
                new_bounds,
                click_uvs,
                depth_mm,
                intrinsics,
                margin_mm=settings.expand_margin_mm,
            )
        if len(click_uvs) >= 4:
            obb = _obb_corners_from_points(click_uvs)
            if obb is not None:
                with self._lock:
                    self._obb_corners = obb

    def apply_shape_mask(
        self,
        mask: np.ndarray,
        contour: Sequence[tuple[int, int]],
        shape_source: ShapeSource,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> None:
        """Apply 2D shape (Haiku or depth) and reconcile 3D cube."""
        if not np.any(mask):
            return
        cluster_pts = _mask_pixels_to_points_m(depth_mm, mask, intrinsics, stride=2)
        with self._lock:
            self._assist_mask = mask
            self._assist_succeeded = True
            self._detected_pixels = int(np.count_nonzero(mask))
            self._shape_source = shape_source
            self._haiku_pending = False
        self._update_image_overlay(contour, shape_source, mask=mask)
        self._reconcile_bounds_with_clicks(
            depth_mm,
            intrinsics,
            cluster_pts_m=cluster_pts if cluster_pts.shape[0] >= 4 else None,
            region_mask=mask,
        )

    def exclude_uvs(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self._exclude_uvs)

    def _exclude_region_at_click(
        self,
        depth_mm: np.ndarray,
        click_uv: tuple[int, int],
    ) -> np.ndarray:
        """Depth cluster connected to click UV (same seed strategy as object_mask)."""
        u, v = click_uv
        d = sample_depth_at_click(depth_mm, u, v)
        if d is None:
            return np.zeros(depth_mm.shape[:2], dtype=bool)
        band_mm = adaptive_depth_band_mm(d)
        return object_mask_from_click(
            depth_mm, u, v, center_mm=d, band_mm=band_mm
        )

    def _combined_exclude_mask(self, depth_mm: np.ndarray) -> np.ndarray:
        with self._lock:
            uvs = list(self._exclude_uvs)
        if not uvs:
            return np.zeros(depth_mm.shape[:2], dtype=bool)
        out = np.zeros(depth_mm.shape[:2], dtype=bool)
        for uv in uvs:
            out |= self._exclude_region_at_click(depth_mm, uv)
        return out

    def _apply_excludes(self, mask: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        ex = self._combined_exclude_mask(depth_mm)
        if not np.any(ex):
            return mask
        return mask & ~ex

    def _refresh_exclude_mask_cache(self, depth_mm: np.ndarray) -> None:
        ex = self._combined_exclude_mask(depth_mm)
        with self._lock:
            self._exclude_mask_cached = ex if np.any(ex) else None

    def set_exclude_clicks(
        self,
        click_uvs: Sequence[tuple[int, int]],
        depth_mm: np.ndarray | None = None,
        intrinsics: Sequence[Sequence[float]] | None = None,
    ) -> None:
        with self._lock:
            self._exclude_uvs = list(click_uvs)
            if click_uvs:
                self._last_click_mode = "exclude"
        if depth_mm is None:
            return
        self._refresh_exclude_mask_cache(depth_mm)
        with self._lock:
            assist = self._assist_mask
            intr = intrinsics or self._intrinsics
        if assist is None or not np.any(assist):
            return
        ex = self._combined_exclude_mask(depth_mm)
        if not np.any(ex):
            return
        trimmed = assist & ~ex
        if not np.any(trimmed):
            return
        from scanner.mask_haiku import contour_points_from_mask

        contour = contour_points_from_mask(trimmed)
        with self._lock:
            self._assist_mask = trimmed
            self._detected_pixels = int(np.count_nonzero(trimmed))
            shape_src = self._shape_source
        self._update_image_overlay(contour, shape_src, mask=trimmed)
        if intr is not None:
            self._recompute_bounds_from_mask(trimmed, depth_mm, intr)

    @property
    def expand_flash_active(self) -> bool:
        with self._lock:
            return self._expand_flash_frames > 0

    def assist_mask(self) -> np.ndarray | None:
        with self._lock:
            base = self._assist_mask
            ex = self._exclude_mask_cached
        if base is None:
            return None
        if ex is not None and np.any(ex):
            return base & ~ex
        return base

    def initialize_assist_detect(
        self,
        seed_uv: tuple[int, int],
        rgb_bgr: np.ndarray | None,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> None:
        settings = self._settings
        seed_u, seed_v = seed_uv
        h, w = depth_mm.shape[:2]
        seed_depth = sample_depth_at_click(depth_mm, seed_u, seed_v)

        with self._lock:
            self._seed_uv = seed_uv
            self._click_uvs = [seed_uv]
            self._refine_seeds.clear()
            self._intrinsics = intrinsics
            self._last_rgb = rgb_bgr
            self._frame_count = 0
            self._tracking = False
            self._image_rect_uv = _uv_rect_from_clicks(
                [seed_uv], h, w, depth_mm=depth_mm, intrinsics=intrinsics
            )

        if seed_depth is None:
            with self._lock:
                self._bounds = None
                self._phase = "image_square"
                self._assist_mask = None
                self._assist_succeeded = False
                self._detected_pixels = 0
                self._last_click_mode = "assist_detect"
                self._contour_points = []
                self._shape_source = "clicks"
            return

        band_mm = adaptive_depth_band_mm(seed_depth)
        settings.depth_band_mm = band_mm
        mask, contour, shape_src = build_adaptive_shape_mask(
            rgb_bgr,
            depth_mm,
            [seed_uv],
            intrinsics,
            settings,
        )
        if np.any(mask):
            mask = _apply_background_rejection(
                mask, depth_mm, intrinsics, [(seed_u, seed_v)]
            )
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = [(int(p[0][0]), int(p[0][1])) for p in largest]
        pixels = int(np.count_nonzero(mask))
        assist_ok = pixels >= 20
        cluster_pts: np.ndarray | None = None
        if assist_ok:
            cluster_pts = _mask_pixels_to_points_m(
                depth_mm, mask, intrinsics, stride=2
            )
            if cluster_pts.shape[0] < 8:
                assist_ok = False
                cluster_pts = None

        self._reconcile_bounds_with_clicks(
            depth_mm,
            intrinsics,
            cluster_pts_m=cluster_pts,
            region_mask=mask if assist_ok else None,
        )

        with self._lock:
            self._points_in_bounds = pixels
            self._assist_mask = mask if assist_ok else None
            self._assist_succeeded = assist_ok
            self._detected_pixels = pixels
            src = shape_src if assist_ok else "clicks"
            self._shape_source = src
            self._last_click_mode = "assist_detect"
        if assist_ok:
            self._update_image_overlay(contour, src, mask=mask)
        elif self._bounds is None:
            with self._lock:
                self._phase = "image_square"

    def handle_additional_click(
        self,
        click_uv: tuple[int, int],
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
        rgb_bgr: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            bounds = self._bounds
            settings = self._settings
            if click_uv not in self._click_uvs:
                self._click_uvs.append(click_uv)
            h, w = depth_mm.shape[:2]
            self._image_rect_uv = _uv_rect_from_clicks(
                self._click_uvs, h, w, depth_mm=depth_mm, intrinsics=intrinsics
            )
            bounds = self._bounds

        d = sample_depth_at_click(depth_mm, click_uv[0], click_uv[1])
        if d is None:
            with self._lock:
                self._last_click_mode = "no_depth"
                if bounds is None:
                    self._phase = "image_square"
            return

        if bounds is None:
            self._reconcile_bounds_with_clicks(depth_mm, intrinsics)
            with self._lock:
                bounds = self._bounds
            if bounds is None:
                with self._lock:
                    self._last_click_mode = "no_depth"
                return

        pt = unproject_uv_to_m(click_uv[0], click_uv[1], d, intrinsics)
        h, w = depth_mm.shape[:2]
        classify_m = settings.classify_inside_margin_mm / 1000.0
        inside_3d = bool(
            bounds.contains_points_m(pt.reshape(1, 3), margin_m=classify_m)[0]
        )
        outside_uv = _click_outside_projected_aabb(
            click_uv,
            bounds,
            intrinsics,
            h=h,
            w=w,
            tolerance_px=settings.outside_uv_tolerance_px,
        )
        if outside_uv:
            self.expand_to_include(click_uv, depth_mm, intrinsics)
        elif not inside_3d:
            self.expand_to_include(click_uv, depth_mm, intrinsics)
        else:
            self._refine_inside(click_uv, d, depth_mm, intrinsics, rgb_bgr)

        self._reconcile_bounds_with_clicks(depth_mm, intrinsics)

    def _refine_inside(
        self,
        click_uv: tuple[int, int],
        depth_mm_at_click: float,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
        rgb_bgr: np.ndarray | None,
    ) -> None:
        with self._lock:
            bounds = self._bounds
            settings = self._settings
            seeds = self._all_seeds_unlocked()
            if click_uv not in self._refine_seeds and click_uv != self._seed_uv:
                self._refine_seeds.append(click_uv)
            seeds = self._all_seeds_unlocked()
        if bounds is None:
            return
        band_mm = adaptive_depth_band_mm(depth_mm_at_click)
        focus_mm = float(bounds.center_m[2] * 1000.0)
        mask = assisted_object_mask_from_seeds(
            rgb_bgr,
            depth_mm,
            seeds,
            center_mm=focus_mm,
            band_mm=band_mm,
            canny_low=int(settings.edge_canny_low),
            canny_high=int(settings.edge_canny_high),
        )
        margin_m = settings.margin_mm / 1000.0
        if np.any(mask):
            ys, xs = np.where(mask)
            z_mm = depth_mm[ys, xs].astype(np.float64)
            valid = z_mm > 0
            if np.any(valid):
                k = intrinsics_matrix(intrinsics)
                fx, fy = k[0, 0], k[1, 1]
                cx, cy = k[0, 2], k[1, 2]
                z = z_mm[valid] / 1000.0
                u = xs[valid].astype(np.float64)
                v = ys[valid].astype(np.float64)
                x = (u - cx) * z / fx
                y = -(v - cy) * z / fy
                pts = np.stack([x, y, z], axis=1)
                inside = bounds.contains_points_m(pts, margin_m=margin_m)
                clipped = np.zeros(mask.shape, dtype=bool)
                clipped[ys[valid][inside], xs[valid][inside]] = True
                mask = clipped
        if np.any(mask):
            mask = _apply_background_rejection(
                mask, depth_mm, intrinsics, seeds
            )
        pixels = int(np.count_nonzero(mask))
        with self._lock:
            if pixels >= 20:
                self._assist_mask = mask
                self._assist_succeeded = True
                self._detected_pixels = pixels
            if click_uv not in self._click_uvs:
                self._click_uvs.append(click_uv)
            self._intrinsics = intrinsics
            if rgb_bgr is not None:
                self._last_rgb = rgb_bgr
            self._last_click_mode = "refine_inside"

        self._reconcile_bounds_with_clicks(depth_mm, intrinsics)

    def expand_to_include(
        self,
        click_uv: tuple[int, int],
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> None:
        with self._lock:
            bounds = self._bounds
            settings = self._settings
            if click_uv not in self._click_uvs:
                self._click_uvs.append(click_uv)
        d = sample_depth_at_click(depth_mm, click_uv[0], click_uv[1])
        if d is None:
            with self._lock:
                self._last_click_mode = "expand_to_include"
            return
        if bounds is None:
            self._reconcile_bounds_with_clicks(depth_mm, intrinsics)
            with self._lock:
                bounds = self._bounds
            if bounds is None:
                return
        pt = unproject_uv_to_m(click_uv[0], click_uv[1], d, intrinsics)
        expanded, delta_mm = _expand_bounds_to_include_point(bounds, pt, settings)
        with self._lock:
            self._bounds = expanded
            self._phase = "depth_cube"
            self._intrinsics = intrinsics
            self._last_click_mode = "expand_to_include"
            self._last_expand_delta_mm = {
                "x": round(float(delta_mm[0]), 1),
                "y": round(float(delta_mm[1]), 1),
                "z": round(float(delta_mm[2]), 1),
            }
            self._expand_flash_frames = 12
            self._expand_hold_frames = int(settings.expand_hold_frames)

    def initialize_from_clicks(
        self,
        click_uvs: Sequence[tuple[int, int]],
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> None:
        if not click_uvs:
            self.clear()
            return

        settings = self._settings
        seed_u, seed_v = click_uvs[0]
        seed_depth = sample_depth_at_click(depth_mm, seed_u, seed_v)
        if seed_depth is None:
            with self._lock:
                self._seed_uv = (seed_u, seed_v)
                self._click_uvs = list(click_uvs)
                self._intrinsics = intrinsics
                self._bounds = None
            return

        if len(click_uvs) == 1:
            center = unproject_uv_to_m(seed_u, seed_v, seed_depth, intrinsics)
            half_m = np.full(3, settings.default_half_extent_mm / 1000.0)
            half_m = _clamp_half_extents_m(
                half_m,
                min_mm=settings.min_half_extent_mm,
                max_mm=settings.max_half_extent_mm,
            )
            z_mm = center[2] * 1000.0
            bounds = ScanBounds(
                center_m=center,
                half_extents_m=half_m,
                depth_min_mm=max(0.0, z_mm - settings.depth_band_mm),
                depth_max_mm=z_mm + settings.depth_band_mm,
            )
        else:
            points_3d = []
            for u, v in click_uvs:
                d = sample_depth_at_click(depth_mm, u, v)
                if d is not None:
                    points_3d.append(unproject_uv_to_m(u, v, d, intrinsics))
            if not points_3d:
                center = unproject_uv_to_m(seed_u, seed_v, seed_depth, intrinsics)
                half_m = np.full(3, settings.default_half_extent_mm / 1000.0)
            else:
                pts = np.stack(points_3d, axis=0)
                center, half_m = _aabb_from_points_m(pts)
                half_m = np.maximum(half_m, settings.min_half_extent_mm / 1000.0)
            half_m = _clamp_half_extents_m(
                half_m,
                min_mm=settings.min_half_extent_mm,
                max_mm=settings.max_half_extent_mm,
            )
            z_lo = (center[2] - half_m[2]) * 1000.0
            z_hi = (center[2] + half_m[2]) * 1000.0
            margin = settings.depth_band_mm * 0.5
            bounds = ScanBounds(
                center_m=center,
                half_extents_m=half_m,
                depth_min_mm=max(0.0, z_lo - margin),
                depth_max_mm=z_hi + margin,
            )

        with self._lock:
            self._bounds = bounds
            self._seed_uv = (seed_u, seed_v)
            self._click_uvs = list(click_uvs)
            self._intrinsics = intrinsics
            self._frame_count = 0
            self._points_in_bounds = 0
            self._tracking = False

    def update(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
    ) -> ScanBounds | None:
        with self._lock:
            bounds = self._bounds
            seed_uv = self._seed_uv
            settings = self._settings
            frame_count = self._frame_count

        if bounds is None or seed_uv is None:
            return None

        search = bounds.expanded(settings.search_expand_ratio)
        margin_m = settings.margin_mm / 1000.0

        with self._lock:
            seed_uvs = self._all_seeds_unlocked()
        band_mm = adaptive_depth_band_mm(float(bounds.center_m[2] * 1000.0))
        combined = _cluster_masks_in_bounds(
            depth_mm,
            seed_uvs,
            search,
            intrinsics,
            band_mm=band_mm,
            margin_m=margin_m,
        )

        if rgb_bgr is not None and rgb_bgr.size > 0:
            combined = _edge_roi_mask(
                rgb_bgr,
                combined,
                canny_low=int(settings.edge_canny_low),
                canny_high=int(settings.edge_canny_high),
            )

        if np.any(combined):
            combined = _apply_background_rejection(
                combined, depth_mm, intrinsics, seed_uvs
            )

        if not np.any(combined):
            with self._lock:
                self._frame_count += 1
                self._tracking = self._frame_count > 1
                self._last_rgb = rgb_bgr
            return bounds

        inlier_pts = _mask_pixels_to_points_m(depth_mm, combined, intrinsics, stride=4)
        if inlier_pts.shape[0] < 8:
            with self._lock:
                self._points_in_bounds = int(inlier_pts.shape[0])
                self._frame_count += 1
                self._tracking = self._frame_count > 1
                self._last_rgb = rgb_bgr
            return bounds

        lo_p = np.percentile(inlier_pts, 2, axis=0)
        hi_p = np.percentile(inlier_pts, 98, axis=0)
        trimmed = inlier_pts[
            np.all((inlier_pts >= lo_p) & (inlier_pts <= hi_p), axis=1)
        ]
        if trimmed.shape[0] < 4:
            trimmed = inlier_pts
        meas_center, meas_half = _aabb_from_points_m(trimmed)
        meas_half = _clamp_half_extents_m(
            meas_half,
            min_mm=settings.min_half_extent_mm,
            max_mm=settings.max_half_extent_mm,
        )
        meas_z_lo = (meas_center[2] - meas_half[2]) * 1000.0
        meas_z_hi = (meas_center[2] + meas_half[2]) * 1000.0
        measured = ScanBounds(
            center_m=meas_center,
            half_extents_m=meas_half,
            depth_min_mm=max(0.0, meas_z_lo - settings.margin_mm),
            depth_max_mm=meas_z_hi + settings.margin_mm,
        )

        with self._lock:
            hold_frames = self._expand_hold_frames
            prevent_shrink = hold_frames > 0

        if frame_count == 0:
            updated = measured
        else:
            updated = _smooth_bounds(
                bounds,
                measured,
                alpha=settings.smooth_alpha,
                shrink_alpha=settings.shrink_alpha,
                prevent_shrink=prevent_shrink,
            )

        with self._lock:
            if self._expand_hold_frames > 0:
                self._expand_hold_frames -= 1
            if self._expand_flash_frames > 0:
                self._expand_flash_frames -= 1
            self._bounds = updated
            self._intrinsics = intrinsics
            self._points_in_bounds = int(trimmed.shape[0])
            self._frame_count += 1
            self._tracking = self._frame_count > 1
            self._last_rgb = rgb_bgr
            click_uvs = list(self._click_uvs)
            intrinsics_ref = self._intrinsics

        if (
            click_uvs
            and intrinsics_ref is not None
            and updated is not None
            and not _all_clicks_inside_bounds(
                updated,
                click_uvs,
                depth_mm,
                intrinsics_ref,
                margin_mm=settings.expand_margin_mm,
            )
        ):
            self._reconcile_bounds_with_clicks(
                depth_mm, intrinsics_ref, cluster_pts_m=trimmed
            )
            with self._lock:
                if self._bounds is not None:
                    updated = self._bounds

        return updated

    def build_mask(self, depth_mm: np.ndarray) -> np.ndarray | None:
        with self._lock:
            bounds = self._bounds
            seed_uv = self._seed_uv
            settings = self._settings
            intrinsics = self._intrinsics
            rgb_bgr = self._last_rgb

        if bounds is None or seed_uv is None or intrinsics is None:
            return None

        margin_m = settings.margin_mm / 1000.0
        with self._lock:
            seed_uvs = self._all_seeds_unlocked()
        band_mm = adaptive_depth_band_mm(float(bounds.center_m[2] * 1000.0))
        mask = _cluster_masks_in_bounds(
            depth_mm,
            seed_uvs,
            bounds,
            intrinsics,
            band_mm=band_mm,
            margin_m=margin_m,
        )
        if not np.any(mask):
            return None

        if rgb_bgr is not None and np.any(mask):
            mask = _edge_roi_mask(
                rgb_bgr,
                mask,
                canny_low=int(settings.edge_canny_low),
                canny_high=int(settings.edge_canny_high),
            )

        if np.any(mask):
            mask = _apply_background_rejection(
                mask, depth_mm, intrinsics, seed_uvs
            )

        if np.any(mask):
            mask = self._apply_excludes(mask, depth_mm)
            self._refresh_exclude_mask_cache(depth_mm)

        return mask if np.any(mask) else None

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            cube = self._bounds.to_status_dict() if self._bounds else None
            s = self._settings
            settings = {
                "margin_mm": s.margin_mm,
                "expand_margin_mm": s.expand_margin_mm,
                "smooth_alpha": s.smooth_alpha,
                "default_half_extent_mm": s.default_half_extent_mm,
                "min_half_extent_mm": s.min_half_extent_mm,
                "max_half_extent_mm": s.max_half_extent_mm,
                "search_expand_ratio": s.search_expand_ratio,
                "shrink_alpha": s.shrink_alpha,
                "depth_band_mm": s.depth_band_mm,
            }
            image_rect = None
            if self._image_rect_uv is not None:
                u0, v0, u1, v1 = self._image_rect_uv
                image_rect = {"u0": u0, "v0": v0, "u1": u1, "v1": v1}
            contour_pts = [{"u": u, "v": v} for u, v in self._contour_points]
            from scanner.mask_haiku import haiku_available

            return {
                "phase": self._phase,
                "shape_source": self._shape_source,
                "contour_points": contour_pts,
                "haiku_available": haiku_available(),
                "haiku_pending": self._haiku_pending,
                "image_rect_uv": image_rect,
                "all_clicks_inside": self._all_clicks_inside,
                "cube": cube,
                "tracking": self._tracking,
                "frames_tracked": self._frame_count,
                "points_in_bounds": self._points_in_bounds,
                "detected_pixels": self._detected_pixels,
                "assist_succeeded": self._assist_succeeded,
                "last_click_mode": self._last_click_mode,
                "last_action": (
                    "expanded_by_mm"
                    if self._last_click_mode == "expand_to_include"
                    and self._last_expand_delta_mm is not None
                    else self._last_click_mode
                ),
                "expanded_by_mm": self._last_expand_delta_mm,
                "expand_flash": self._expand_flash_frames > 0,
                "settings": settings,
                "refine_seed_count": len(self._refine_seeds),
                "exclude_count": len(self._exclude_uvs),
                "limitations": (
                    "Adaptive 2D contour from click hull + depth/Canny (yellow outline); "
                    "optional AI shape (Haiku) when API key is set. "
                    "Exclude clicks subtract regions and tighten the 3D cube. "
                    "Cube tracks during scan. Session-only — cleared on restart."
                ),
            }
