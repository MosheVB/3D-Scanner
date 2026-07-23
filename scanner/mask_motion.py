"""Motion / static-pixel masks from a quick turntable preview spin.

Fixed camera + rotating object: background pixels stay stable in RGB and depth;
object pixels shift. A 360° preview yields a *union corridor* in image space that
covers everywhere the object appears during the full scan (the “swirled” region).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionMaskResult:
    """Include mask (True = keep depth) and diagnostics."""

    include: np.ndarray  # bool H×W
    depth_range_mm: np.ndarray  # float32 H×W
    rgb_std: np.ndarray  # float32 H×W
    static: np.ndarray  # bool H×W
    depth_band_mm: tuple[float, float]
    tau_depth_mm: float
    tau_rgb: float
    preview_frame_count: int


def _robust_depth_range(depth_stack: np.ndarray) -> np.ndarray:
    """Per-pixel depth span across frames (ignores invalid zeros)."""
    # depth_stack: (N, H, W) float32
    valid = depth_stack > 0
    d_inf = np.where(valid, depth_stack, np.inf)
    d_neg = np.where(valid, depth_stack, -np.inf)
    d_min = np.min(d_inf, axis=0)
    d_max = np.max(d_neg, axis=0)
    out = np.zeros(depth_stack.shape[1:], dtype=np.float32)
    ok = np.isfinite(d_min) & np.isfinite(d_max) & (d_max >= d_min)
    out[ok] = (d_max[ok] - d_min[ok]).astype(np.float32)
    return out


def _pick_center_component(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    *,
    cx: float,
    cy: float,
    min_area: int,
) -> np.ndarray:
    """Largest motion blob, biased toward the turntable center."""
    h, w = labels.shape
    best_label: int | None = None
    best_score = -1.0
    for label in range(1, int(stats.shape[0])):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ccx, ccy = centroids[label]
        dist = float(np.hypot(ccx - cx, ccy - cy))
        max_dist = float(np.hypot(w, h))
        score = area * (1.0 - 0.35 * (dist / max(max_dist, 1.0)))
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return np.zeros(labels.shape, dtype=bool)
    return labels == best_label


def compute_motion_mask(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    *,
    min_valid_fraction: float = 0.35,
    depth_tau_floor_mm: float = 8.0,
    rgb_tau_floor: float = 3.5,
    morph_kernel: int = 7,
    min_component_area: int = 800,
    center_bias: tuple[float, float] | None = None,
) -> MotionMaskResult:
    """Build include mask from preview spin frames."""
    if len(rgb_frames) < 3 or len(rgb_frames) != len(depth_frames):
        raise ValueError("Need at least 3 matched RGB-D preview frames.")

    n = len(rgb_frames)
    h, w = depth_frames[0].shape[:2]
    depth_stack = np.stack(
        [d.astype(np.float32) for d in depth_frames],
        axis=0,
    )
    gray_stack = np.stack(
        [
            cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for r in rgb_frames
        ],
        axis=0,
    )

    depth_range = _robust_depth_range(depth_stack)
    rgb_std = np.std(gray_stack, axis=0).astype(np.float32)
    valid_count = (depth_stack > 0).sum(axis=0).astype(np.float32)
    min_valid = max(2.0, n * min_valid_fraction)

    positive = depth_range[depth_range > 0]
    if positive.size > 100:
        tau_d = float(max(depth_tau_floor_mm, np.percentile(positive, 35)))
    else:
        tau_d = depth_tau_floor_mm

    rgb_pos = rgb_std[rgb_std > 0]
    if rgb_pos.size > 100:
        tau_g = float(max(rgb_tau_floor, np.percentile(rgb_pos, 35)))
    else:
        tau_g = rgb_tau_floor

    with np.errstate(all="ignore"):
        med_depth = np.nanmedian(
            np.where(depth_stack > 0, depth_stack, np.nan),
            axis=0,
        )
    med_valid = med_depth[np.isfinite(med_depth) & (med_depth > 0)]
    if med_valid.size > 100:
        fg_cut = float(np.percentile(med_valid, 72))
    else:
        fg_cut = 900.0
    foreground = np.isfinite(med_depth) & (med_depth > 0) & (med_depth <= fg_cut)

    static = (
        (depth_range < tau_d)
        & (rgb_std < tau_g)
        & (valid_count >= min_valid)
    )
    # Definite background: stable across (almost) every frame
    static |= (valid_count >= n * 0.92) & (depth_range < tau_d * 1.5)

    motion = (
        ((depth_range >= tau_d) | (rgb_std >= tau_g))
        & (valid_count >= 3)
        & foreground
        & ~static
    )

    motion_u8 = motion.astype(np.uint8) * 255
    k = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    motion_u8 = cv2.morphologyEx(motion_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    motion_u8 = cv2.morphologyEx(motion_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    cx, cy = center_bias if center_bias else (w * 0.5, h * 0.5)
    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(
        motion_u8, connectivity=8
    )
    include = _pick_center_component(
        labels,
        stats,
        centroids,
        cx=cx,
        cy=cy,
        min_area=min_component_area,
    )

    if int(include.sum()) < min_component_area:
        # Fallback: threshold motion score directly
        score = depth_range / max(tau_d, 1e-3) + rgb_std / max(tau_g, 1e-3)
        thr = float(np.percentile(score[score > 0], 55)) if np.any(score > 0) else 1.0
        include = score >= thr
        include_u8 = include.astype(np.uint8) * 255
        include_u8 = cv2.morphologyEx(include_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(
            include_u8, connectivity=8
        )
        include = _pick_center_component(
            labels,
            stats,
            centroids,
            cx=cx,
            cy=cy,
            min_area=min_component_area // 2,
        )

    # Slight dilate so edges of rotating object stay inside corridor
    inc_u8 = include.astype(np.uint8) * 255
    inc_u8 = cv2.dilate(inc_u8, kernel, iterations=1)
    include = inc_u8 > 0

    # Depth band from preview include region
    band_vals = depth_stack[:, include]
    band_vals = band_vals[band_vals > 0]
    if band_vals.size >= 50:
        med = float(np.median(band_vals))
        spread = float(np.percentile(band_vals, 90) - np.percentile(band_vals, 10))
        margin = max(20.0, spread * 0.25)
        depth_band = (max(70.0, med - margin), med + margin)
    else:
        depth_band = (0.0, 2500.0)

    return MotionMaskResult(
        include=include,
        depth_range_mm=depth_range,
        rgb_std=rgb_std,
        static=static,
        depth_band_mm=depth_band,
        tau_depth_mm=tau_d,
        tau_rgb=tau_g,
        preview_frame_count=n,
    )


def render_motion_diagnostics(
    rgb_frames: list[np.ndarray],
    result: MotionMaskResult,
) -> dict[str, np.ndarray]:
    """Debug images: median RGB, motion heatmap, high-motion frame, mask overlay."""
    gray_stack = np.stack(
        [
            cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for r in rgb_frames
        ],
        axis=0,
    )
    median_gray = np.median(gray_stack, axis=0).astype(np.uint8)
    median_rgb = cv2.cvtColor(median_gray, cv2.COLOR_GRAY2BGR)

    dr = result.depth_range_mm
    valid = dr > 0
    if np.any(valid):
        lo, hi = float(np.percentile(dr[valid], 5)), float(np.percentile(dr[valid], 98))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((dr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(dr, dtype=np.uint8)
    motion_heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    motion_heat[~valid] = 0

    # Frame with strongest per-pixel deviation from temporal median (swirl proxy)
    dev = np.max(np.abs(gray_stack - median_gray[None, :, :]), axis=(1, 2))
    swirl = rgb_frames[int(np.argmax(dev))].copy()

    overlay = median_rgb.copy()
    overlay[result.include] = (
        overlay[result.include].astype(np.float32) * 0.45
        + np.array([0, 200, 0], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    overlay[result.static] = (
        overlay[result.static].astype(np.float32) * 0.5
        + np.array([0, 0, 200], dtype=np.float32) * 0.5
    ).astype(np.uint8)

    combo = np.hstack([median_rgb, motion_heat, overlay])
    return {
        "median_rgb": median_rgb,
        "motion_heat": motion_heat,
        "swirl": swirl,
        "overlay": overlay,
        "combo": combo,
    }
