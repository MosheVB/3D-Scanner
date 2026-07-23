"""Mask from continuous rotation video — per-pixel depth mean / stdev / MAD."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scanner.mask_motion import _pick_center_component, _robust_depth_range


@dataclass
class VideoMaskResult:
    include: np.ndarray
    platform: np.ndarray
    static_bg: np.ndarray
    depth_mean_mm: np.ndarray
    depth_std_mm: np.ndarray
    depth_mad_mm: np.ndarray
    depth_range_mm: np.ndarray
    valid_fraction: np.ndarray
    depth_band_mm: tuple[float, float]
    tau_std_mm: float
    frame_count: int
    fps: float


def _depth_temporal_stats(depth_stack: np.ndarray) -> tuple[np.ndarray, ...]:
    """depth_stack (N,H,W) float32, zeros = invalid."""
    n = depth_stack.shape[0]
    valid = depth_stack > 0
    valid_count = valid.sum(axis=0).astype(np.float32)
    valid_fraction = valid_count / max(n, 1)

    with np.errstate(invalid="ignore"):
        depth_mean = np.nanmean(np.where(valid, depth_stack, np.nan), axis=0)
        depth_std = np.nanstd(np.where(valid, depth_stack, np.nan), axis=0)

    # MAD per pixel
    mean_bc = np.where(np.isfinite(depth_mean), depth_mean, 0.0)[None, :, :]
    abs_dev = np.abs(np.where(valid, depth_stack, np.nan) - mean_bc)
    depth_mad = np.nanmedian(abs_dev, axis=0)
    depth_mad = np.where(np.isfinite(depth_mad), depth_mad, 0.0).astype(np.float32)

    depth_mean = np.where(np.isfinite(depth_mean), depth_mean, 0.0).astype(np.float32)
    depth_std = np.where(np.isfinite(depth_std), depth_std, 0.0).astype(np.float32)
    depth_range = _robust_depth_range(depth_stack)

    return depth_mean, depth_std, depth_mad, depth_range, valid_fraction


def compute_video_mask(
    depth_frames: list[np.ndarray],
    rgb_frames: list[np.ndarray] | None = None,
    *,
    min_valid_fraction: float = 0.75,
    std_tau_floor_mm: float = 4.0,
    mad_tau_floor_mm: float = 3.0,
    morph_kernel: int = 7,
    min_component_area: int = 600,
    center_bias: tuple[float, float] | None = None,
) -> VideoMaskResult:
    """Platform = low depth stdev; object = high stdev corridor on turntable."""
    if len(depth_frames) < 8:
        raise ValueError(f"Need at least 8 depth frames, got {len(depth_frames)}")

    n = len(depth_frames)
    h, w = depth_frames[0].shape[:2]
    depth_stack = np.stack([d.astype(np.float32) for d in depth_frames], axis=0)

    depth_mean, depth_std, depth_mad, depth_range, valid_fraction = _depth_temporal_stats(
        depth_stack
    )

    # Near-field gate (ignore far kitchen/wall for threshold estimation)
    med_valid = depth_mean[(depth_mean > 0) & (valid_fraction > 0.5)]
    if med_valid.size > 100:
        fg_cut = float(np.percentile(med_valid, 78))
    else:
        fg_cut = 950.0
    foreground = (depth_mean > 0) & (depth_mean <= fg_cut) & (valid_fraction >= 0.4)

    std_fg = depth_std[foreground & (depth_std > 0)]
    if std_fg.size > 100:
        tau_std = float(max(std_tau_floor_mm, np.percentile(std_fg, 28)))
    else:
        tau_std = std_tau_floor_mm

    mad_fg = depth_mad[foreground & (depth_mad > 0)]
    if mad_fg.size > 100:
        tau_mad = float(max(mad_tau_floor_mm, np.percentile(mad_fg, 28)))
    else:
        tau_mad = mad_tau_floor_mm

    # Rotating platform: stable depth (low std & mad), seen most of the spin
    platform = (
        (depth_std <= tau_std)
        & (depth_mad <= tau_mad)
        & (valid_fraction >= min_valid_fraction)
        & foreground
    )

    # Distant static background (also low std, often far)
    static_bg = (
        (depth_std <= tau_std * 1.2)
        & (valid_fraction >= min_valid_fraction)
        & ~foreground
    )

    # Object corridor: depth varies over time (not platform)
    motion = (
        ((depth_std > tau_std) | (depth_mad > tau_mad) | (depth_range > tau_std * 2.5))
        & (valid_fraction >= 0.35)
        & foreground
        & ~platform
    )

    # Optional: dark turntable in median RGB + low std
    if rgb_frames and len(rgb_frames) == n:
        gray_stack = np.stack(
            [cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) for r in rgb_frames], axis=0
        )
        med_gray = np.median(gray_stack, axis=0).astype(np.float32)
        dark_plate = (med_gray < 72) & foreground & (depth_std <= tau_std * 1.35)
        platform |= dark_plate

    k = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    motion_u8 = motion.astype(np.uint8) * 255
    motion_u8 = cv2.morphologyEx(motion_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    motion_u8 = cv2.morphologyEx(motion_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    cx, cy = center_bias if center_bias else (w * 0.5, h * 0.55)
    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(
        motion_u8, connectivity=8
    )
    include = _pick_center_component(
        labels, stats, centroids, cx=cx, cy=cy, min_area=min_component_area
    )

    if int(include.sum()) < min_component_area:
        score = depth_std / max(tau_std, 1e-3)
        thr = float(np.percentile(score[score > 0], 58)) if np.any(score > 0) else 1.0
        include = (score >= thr) & foreground & ~platform
        inc_u8 = include.astype(np.uint8) * 255
        inc_u8 = cv2.morphologyEx(inc_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inc_u8, connectivity=8
        )
        include = _pick_center_component(
            labels, stats, centroids, cx=cx, cy=cy, min_area=min_component_area // 2
        )

    inc_u8 = include.astype(np.uint8) * 255
    inc_u8 = cv2.dilate(inc_u8, kernel, iterations=1)
    include = inc_u8 > 0

    band_vals = depth_stack[:, include]
    band_vals = band_vals[band_vals > 0]
    if band_vals.size >= 50:
        med = float(np.median(band_vals))
        spread = float(np.percentile(band_vals, 90) - np.percentile(band_vals, 10))
        margin = max(18.0, spread * 0.22)
        depth_band = (max(70.0, med - margin), med + margin)
    else:
        depth_band = (0.0, 2500.0)

    return VideoMaskResult(
        include=include,
        platform=platform,
        static_bg=static_bg,
        depth_mean_mm=depth_mean,
        depth_std_mm=depth_std,
        depth_mad_mm=depth_mad,
        depth_range_mm=depth_range,
        valid_fraction=valid_fraction,
        depth_band_mm=depth_band,
        tau_std_mm=tau_std,
        frame_count=n,
        fps=0.0,
    )


def _colorize_scalar(
    field: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    cmap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    valid = field > 0 if mask is None else mask & (field > 0)
    out = np.zeros((*field.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = field[valid]
    lo, hi = float(np.percentile(vals, 3)), float(np.percentile(vals, 97))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(field.shape, dtype=np.uint8)
    norm[valid] = np.clip((field[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cmap)
    colored[~valid] = 0
    return colored


def render_video_diagnostics(
    rgb_frames: list[np.ndarray],
    result: VideoMaskResult,
) -> dict[str, np.ndarray]:
    gray_stack = np.stack(
        [cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) for r in rgb_frames], axis=0
    )
    median_gray = np.median(gray_stack, axis=0).astype(np.uint8)
    median_rgb = cv2.cvtColor(median_gray, cv2.COLOR_GRAY2BGR)

    mean_vis = _colorize_scalar(result.depth_mean_mm, mask=result.depth_mean_mm > 0)
    std_vis = _colorize_scalar(result.depth_std_mm, mask=result.depth_std_mm > 0)
    mad_vis = _colorize_scalar(result.depth_mad_mm, mask=result.depth_mad_mm > 0)
    range_vis = _colorize_scalar(
        result.depth_range_mm, mask=result.depth_range_mm > 0
    )

    overlay = median_rgb.copy()
    overlay[result.platform] = (
        overlay[result.platform].astype(np.float32) * 0.45
        + np.array([40, 40, 200], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    overlay[result.static_bg] = (
        overlay[result.static_bg].astype(np.float32) * 0.5
        + np.array([180, 40, 40], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    overlay[result.include] = (
        overlay[result.include].astype(np.float32) * 0.4
        + np.array([40, 220, 40], dtype=np.float32) * 0.6
    ).astype(np.uint8)

    stats_combo = np.hstack([mean_vis, std_vis, mad_vis])
    maps_combo = np.hstack([range_vis, std_vis, overlay])

    return {
        "median_rgb": median_rgb,
        "depth_mean_vis": mean_vis,
        "depth_std_vis": std_vis,
        "depth_mad_vis": mad_vis,
        "depth_range_vis": range_vis,
        "overlay": overlay,
        "stats_combo": stats_combo,
        "maps_combo": maps_combo,
    }
