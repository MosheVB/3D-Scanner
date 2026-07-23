"""Temporal RGB statistics from rotation video (no depth)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RgbVideoMaskResult:
    """Include mask from RGB temporal stats (rotation video)."""

    include: np.ndarray  # bool HxW
    gray_mad: np.ndarray
    gray_mean: np.ndarray
    gray_std: np.ndarray
    tau_mad: float = 0.0
    tau_std: float = 0.0
    tau_mean: float = 0.0
    preview_frame_count: int = 0
    method: str = "rgb_video_mean"


def _fill_holes_binary(mask_u8: np.ndarray) -> np.ndarray:
    """Fill interior holes in a foreground mask (255=fg)."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, holes)


def _pick_center_component(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    *,
    cx: float,
    cy: float,
    min_area: int,
) -> np.ndarray:
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


def estimate_rotation_record_s(
    rotate_speed: float,
    *,
    rotation_deg: float = 360.0,
    base_s: float = 12.0,
    base_speed: float = 36.0,
    buffer_s: float = 3.0,
) -> float:
    """Seconds to record for a full revolution (larger TURNSPEED = slower)."""
    scale = max(rotate_speed, 1.0) / base_speed
    return max(8.0, base_s * scale * (rotation_deg / 360.0) + buffer_s)


def _colorize_scalar(field: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
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
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def compute_rgb_temporal_stats(
    rgb_frames: list[np.ndarray],
    *,
    max_frames: int = 0,
    detrend: bool = True,
) -> dict[str, np.ndarray]:
    """Per-pixel gray mean, std, MAD, range across frames."""
    if len(rgb_frames) < 3:
        raise ValueError("Need at least 3 RGB frames.")

    frames = rgb_frames
    if max_frames > 0 and len(frames) > max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]

    gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames],
        axis=0,
    )
    # Per-frame mean removal helps when AE is on; skip when exposure is fixed.
    if detrend:
        gray_stack = gray_stack - gray_stack.mean(axis=(1, 2), keepdims=True)
    n = gray_stack.shape[0]

    gray_mean = np.mean(gray_stack, axis=0).astype(np.float32)
    gray_std = np.std(gray_stack, axis=0).astype(np.float32)
    abs_dev = np.abs(gray_stack - gray_mean[None, :, :])
    gray_mad = np.median(abs_dev, axis=0).astype(np.float32)
    gray_range = (np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0)).astype(
        np.float32
    )

    bgr = np.stack([f.astype(np.float32) for f in frames], axis=0)
    ch_std = np.max(np.std(bgr, axis=0), axis=2).astype(np.float32)

    median_gray = np.median(gray_stack, axis=0).astype(np.uint8)
    median_rgb = cv2.cvtColor(median_gray, cv2.COLOR_GRAY2BGR)

    return {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "gray_mad": gray_mad,
        "gray_range": gray_range,
        "ch_std": ch_std,
        "median_rgb": median_rgb,
        "frame_count": np.array([n]),
    }


def render_rgb_temporal_diagnostics(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean_vis = _colorize_scalar(stats["gray_mean"])
    std_vis = _colorize_scalar(stats["gray_std"])
    mad_vis = _colorize_scalar(stats["gray_mad"])
    range_vis = _colorize_scalar(stats["gray_range"])
    ch_std_vis = _colorize_scalar(stats["ch_std"])

    # Static vs motion thresholds on gray_std
    gs = stats["gray_std"]
    pos = gs[gs > 0]
    tau = float(max(2.5, np.percentile(pos, 25))) if pos.size > 100 else 3.0
    static = gs < tau
    motion = gs >= tau

    overlay = stats["median_rgb"].copy()
    overlay[static] = (
        overlay[static].astype(np.float32) * 0.5
        + np.array([40, 40, 200], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    overlay[motion] = (
        overlay[motion].astype(np.float32) * 0.45
        + np.array([40, 220, 40], dtype=np.float32) * 0.55
    ).astype(np.uint8)

    stats_combo = np.hstack([mean_vis, std_vis, mad_vis])
    delta_combo = np.hstack([range_vis, std_vis, ch_std_vis])
    full_combo = np.hstack([stats["median_rgb"], delta_combo])

    return {
        "rgb_mean_std_mad": stats_combo,
        "rgb_range_std_chstd": delta_combo,
        "rgb_pixel_delta": delta_combo,
        "rgb_full_combo": full_combo,
        "rgb_std_vis": std_vis,
        "rgb_range_vis": range_vis,
        "rgb_overlay": overlay,
        "tau_gray_std": np.array([tau]),
    }


def _clean_binary_mask(
    mask_u8: np.ndarray,
    *,
    morph_kernel: int = 9,
    fill_holes: bool = True,
    open_iters: int = 1,
) -> np.ndarray:
    k = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if open_iters > 0:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if fill_holes:
        u8 = _fill_holes_binary(u8)
    return u8


def _turbo_warm_mask(
    field: np.ndarray,
    *,
    percentile: float = 18.0,
    floor: float = 0.12,
    use_abs: bool = False,
) -> tuple[np.ndarray, float]:
    """Pixels warm in turbo panel (p3-p97 norm >= percentile/100)."""
    data = np.abs(field) if use_abs else field.copy()
    valid = data > 0
    vals = data[valid]
    if vals.size < 100:
        tau = float(floor)
        return data >= tau, tau
    lo = float(np.percentile(vals, 3))
    hi = float(np.percentile(vals, 97))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(data.shape, dtype=np.float32)
    norm[valid] = (data[valid] - lo) / (hi - lo)
    tau_norm = max(floor, percentile / 100.0)
    tau = lo + tau_norm * (hi - lo)
    return norm >= tau_norm, tau


def build_rgb_mean_include_mask(
    stats: dict[str, np.ndarray],
    *,
    mean_percentile: float = 18.0,
    mean_floor: float = 4.0,
    morph_kernel: int = 11,
    fill_holes: bool = True,
    center_bias: bool = True,
    min_area_frac: float = 0.002,
) -> RgbVideoMaskResult:
    """Mask from temporal mean panel: warm = object corridor, black = static (exclude).

    Matches the left panel of rgb_mean_std_mad.jpg (turbo colormap on gray_mean).
    """
    mean = stats["gray_mean"]
    mad = stats["gray_mad"]
    std = stats["gray_std"]
    n = int(stats.get("frame_count", np.array([0]))[0])

    # Match turbo panel: black = bottom of p3-p97 range, warm = top.
    include_raw, tau_mean = _turbo_warm_mask(
        mean, percentile=mean_percentile, floor=0.12, use_abs=True
    )

    u8 = _clean_binary_mask(
        include_raw.astype(np.uint8) * 255,
        morph_kernel=morph_kernel,
        fill_holes=fill_holes,
        open_iters=1,
    )

    if center_bias:
        h, w = u8.shape
        nlab, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
            (u8 > 0).astype(np.uint8), connectivity=8
        )
        min_area = max(64, int(h * w * min_area_frac))
        include = _pick_center_component(
            labels, stats_cc, centroids, cx=w * 0.5, cy=h * 0.52, min_area=min_area
        )
    else:
        include = u8 > 0

    return RgbVideoMaskResult(
        include=include,
        gray_mad=mad,
        gray_mean=mean,
        gray_std=std,
        tau_mean=tau_mean,
        preview_frame_count=n,
        method="rgb_video_mean",
    )


def build_rgb_mean_mad_include_mask(
    stats: dict[str, np.ndarray],
    *,
    mean_percentile: float = 18.0,
    mad_percentile: float = 20.0,
    morph_kernel: int = 11,
    fill_holes: bool = True,
    center_bias: bool = True,
    min_area_frac: float = 0.002,
) -> RgbVideoMaskResult:
    """Intersection of mean + MAD turbo-warm regions.

    Mean drops the platform; MAD drops static background and reflection glints.
    """
    mean = stats["gray_mean"]
    mad = stats["gray_mad"]
    std = stats["gray_std"]
    n = int(stats.get("frame_count", np.array([0]))[0])

    mean_warm, tau_mean = _turbo_warm_mask(
        mean, percentile=mean_percentile, floor=0.12, use_abs=True
    )
    mad_warm, tau_mad = _turbo_warm_mask(mad, percentile=mad_percentile, floor=0.12)

    include_raw = mean_warm & mad_warm

    u8 = _clean_binary_mask(
        include_raw.astype(np.uint8) * 255,
        morph_kernel=morph_kernel,
        fill_holes=fill_holes,
        open_iters=0,
    )

    if center_bias:
        h, w = u8.shape
        nlab, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
            (u8 > 0).astype(np.uint8), connectivity=8
        )
        min_area = max(64, int(h * w * min_area_frac))
        include = _pick_center_component(
            labels, stats_cc, centroids, cx=w * 0.5, cy=h * 0.52, min_area=min_area
        )
    else:
        include = u8 > 0

    if fill_holes:
        filled = _fill_holes_binary(include.astype(np.uint8) * 255)
        include = filled > 0

    return RgbVideoMaskResult(
        include=include,
        gray_mad=mad,
        gray_mean=mean,
        gray_std=std,
        tau_mean=tau_mean,
        tau_mad=tau_mad,
        preview_frame_count=n,
        method="rgb_video_mean_mad",
    )


def render_mean_mad_mask_combo(
    stats: dict[str, np.ndarray],
    result: RgbVideoMaskResult,
    *,
    mean_percentile: float = 18.0,
    mad_percentile: float = 20.0,
) -> np.ndarray:
    """mean warm | mad warm | intersection overlay."""
    mean_warm, _ = _turbo_warm_mask(
        stats["gray_mean"], percentile=mean_percentile, floor=0.12, use_abs=True
    )
    mad_warm, _ = _turbo_warm_mask(stats["gray_mad"], percentile=mad_percentile, floor=0.12)
    diag = render_rgb_temporal_diagnostics(stats)
    w = diag["rgb_mean_std_mad"].shape[1] // 3
    mean_vis = diag["rgb_mean_std_mad"][:, :w]
    mad_vis = diag["rgb_mean_std_mad"][:, 2 * w : 3 * w]
    base = stats["median_rgb"].copy()
    overlay = base.copy()
    inc = result.include
    overlay[inc] = (
        overlay[inc].astype(np.float32) * 0.35 + np.array([40, 220, 80], dtype=np.float32) * 0.65
    ).astype(np.uint8)
    overlay[~inc] = (
        overlay[~inc].astype(np.float32) * 0.5 + np.array([40, 40, 200], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    # tint mean-only warm cyan, mad-only warm magenta on small previews
    mean_only = mean_vis.copy()
    mean_only[~mean_warm] = (mean_only[~mean_warm] * 0.25).astype(np.uint8)
    mad_only = mad_vis.copy()
    mad_only[~mad_warm] = (mad_only[~mad_warm] * 0.25).astype(np.uint8)
    return np.hstack([mean_only, mad_only, overlay])


def build_rgb_video_include_mask(
    stats: dict[str, np.ndarray],
    *,
    mad_percentile: float = 20.0,
    std_percentile: float = 18.0,
    mad_floor: float = 2.0,
    std_floor: float = 2.5,
    morph_kernel: int = 9,
    fill_holes: bool = True,
    center_bias: bool = True,
    min_area_frac: float = 0.002,
) -> RgbVideoMaskResult:
    """Build include mask from temporal MAD (primary) and std (fills low-MAD object patches).

    Matches the turbo heatmaps: warm yellow/red/green = motion/object corridor,
    dark blue = static background. Morphological close + flood-fill keeps blue
    holes that sit inside a red boundary (uniform object surface during spin).
    """
    mad = stats["gray_mad"]
    std = stats["gray_std"]
    mean = stats["gray_mean"]
    n = int(stats.get("frame_count", np.array([0]))[0])

    pct_mad = mad_percentile + (25.0 if n > 400 else 12.0 if n > 200 else 0.0)
    pct_std = std_percentile + (15.0 if n > 400 else 8.0 if n > 200 else 0.0)

    pos_mad = mad[mad > 0]
    tau_mad = float(max(mad_floor, np.percentile(pos_mad, pct_mad))) if pos_mad.size > 100 else mad_floor
    pos_std = std[std > 0]
    tau_std = float(max(std_floor, np.percentile(pos_std, pct_std))) if pos_std.size > 100 else std_floor

    # MAD panel (right) drops platform; std recovers low-MAD patches inside the corridor.
    core = mad >= tau_mad
    k = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core_u8 = cv2.morphologyEx(core.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        core_u8 = _fill_holes_binary(core_u8)
    corridor = cv2.dilate(core_u8, kernel, iterations=2) > 0
    std_fill = (std >= tau_std) & corridor
    include = (core_u8 > 0) | std_fill
    u8 = include.astype(np.uint8) * 255

    if center_bias:
        h, w = u8.shape
        nlab, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
            (u8 > 0).astype(np.uint8), connectivity=8
        )
        min_area = max(64, int(h * w * min_area_frac))
        center = _pick_center_component(
            labels, stats_cc, centroids, cx=w * 0.5, cy=h * 0.52, min_area=min_area
        )
        include = center
    else:
        include = u8 > 0

    return RgbVideoMaskResult(
        include=include,
        gray_mad=mad,
        gray_mean=mean,
        gray_std=std,
        tau_mad=tau_mad,
        tau_std=tau_std,
        preview_frame_count=n,
        method="rgb_video_mad",
    )


def render_rgb_video_mask_diagnostics(
    stats: dict[str, np.ndarray],
    result: RgbVideoMaskResult,
) -> dict[str, np.ndarray]:
    """Overlay include mask on median RGB + turbo panels."""
    diag = render_rgb_temporal_diagnostics(stats)
    median = stats["median_rgb"]
    overlay = median.copy()
    inc = result.include
    overlay[inc] = (
        overlay[inc].astype(np.float32) * 0.4 + np.array([40, 220, 80], dtype=np.float32) * 0.6
    ).astype(np.uint8)
    overlay[~inc] = (
        overlay[~inc].astype(np.float32) * 0.55 + np.array([40, 40, 200], dtype=np.float32) * 0.45
    ).astype(np.uint8)

    mask_u8 = (inc.astype(np.uint8)) * 255
    return {
        "rgb_mask_overlay": overlay,
        "rgb_mask_include": mask_u8,
        "rgb_mask_combo": np.hstack([median, overlay]),
        "rgb_full_combo": diag["rgb_full_combo"],
        "rgb_mean_std_mad": diag["rgb_mean_std_mad"],
    }
