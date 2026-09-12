"""Pick the sharpest frame in each rotation-degree bin from video capture."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BinPick:
    """One winning frame index for a rotation-degree bin."""

    source_index: int
    rotation_deg: float
    quality: float
    bin_index: int


@dataclass(frozen=True)
class VideoFramePick:
    """One saved pose from a continuous rotation recording."""

    source_index: int
    rgb: np.ndarray
    depth_mm: np.ndarray | None
    rotation_deg: float
    quality: float
    bin_index: int


def frame_sharpness(rgb: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Comparable only at a fixed scale."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_quality_score(rgb: np.ndarray, depth_mm: np.ndarray | None) -> float:
    """Higher = sharper RGB and more valid object-range depth.

    Depth-less sources (phone video, any RGB-only camera) score on sharpness
    alone — the depth-coverage term is simply absent rather than zero.
    """
    sharp = frame_sharpness(rgb)
    if depth_mm is None:
        return sharp
    valid = (depth_mm > 0) & (depth_mm >= 80) & (depth_mm <= 300)
    depth_cov = float(valid.sum())
    return sharp + depth_cov * 0.01


def _timestamp_rotation_deg(
    ts: float,
    *,
    spin_t0: float,
    spin_duration: float,
    rotation_deg_total: float,
) -> float:
    frac = (ts - spin_t0) / max(spin_duration, 1e-3)
    return min(float(frac * rotation_deg_total), float(rotation_deg_total))


def select_best_indices_per_degree(
    qualities: list[float],
    frame_times: list[float],
    *,
    spin_t0: float | None = None,
    spin_duration: float | None = None,
    rotation_deg_total: float = 360.0,
    pose_step_deg: float = 15.0,
) -> list[BinPick]:
    """Keep the highest-quality frame index per pose_step_deg bin.

    Frames are assigned an angle by linear interpolation over the spin, so this
    assumes constant angular velocity between spin_t0 and spin_t0 + duration.
    Takes scores rather than frames so callers can stream a large source and
    keep only one frame in memory at a time.
    """
    n = len(qualities)
    if n == 0:
        return []

    t0 = frame_times[0] if spin_t0 is None else spin_t0
    t1 = frame_times[-1] if n > 1 else t0 + 1.0
    duration = (t1 - t0) if spin_duration is None else spin_duration
    duration = max(float(duration), 1e-3)

    step = max(float(pose_step_deg), 1e-3)
    n_bins = max(1, int(round(rotation_deg_total / step)))
    buckets: dict[int, list[tuple[int, float, float]]] = {}

    for i, (quality, ts) in enumerate(zip(qualities, frame_times)):
        rot_est = _timestamp_rotation_deg(
            ts, spin_t0=t0, spin_duration=duration, rotation_deg_total=rotation_deg_total
        )
        bin_idx = int(rot_est / step) % n_bins
        bin_idx = min(bin_idx, n_bins - 1)
        buckets.setdefault(bin_idx, []).append((i, quality, rot_est))

    picks: list[BinPick] = []
    for bin_idx in sorted(buckets):
        best_i, best_q, best_rot = max(buckets[bin_idx], key=lambda item: item[1])
        picks.append(
            BinPick(
                source_index=best_i,
                rotation_deg=min(float(best_rot), float(rotation_deg_total)),
                quality=best_q,
                bin_index=bin_idx,
            )
        )
    return picks


def select_best_frames_per_degree(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray] | None,
    frame_times: list[float],
    *,
    spin_t0: float | None = None,
    spin_duration: float | None = None,
    rotation_deg_total: float = 360.0,
    pose_step_deg: float = 15.0,
) -> list[VideoFramePick]:
    """Keep one frame per pose_step_deg bin, preferring sharpness + depth coverage.

    Pass depth_frames=None for an RGB-only recording; picks then carry
    depth_mm=None and rank on sharpness alone.
    """
    n = len(rgb_frames)
    if n == 0:
        return []

    if depth_frames is None:
        depths: list[np.ndarray | None] = [None] * n
    else:
        if len(depth_frames) != n:
            raise ValueError(
                f"depth_frames has {len(depth_frames)} entries but rgb_frames has {n}"
            )
        depths = list(depth_frames)

    qualities = [_frame_quality_score(rgb, d) for rgb, d in zip(rgb_frames, depths)]
    picks = select_best_indices_per_degree(
        qualities,
        frame_times,
        spin_t0=spin_t0,
        spin_duration=spin_duration,
        rotation_deg_total=rotation_deg_total,
        pose_step_deg=pose_step_deg,
    )
    return [
        VideoFramePick(
            source_index=p.source_index,
            rgb=rgb_frames[p.source_index],
            depth_mm=depths[p.source_index],
            rotation_deg=p.rotation_deg,
            quality=p.quality,
            bin_index=p.bin_index,
        )
        for p in picks
    ]
