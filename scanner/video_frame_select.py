"""Pick the sharpest frame in each rotation-degree bin from video capture."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFramePick:
    """One saved pose from a continuous rotation recording."""

    source_index: int
    rgb: np.ndarray
    depth_mm: np.ndarray
    rotation_deg: float
    quality: float
    bin_index: int


def _frame_quality_score(rgb: np.ndarray, depth_mm: np.ndarray) -> float:
    """Higher = sharper RGB and more valid object-range depth."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
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


def select_best_frames_per_degree(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    frame_times: list[float],
    *,
    spin_t0: float | None = None,
    spin_duration: float | None = None,
    rotation_deg_total: float = 360.0,
    pose_step_deg: float = 15.0,
) -> list[VideoFramePick]:
    """Keep one frame per pose_step_deg bin, preferring sharpness + depth coverage."""
    n = len(rgb_frames)
    if n == 0:
        return []

    t0 = frame_times[0] if spin_t0 is None else spin_t0
    t1 = frame_times[-1] if n > 1 else t0 + 1.0
    duration = (t1 - t0) if spin_duration is None else spin_duration
    duration = max(float(duration), 1e-3)

    step = max(float(pose_step_deg), 1e-3)
    n_bins = max(1, int(round(rotation_deg_total / step)))
    buckets: dict[int, list[tuple[int, float, float]]] = {}

    for i, (rgb, depth_mm, ts) in enumerate(zip(rgb_frames, depth_frames, frame_times)):
        rot_est = _timestamp_rotation_deg(
            ts, spin_t0=t0, spin_duration=duration, rotation_deg_total=rotation_deg_total
        )
        bin_idx = int(rot_est / step) % n_bins
        bin_idx = min(bin_idx, n_bins - 1)
        quality = _frame_quality_score(rgb, depth_mm)
        buckets.setdefault(bin_idx, []).append((i, quality, rot_est))

    picks: list[VideoFramePick] = []
    for bin_idx in sorted(buckets):
        candidates = buckets[bin_idx]
        best_i, best_q, best_rot = max(candidates, key=lambda item: item[1])
        rotation_deg = min(float(best_rot), float(rotation_deg_total))
        picks.append(
            VideoFramePick(
                source_index=best_i,
                rgb=rgb_frames[best_i],
                depth_mm=depth_frames[best_i],
                rotation_deg=rotation_deg,
                quality=best_q,
                bin_index=bin_idx,
            )
        )
    return picks
