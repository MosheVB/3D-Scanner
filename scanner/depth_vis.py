"""Depth visualization and export (actual mm values, not false-color maps)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scanner.config import D405_MIN_DEPTH_MM, LIVE_DEPTH_BAND_MM


def colorize_depth_mm(
    depth_mm: np.ndarray,
    *,
    focus_mm: float | None = None,
    band_mm: float = LIVE_DEPTH_BAND_MM,
    use_d405_range: bool = False,
) -> np.ndarray:
    """Turbo colormap; optional focus band or D405 working-range scale."""
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)

    if focus_mm is not None:
        lo = max(float(D405_MIN_DEPTH_MM), focus_mm - band_mm)
        hi = focus_mm + band_mm
    elif use_d405_range:
        lo = float(D405_MIN_DEPTH_MM)
        hi = 740.0
    else:
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))

    if hi <= lo:
        hi = lo + 1.0

    scaled = np.clip(
        (depth_mm.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255
    ).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def depth_mm_uint16_image(depth_mm: np.ndarray) -> np.ndarray:
    """16-bit grayscale: pixel value = depth in mm (0 = no reading)."""
    return depth_mm.astype(np.uint16)


def depth_mm_linear_gray8(depth_mm: np.ndarray) -> np.ndarray:
    """8-bit gray: linear map of valid depth mm (0 = invalid). No colormap."""
    out = np.zeros(depth_mm.shape, dtype=np.uint8)
    valid = depth_mm > 0
    if not np.any(valid):
        return out
    lo = float(depth_mm[valid].min())
    hi = float(depth_mm[valid].max())
    span = max(hi - lo, 1.0)
    out[valid] = np.clip(
        (depth_mm[valid].astype(np.float32) - lo) / span * 255.0, 0, 255
    ).astype(np.uint8)
    return out


def render_depth_mm_grid(
    depth_mm: np.ndarray,
    *,
    step: int = 24,
    font_scale: float = 0.45,
    invalid_label: str = "-",
) -> np.ndarray:
    """BGR image with the depth in mm printed at each sampled pixel."""
    h, w = depth_mm.shape
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    for y in range(0, h, step):
        for x in range(0, w, step):
            z = int(depth_mm[y, x])
            label = invalid_label if z <= 0 else str(z)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = x + max(0, (step - tw) // 2)
            ty = y + (step + th) // 2
            cv2.putText(panel, label, (tx, ty), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return panel


def save_depth_mm_artifacts(
    depth_mm: np.ndarray,
    out_dir: Path,
    *,
    grid_step: int | None = None,
) -> dict[str, Path]:
    """Save raw depth arrays and mm-per-pixel views (no turbo colormap)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    npy = out_dir / "depth_mm.npy"
    np.save(str(npy), depth_mm)
    paths["npy"] = npy

    u16 = out_dir / "depth_mm_uint16.png"
    cv2.imwrite(str(u16), depth_mm_uint16_image(depth_mm))
    paths["uint16"] = u16

    gray = out_dir / "depth_linear.png"
    cv2.imwrite(str(gray), depth_mm_linear_gray8(depth_mm))
    paths["linear"] = gray

    h, w = depth_mm.shape
    step = grid_step or max(16, min(w, h) // 40)
    grid = out_dir / "depth_mm_grid.jpg"
    cv2.imwrite(str(grid), render_depth_mm_grid(depth_mm, step=step))
    paths["grid"] = grid

    return paths
