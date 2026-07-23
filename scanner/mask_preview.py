"""Quick 360° preview spin to auto-build a motion union mask."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from scanner.session_mask import SessionMask, build_mask_from_preview, save_session_mask
from scanner.turntable import RevopointTurntable


class FrameSource(Protocol):
    def read_frame(self, *, block: bool = True) -> tuple | None: ...


@dataclass
class MaskPreviewConfig:
    """Fast rotation-only preview before the main scan."""

    steps: int = 24
    rotate_step_deg: float = 15.0
    rotate_wait_s: float = 1.25
    tilt_deg: float = 0.0
    tilt_wait_s: float = 2.0
    home_after: bool = True


def _grab_frame(
    cam: FrameSource, *, timeout_s: float = 3.0
) -> tuple | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pair = cam.read_frame(block=False)
        if pair is not None:
            return pair
        time.sleep(0.02)
    return None


def collect_preview_frames(
    frame_source: FrameSource,
    tt: RevopointTurntable,
    cfg: MaskPreviewConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[list, list]:
    if log is None:
        log = print

    log(
        f"Mask preview: {cfg.steps} steps × {cfg.rotate_step_deg:.0f}° "
        f"@ tilt {cfg.tilt_deg:+.0f}° …"
    )
    tt.set_tilt(cfg.tilt_deg, wait_s=cfg.tilt_wait_s)

    rgb_frames: list = []
    depth_frames: list = []

    for i in range(cfg.steps):
        if i > 0:
            tt.rotate_step(cfg.rotate_step_deg, wait_s=cfg.rotate_wait_s)
        pair = _grab_frame(frame_source)
        if pair is None:
            log(f"  preview step {i + 1}/{cfg.steps}: no frame")
            continue
        rgb, depth = pair
        rgb_frames.append(rgb)
        depth_frames.append(depth)

    if cfg.home_after:
        tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
        tt.rotate_to_zero(wait_s=8.0)

    log(f"Mask preview captured {len(rgb_frames)} frames.")
    return rgb_frames, depth_frames


def run_mask_preview(
    frame_source: FrameSource,
    tt: RevopointTurntable,
    session_root,
    cfg: MaskPreviewConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> SessionMask | None:
    """Spin, compute motion mask, save under session_root."""
    from pathlib import Path

    root = Path(session_root)
    if cfg is None:
        cfg = MaskPreviewConfig()

    rgb_frames, depth_frames = collect_preview_frames(
        frame_source, tt, cfg, log=log
    )
    if len(rgb_frames) < 3:
        if log:
            log("Mask preview failed — too few frames; continuing without mask.")
        return None

    result = build_mask_from_preview(rgb_frames, depth_frames)
    spec = save_session_mask(root, result, rgb_frames)
    if log:
        lo, hi = spec.depth_band_mm
        log(
            f"Motion mask: {spec.extra.get('include_pixels', 0):,} include px, "
            f"depth band {lo:.0f}–{hi:.0f} mm "
            f"(τ depth {spec.tau_depth_mm:.1f} mm, τ rgb {spec.tau_rgb:.1f})"
        )
        log(f"Saved mask previews → {root / 'mask_motion_combo.jpg'}")
    return spec
