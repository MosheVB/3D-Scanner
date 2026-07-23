"""RGB rotation video mask preview before turntable scan."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from scanner.config import MASK_RGB_RECORD_S, TURNTABLE_ROTATE_SPEED
from scanner.mask_rgb_video import (
    build_rgb_mean_mad_include_mask,
    compute_rgb_temporal_stats,
    render_mean_mad_mask_combo,
)
from scanner.session_mask import SessionMask, save_rgb_video_session_mask
from scanner.turntable import RevopointTurntable


class FrameSource(Protocol):
    def read_frame(self, *, block: bool = True) -> tuple | None: ...


@dataclass
class RgbVideoMaskConfig:
    rotation_deg: float = 360.0
    record_s: float = MASK_RGB_RECORD_S
    rotate_speed: float = TURNTABLE_ROTATE_SPEED
    tilt_deg: float = 0.0
    tilt_wait_s: float = 2.0
    home_after: bool = True
    detrend: bool = True


def run_rgb_video_mask_preview(
    frame_source: FrameSource,
    tt: RevopointTurntable,
    session_root: Path,
    cfg: RgbVideoMaskConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> SessionMask | None:
    """One 360 spin, mean+MAD mask, save mask_include.png under session."""
    if cfg is None:
        cfg = RgbVideoMaskConfig()
    if log is None:
        log = print

    tt.set_rotate_speed(cfg.rotate_speed)
    tt.set_tilt(cfg.tilt_deg, wait_s=cfg.tilt_wait_s)

    rgb_frames: list = []
    log(
        f"RGB video mask: rotate {cfg.rotation_deg:.0f} deg, "
        f"record {cfg.record_s:.0f}s..."
    )
    tt.send(f"+CT,TURNANGLE={cfg.rotation_deg:.4f};")
    time.sleep(0.15)

    deadline = time.monotonic() + cfg.record_s
    while time.monotonic() < deadline:
        pair = frame_source.read_frame(block=True)
        if pair is not None:
            rgb_frames.append(pair[0])

    if cfg.home_after:
        log("Homing turntable after RGB mask preview...")
        tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
        tt.rotate_to_zero(wait_s=10.0)

    if len(rgb_frames) < 30:
        log("RGB video mask: too few frames; continuing without mask.")
        return None

    stats = compute_rgb_temporal_stats(rgb_frames, detrend=cfg.detrend)
    result = build_rgb_mean_mad_include_mask(stats)
    combo = render_mean_mad_mask_combo(stats, result)
    spec = save_rgb_video_session_mask(session_root, result, combo_bgr=combo)
    log(
        f"RGB video mask: {spec.extra.get('include_pixels', 0):,} include px "
        f"({result.include.mean():.0%}), method={result.method}"
    )
    return spec
