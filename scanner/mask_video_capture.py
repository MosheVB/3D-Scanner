"""Capture depth/RGB buffer during continuous turntable rotation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from scanner.turntable import RevopointTurntable


class FrameSource(Protocol):
    def read_frame(
        self, *, block: bool = True, apply_filters: bool | None = None
    ) -> tuple | None: ...


@dataclass
class VideoCaptureConfig:
    rotation_deg: float = 360.0
    capture_timeout_s: float = 18.0
    min_frames: int = 30
    max_frames: int = 600
    rotate_speed: float = 12.0
    tilt_deg: float = 0.0
    tilt_wait_s: float = 2.0
    home_after: bool = True


def capture_rotation_video(
    frame_source: FrameSource,
    tt: RevopointTurntable,
    cfg: VideoCaptureConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[list, list, float, list[float], float]:
    """Spin turntable while polling camera.

    Returns (rgb, depth, fps, frame_times_s, spin_start_monotonic).
    """
    if cfg is None:
        cfg = VideoCaptureConfig()
    if log is None:
        log = print

    tt.set_rotate_speed(cfg.rotate_speed)
    tt.set_tilt(cfg.tilt_deg, wait_s=cfg.tilt_wait_s)

    rgb_frames: list = []
    depth_frames: list = []
    frame_times: list[float] = []
    t0 = time.monotonic()

    log(
        f"Video capture: rotate {cfg.rotation_deg:.0f} deg "
        f"(max {cfg.capture_timeout_s:.0f}s, up to {cfg.max_frames} frames)..."
    )
    spin_t0 = time.monotonic()
    tt.send(f"+CT,TURNANGLE={cfg.rotation_deg:.4f};")

    deadline = spin_t0 + cfg.capture_timeout_s
    while time.monotonic() < deadline and len(rgb_frames) < cfg.max_frames:
        pair = frame_source.read_frame(block=True, apply_filters=False)
        if pair is not None:
            rgb, depth = pair
            rgb_frames.append(rgb)
            depth_frames.append(depth)
            frame_times.append(time.monotonic())
        else:
            time.sleep(0.001)

    elapsed = max(time.monotonic() - spin_t0, 1e-3)
    fps = len(rgb_frames) / elapsed

    if cfg.home_after:
        log("Homing turntable after video capture...")
        tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
        tt.rotate_to_zero(wait_s=10.0)

    log(f"Captured {len(rgb_frames)} frames in {elapsed:.1f}s ({fps:.1f} fps effective).")
    if len(rgb_frames) < cfg.min_frames:
        raise RuntimeError(
            f"Too few frames ({len(rgb_frames)} < {cfg.min_frames}). "
            "Check camera stream and turntable rotation."
        )
    return rgb_frames, depth_frames, fps, frame_times, spin_t0


def probe_best_fps(
    *,
    width: int = 640,
    height: int = 480,
    candidates: tuple[int, ...] = (30, 15, 10, 5),
    lock_holder: str = "mask_video",
) -> tuple[int, object]:
    """Return (fps, started RealSenseD405) using the first working profile."""
    from scanner.realsense_camera import RealSenseD405

    last_err: Exception | None = None
    for fps in candidates:
        cam = RealSenseD405(fps=fps, width=width, height=height, lock_holder=lock_holder)
        try:
            cam.start()
            print(f"Camera started at {width}x{height} @ {fps} fps")
            return fps, cam
        except RuntimeError as exc:
            last_err = exc
            try:
                cam.stop()
            except Exception:
                pass
    raise RuntimeError(f"No RealSense profile worked. Last: {last_err}")
