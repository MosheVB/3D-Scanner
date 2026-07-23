"""RealSense .bag recording and playback — native SDK capture at sensor rate."""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from scanner.camera_lock import acquire as acquire_camera_lock
from scanner.camera_lock import release as release_camera_lock
from scanner.rgb_video_camera import (
    RgbProfile,
    _require_rs,
    configure_rgb_sensor,
    configure_rgb_sensor_fixed,
    lock_rgb_exposure,
)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # type: ignore[assignment]


def record_rgb_bag(
    profile: RgbProfile,
    bag_path: Path,
    duration_s: float,
    *,
    lock_holder: str = "rgb_bag_record",
    warmup_frames: int = 30,
    fixed_exposure: bool = True,
    exposure_us: float = 28000.0,
) -> int:
    """Record color stream to a .bag file; drain pipeline without per-frame Python work.

    Intel's recorder writes frames in the SDK. Python only calls wait_for_frames() to
    keep the USB queue drained (no numpy copies during capture).
    """
    _require_rs()
    assert rs is not None
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    if bag_path.exists():
        bag_path.unlink()
    record_path = bag_path
    if record_path.suffix.lower() == ".bag":
        record_path = record_path.with_suffix(".db3")

    acquire_camera_lock(lock_holder, command="rgb_bag_record")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, profile.width, profile.height, rs.format.bgr8, profile.fps
    )
    config.enable_record_to_file(str(record_path))
    drained = [0]
    stop_ev = threading.Event()

    def _drain() -> None:
        while not stop_ev.is_set():
            try:
                pipeline.wait_for_frames(timeout_ms=500)
                drained[0] += 1
            except RuntimeError:
                if stop_ev.is_set():
                    break

    try:
        profile = pipeline.start(config)
        if fixed_exposure:
            configure_rgb_sensor_fixed(profile.get_device(), exposure_us=exposure_us, warmup_s=0.5)
        else:
            configure_rgb_sensor(profile.get_device(), settle_s=1.5)
        for _ in range(max(5, warmup_frames)):
            pipeline.wait_for_frames(timeout_ms=5000)

        t0 = time.monotonic()
        worker = threading.Thread(target=_drain, name="rs-drain", daemon=True)
        worker.start()
        time.sleep(duration_s)
        stop_ev.set()
        worker.join(timeout=2.0)
        elapsed = max(time.monotonic() - t0, 1e-3)
        n = drained[0]
        print(
            f"Bag record: drained {n} frames in {elapsed:.2f}s "
            f"({n / elapsed:.1f} fps drain rate) -> {record_path.name}"
        )
        return n
    finally:
        stop_ev.set()
        try:
            pipeline.stop()
        except RuntimeError:
            pass
        del pipeline
        gc.collect()
        release_camera_lock()


def count_bag_color_frames(bag_path: Path) -> int:
    """Count color frames in a bag (non-real-time playback)."""
    n = 0
    for _ in iter_bag_color_frames(bag_path):
        n += 1
    return n


def iter_bag_color_frames(bag_path: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames from a recorded bag/db3 without dropping frames."""
    _require_rs()
    assert rs is not None
    path = bag_path
    if path.suffix.lower() == ".bag" and path.with_suffix(".db3").is_file():
        path = path.with_suffix(".db3")
    config = rs.config()
    rs.config.enable_device_from_file(config, str(path), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError:
                break
            color = frames.get_color_frame()
            if not color:
                continue
            # Copy so librealsense can recycle frame buffers during playback.
            yield np.asanyarray(color.get_data()).copy()
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass
        del pipeline
        gc.collect()


def load_bag_color_frames(bag_path: Path) -> list[np.ndarray]:
    return list(iter_bag_color_frames(bag_path))
