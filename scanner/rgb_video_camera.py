"""RGB-only RealSense capture for high-fps rotation video (no depth stream)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from scanner.camera_lock import acquire as acquire_camera_lock
from scanner.camera_lock import release as release_camera_lock

try:
    import pyrealsense2 as rs
except ImportError as exc:
    rs = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RgbProfile:
    width: int
    height: int
    fps: int


def _require_rs() -> None:
    if rs is None:
        raise ImportError("pyrealsense2 not installed") from _IMPORT_ERROR


def list_color_profiles() -> list[RgbProfile]:
    """All color stream modes advertised by the first RealSense device."""
    _require_rs()
    assert rs is not None
    out: list[RgbProfile] = []
    seen: set[tuple[int, int, int]] = set()
    for dev in rs.context().query_devices():
        for sensor in dev.sensors:
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                vp = profile.as_video_stream_profile()
                key = (vp.width(), vp.height(), vp.fps())
                if key in seen:
                    continue
                seen.add(key)
                out.append(RgbProfile(vp.width(), vp.height(), vp.fps()))
    out.sort(key=lambda p: (-p.fps, -(p.width * p.height)))
    return out


def pick_rgb_profile(*, target_fps: int = 90) -> RgbProfile:
    """Highest resolution among modes at *target_fps*, else best fps at max res."""
    profiles = list_color_profiles()
    if not profiles:
        raise RuntimeError("No RealSense color profiles found.")

    at_fps = [p for p in profiles if p.fps == target_fps]
    if at_fps:
        best = max(at_fps, key=lambda p: p.width * p.height)
        return best

    # nearest: max resolution then highest fps
    best = max(profiles, key=lambda p: (p.width * p.height, p.fps))
    return best


def pick_mask_rgb_profile(
    *,
    width: int = 848,
    height: int = 480,
    fps: int = 15,
) -> RgbProfile:
    """Stable mask-preview profile: fixed fps, prefer requested resolution."""
    profiles = list_color_profiles()
    if not profiles:
        raise RuntimeError("No RealSense color profiles found.")
    for p in profiles:
        if p.width == width and p.height == height and p.fps == fps:
            return p
    at_fps = [p for p in profiles if p.fps == fps]
    if at_fps:
        return max(at_fps, key=lambda x: x.width * x.height)
    return pick_rgb_profile(target_fps=fps)


def configure_rgb_sensor(device: Any, *, settle_s: float = 1.5) -> None:
    """Enable auto-exposure on D405 stereo module (RGB shares this sensor).

    Manual exposure or AE off commonly caps color delivery at ~15 fps even when
    the stream profile requests 30/60/90 fps.
    """
    assert rs is not None
    from scanner.rs_options import apply_depth_options

    depth_sensor = device.first_depth_sensor()
    try:
        apply_depth_options(
            rs,
            depth_sensor,
            {"visual_preset": "short_range", "auto_exposure": True, "gain": 16},
        )
    except (RuntimeError, ValueError):
        pass

    for sensor in device.sensors:
        try:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        except RuntimeError:
            pass
        try:
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 1.0)
        except RuntimeError:
            pass

    if settle_s > 0:
        time.sleep(settle_s)


def configure_rgb_sensor_fixed(
    device: Any,
    *,
    exposure_us: float = 28000.0,
    gain: float = 16.0,
    warmup_s: float = 0.5,
) -> None:
    """Fixed exposure for mask preview: stable frames, native 15 fps delivery."""
    assert rs is not None
    from scanner.rs_options import apply_depth_options

    depth_sensor = device.first_depth_sensor()
    try:
        apply_depth_options(
            rs,
            depth_sensor,
            {
                "visual_preset": "short_range",
                "auto_exposure": False,
                "exposure_us": exposure_us,
                "gain": gain,
            },
        )
    except (RuntimeError, ValueError):
        try:
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if depth_sensor.supports(rs.option.exposure):
                depth_sensor.set_option(rs.option.exposure, float(exposure_us))
            if depth_sensor.supports(rs.option.gain):
                depth_sensor.set_option(rs.option.gain, float(gain))
        except RuntimeError:
            pass
    if warmup_s > 0:
        time.sleep(warmup_s)


def lock_rgb_exposure(device: Any) -> float | None:
    """Freeze exposure after AE settle so long spins do not drift global brightness."""
    assert rs is not None
    depth_sensor = device.first_depth_sensor()
    exp_us: float | None = None
    try:
        if depth_sensor.supports(rs.option.exposure):
            exp_us = float(depth_sensor.get_option(rs.option.exposure))
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        if exp_us is not None and depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, exp_us)
    except RuntimeError:
        return None
    return exp_us


class RgbOnlyCamera:
    """Color stream only - frees USB bandwidth for high fps."""

    def __init__(
        self,
        profile: RgbProfile,
        *,
        lock_holder: str = "rgb_video",
    ) -> None:
        _require_rs()
        self._profile = profile
        self._lock_holder = lock_holder
        self._lock_held = False
        self._pipeline: Any = None
        self._running = False

    @property
    def profile(self) -> RgbProfile:
        return self._profile

    def start(self) -> None:
        assert rs is not None
        if not self._lock_held:
            acquire_camera_lock(self._lock_holder, command="rgb_video")
            self._lock_held = True
        self.stop()
        p = self._profile
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, p.width, p.height, rs.format.bgr8, p.fps)
        profile = pipeline.start(config)
        configure_rgb_sensor(profile.get_device(), settle_s=1.0)
        self._pipeline = pipeline
        self._running = True

    def stop(self) -> None:
        pipeline = self._pipeline
        self._running = False
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            del pipeline
            gc.collect()
        if self._lock_held:
            release_camera_lock()
            self._lock_held = False

    def read_rgb(self, *, block: bool = True) -> np.ndarray | None:
        if not self._running or self._pipeline is None:
            return None
        try:
            if block:
                frames = self._pipeline.wait_for_frames(timeout_ms=200)
            else:
                frames = self._pipeline.poll_for_frames()
                if not frames:
                    return None
        except RuntimeError:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())
