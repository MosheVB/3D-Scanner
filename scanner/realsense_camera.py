"""Intel RealSense D405 — short-range RGB-D capture via pyrealsense2."""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from scanner.camera_lock import acquire as acquire_camera_lock
from scanner.camera_lock import release as release_camera_lock
from scanner.config import CAPTURE_FPS, RGB_HEIGHT, RGB_WIDTH
from scanner.rs_options import (
    D405_FALLBACK_PRESETS,
    apply_depth_options,
    depth_option_support,
    read_camera_telemetry,
    read_depth_options,
)

# https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/
MACOS_REALSENSE_SDK_DOC = (
    "https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/"
)

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    rs = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# D405-friendly modes (width, height, fps) — try in order
_STREAM_PROFILES: list[tuple[int, int, int]] = [
    (1280, 720, 5),
    (848, 480, 10),
    (RGB_WIDTH, RGB_HEIGHT, int(CAPTURE_FPS)),
    (848, 480, 30),
    (640, 480, 15),
]


def _require_rs() -> None:
    if rs is None:
        raise ImportError(
            "pyrealsense2 is not installed. On macOS: pip install pyrealsense2-macosx"
        ) from _IMPORT_ERROR


def macos_realsense_usb_visible() -> bool:
    """True if a RealSense camera shows up in IORegistry (USB plugged in)."""
    if platform.system() != "Darwin":
        return False
    try:
        proc = subprocess.run(
            ["ioreg", "-p", "IOUSB", "-l", "-w", "0"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    text = proc.stdout or ""
    return "RealSense" in text or "Depth Camera 405" in text


def _pyrealsense_device_count() -> int | None:
    if rs is None:
        return None
    try:
        return int(rs.context().devices.size())
    except Exception:
        return None


def macos_realsense_sudo_hint() -> str:
    py = sys.executable
    n = _pyrealsense_device_count()
    if os.geteuid() == 0:
        return (
            "\nRunning as root but RealSense still failed. librealsense reports "
            f"{n if n is not None else '?'} device(s) while USB shows the D405 — "
            "macOS is likely holding the camera (UVC) or blocking access.\n"
            "Run: python scripts/realsense_mac_diag.py\n"
            "Then: quit all camera apps, replug USB, try Terminal.app (not IDE), "
            "or use --camera oak / Linux for reliable capture."
        )
    if n == 0 and macos_realsense_usb_visible():
        return (
            "\nUSB shows the D405 but librealsense sees 0 devices (driver conflict). "
            "On macOS 12+ Intel requires sudo for libusb (see "
            f"{MACOS_REALSENSE_SDK_DOC}). Try:\n"
            f"  python scripts/realsense_mac_diag.py\n"
            f"  ./scripts/realsense_sudo.sh scripts/realsense_probe.py\n"
            "  sudo /opt/homebrew/bin/rs-enumerate-devices -s   # brew install librealsense\n"
            "Quit FaceTime/Zoom (RealSense Viewer is unsupported on macOS); "
            "grant Camera privacy to Terminal; replug on USB 3. "
            "If still 0 devices, use OAK or Linux — docs/realsense-macos.md"
        )
    return (
        "\nmacOS blocked USB access to the D405. On macOS 12+ (Monterey+), Intel "
        "requires sudo for librealsense USB tools — see "
        f"{MACOS_REALSENSE_SDK_DOC}\n"
        f"  ./scripts/realsense_sudo.sh scripts/realsense_probe.py\n"
        f"  sudo -E {py} scripts/realsense_probe.py   # same, in Terminal.app\n"
        "If probe still fails: python scripts/realsense_mac_diag.py"
    )


def macos_release_uvc(*, verbose: bool = False) -> bool:
    """Attempt to release the UVC device held by macOS's UVCAssistant daemon.

    On macOS 12+, UVCAssistant is a protected system extension that requires
    root to kill (and launchd restarts it immediately).  The only reliable way
    to make a RealSense/UVC camera available to a non-UVC SDK like librealsense
    is to run the capturing process itself with sudo, which causes macOS to
    transfer device ownership from UVCAssistant to the root process.

    This function tries a best-effort kill (only succeeds if already running as
    root) and returns True if the process was signalled.  In all other cases
    use: sudo ./scripts/realsense_sudo.sh scripts/realsense_preview.py
    """
    if platform.system() != "Darwin":
        return False

    # Find the PID via pgrep (searches full command line for system extensions)
    try:
        pgrep = subprocess.run(
            ["pgrep", "-f", "UVCAssistant"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        pids = [int(p) for p in pgrep.stdout.split() if p.strip().isdigit()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pids = []

    if not pids:
        if verbose:
            print("UVCAssistant is not running — camera may already be free.")
        return False

    if os.geteuid() != 0:
        if verbose:
            print(
                "UVCAssistant is running but killing it requires root.\n"
                "Run the scanner with sudo to transfer camera ownership:\n"
                "  ./scripts/realsense_sudo.sh scripts/realsense_preview.py\n"
                "  ./scripts/realsense_sudo.sh -m scanner capture --mode handheld --name test"
            )
        return False

    # Running as root — kill each PID and give macOS time to release the device
    killed_any = False
    for pid in pids:
        try:
            os.kill(pid, 9)
            killed_any = True
        except (ProcessLookupError, PermissionError):
            pass

    if killed_any:
        if verbose:
            print("Sent SIGKILL to UVCAssistant — waiting for device release...")
        time.sleep(1.0)

    return killed_any


def format_realsense_access_error(detail: str) -> str:
    msg = (
        f"Could not access RealSense ({detail}). "
        "Close RealSense Viewer, unplug/replug USB, wait 5s, retry."
    )
    if platform.system() == "Darwin" and (
        "power state" in detail.lower() or macos_realsense_usb_visible()
    ):
        msg += macos_realsense_sudo_hint()
    return msg


def _intrinsics_from_profile(profile: rs.video_stream_profile) -> list[list[float]]:
    intr = profile.get_intrinsics()
    return [
        [float(intr.fx), 0.0, float(intr.ppx)],
        [0.0, float(intr.fy), float(intr.ppy)],
        [0.0, 0.0, 1.0],
    ]


_OPEN_TEST_SUBPROCESS = r"""
import json, os, sys
serial = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
profiles = [
    (848, 480, 30), (640, 480, 30), (640, 480, 15), (848, 480, 15),
]
try:
    import pyrealsense2 as rs
    last = None
    for w, h, fps in profiles:
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            profile = pipeline.start(config)
            dev = profile.get_device()
            name = dev.get_info(rs.camera_info.name)
            pipeline.stop()
            print(json.dumps({"ok": True, "name": name, "width": w, "height": h, "fps": fps}))
            sys.stdout.flush()
            os._exit(0)
        except RuntimeError as exc:
            last = str(exc)
            try:
                pipeline.stop()
            except RuntimeError:
                pass
    print(json.dumps({"ok": False, "error": last or "could not start"}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
sys.stdout.flush()
os._exit(0)
"""

_ENUM_SUBPROCESS = r"""
import json, os, sys
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append({
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
        })
    print(json.dumps({"ok": True, "devices": devices}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
sys.stdout.flush()
os._exit(0)
"""


def _list_realsense_devices_inprocess() -> list[dict[str, str]]:
    _require_rs()
    assert rs is not None
    out: list[dict[str, str]] = []
    ctx = rs.context()
    for dev in ctx.query_devices():
        out.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
            }
        )
    return out


def _list_realsense_devices_subprocess() -> list[dict[str, str]]:
    proc = subprocess.run(
        [sys.executable, "-c", _ENUM_SUBPROCESS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(
            f"RealSense enumeration failed ({err}). "
            "Quit RealSense Viewer, unplug/replug USB, wait 5s."
        )
    try:
        payload = json.loads(line[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"RealSense enumeration failed (invalid response). "
            "Quit RealSense Viewer, unplug/replug USB, wait 5s."
        ) from exc
    if not payload.get("ok"):
        raise RuntimeError(
            format_realsense_access_error(str(payload.get("error", "unknown")))
        )
    return list(payload.get("devices") or [])


def _probe_open_subprocess(serial: str | None = None) -> dict[str, Any]:
    args = [sys.executable, "-c", _OPEN_TEST_SUBPROCESS]
    if serial:
        args.append(serial)
    else:
        args.append("")
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    line = (proc.stdout or "").strip().splitlines()
    if proc.returncode in (-11, 139):
        return {
            "ok": False,
            "error": "failed to set power state (device busy or USB; quit RealSense Viewer)",
        }
    if not line:
        return {"ok": False, "error": (proc.stderr or "").strip() or f"exit {proc.returncode}"}
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid probe response"}


def list_realsense_devices() -> list[dict[str, str]]:
    """Enumerate devices without starting a pipeline (for diagnostics)."""
    _require_rs()
    try:
        if platform.system() == "Darwin":
            return _list_realsense_devices_subprocess()
        return _list_realsense_devices_inprocess()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(format_realsense_access_error(str(exc))) from exc


class RealSenseD405:
    """Aligned color + depth streams for close-range scanning (7 cm+)."""

    def __init__(
        self,
        *,
        width: int = RGB_WIDTH,
        height: int = RGB_HEIGHT,
        fps: int = int(CAPTURE_FPS),
        serial: str | None = None,
        lock_holder: str = "capture",
    ) -> None:
        _require_rs()
        self._width = width
        self._height = height
        self._fps = fps
        self._serial = serial
        self._lock_holder = lock_holder
        self._lock_held = False
        self._pipeline: Any = None
        self._config: Any = None
        self._align: Any = None
        self._decimation: Any = None
        self._spatial: Any = None
        self._temporal: Any = None
        self._hole_fill: Any = None
        self._depth_scale = 0.001
        self._intrinsics: list[list[float]] = []
        self._device_name = "Intel RealSense D405"
        self._device_serial = ""
        self._running = False
        self._device: Any = None
        self._depth_sensor: Any = None
        self._filters: dict[str, Any] = {
            "spatial_enabled": True,
            "spatial_magnitude": 2,
            "temporal_enabled": True,
            "hole_fill_enabled": True,
            "decimation_enabled": False,
            "decimation_magnitude": 1,
        }
        self._depth_session: dict[str, Any] = {
            "visual_preset": "medium_density",
            "auto_exposure": True,
            "gain": 16.0,
        }

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def intrinsics(self) -> list[list[float]]:
        return self._intrinsics

    @property
    def label(self) -> str:
        return self._device_name

    @property
    def fps(self) -> int:
        return self._fps

    def host_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "os": platform.system(),
            "arch": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "camera_backend": "realsense",
            "device_name": self._device_name,
            "device_serial": self._device_serial,
            "resolution": [self._width, self._height],
            "fps": self._fps,
            "depth_scale_mm": self._depth_scale * 1000.0,
        }
        if meta["arch"] in ("arm64", "aarch64"):
            meta["host_mode"] = "apple_silicon_usb"
        return meta

    def _release(self) -> None:
        """Drop native handles so the USB device is freed immediately."""
        # Pull refs out of self first so Python refcount drops to 0 after stop()
        pipeline = self._pipeline
        align = self._align
        self._running = False
        self._pipeline = None
        self._config = None
        self._align = None
        self._decimation = None
        self._spatial = None
        self._temporal = None
        self._hole_fill = None
        self._device = None
        self._depth_sensor = None

        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            del pipeline

        if align is not None:
            del align

        # Force C++ destructors to run now rather than at next GC cycle.
        # On macOS this is what actually releases the USB/UVC device handle.
        gc.collect()

    def start(self) -> None:
        _require_rs()
        assert rs is not None
        if not self._lock_held:
            cmd = f"python -m scanner {self._lock_holder}" if self._lock_holder == "web" else ""
            acquire_camera_lock(self._lock_holder, command=cmd or f"PID {os.getpid()}")
            self._lock_held = True
        self._release()

        if platform.system() == "Darwin":
            probe = _probe_open_subprocess(self._serial)
            if not probe.get("ok"):
                if self._lock_held:
                    release_camera_lock()
                    self._lock_held = False
                raise RuntimeError(
                    format_realsense_access_error(str(probe.get("error", "unknown")))
                )
            # Give macOS time to release the USB device from the probe subprocess
            # before this process tries to claim it.  Without this pause, the OS
            # occasionally returns "failed to set power state" on the first open.
            time.sleep(0.3)

        profiles = [(self._width, self._height, self._fps)] + [
            p for p in _STREAM_PROFILES
            if p != (self._width, self._height, self._fps)
        ]
        last_err: Exception | None = None

        for w, h, fps in profiles:
            try:
                self._start_profile(w, h, fps)
                return
            except RuntimeError as exc:
                last_err = exc
                self._release()

        if self._lock_held:
            release_camera_lock()
            self._lock_held = False
        raise RuntimeError(
            "Could not start RealSense D405. "
            "Close RealSense Viewer, unplug/replug USB, wait 5s, retry.\n"
            f"Last error: {last_err}"
        ) from last_err

    def _start_profile(self, width: int, height: int, fps: int) -> None:
        assert rs is not None
        pipeline = rs.pipeline()
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = pipeline.start(config)
        except RuntimeError:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            raise
        device = profile.get_device()
        self._device_name = device.get_info(rs.camera_info.name)
        self._device_serial = device.get_info(rs.camera_info.serial_number)

        if "D405" not in self._device_name.upper() and self._serial is None:
            pipeline.stop()
            raise RuntimeError(
                f"Expected Intel RealSense D405, got {self._device_name}. "
                "Use --serial or disconnect other RealSense cameras."
            )

        depth_sensor = device.first_depth_sensor()
        self._device = device
        self._depth_sensor = depth_sensor
        self._depth_scale = depth_sensor.get_depth_scale()
        if depth_sensor.supports(rs.option.depth_units):
            try:
                depth_sensor.set_option(rs.option.depth_units, 0.0001)
                self._depth_scale = depth_sensor.get_depth_scale()
            except RuntimeError:
                pass

        try:
            apply_depth_options(rs, depth_sensor, self._depth_session)
        except (RuntimeError, ValueError):
            if depth_sensor.supports(rs.option.visual_preset):
                for preset in (
                    getattr(rs.l500_visual_preset, "short_range", None),
                    getattr(rs.rs400_visual_preset, "short_range", None),
                ):
                    if preset is None:
                        continue
                    try:
                        depth_sensor.set_option(rs.option.visual_preset, preset)
                        break
                    except RuntimeError:
                        continue
            self._configure_sensor_options(device)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = pipeline
        self._config = config
        self._align = rs.align(rs.stream.color)
        self._decimation = rs.decimation_filter()
        mag = max(1, int(self._filters.get("decimation_magnitude", 1)))
        self._decimation.set_option(rs.option.filter_magnitude, mag)
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(
            rs.option.filter_magnitude, int(self._filters.get("spatial_magnitude", 2))
        )
        self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial.set_option(rs.option.filter_smooth_delta, 20)
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._temporal.set_option(rs.option.filter_smooth_delta, 20)
        self._hole_fill = rs.hole_filling_filter()
        self._hole_fill.set_option(rs.option.holes_fill, 1)
        self._intrinsics = _intrinsics_from_profile(color_profile)
        self._running = True

    def _apply_depth_filters(self, depth_frame: Any) -> Any:
        assert rs is not None
        if self._decimation is None or self._spatial is None:
            return depth_frame
        if self._filters.get("decimation_enabled") and int(
            self._filters.get("decimation_magnitude", 1)
        ) > 1:
            self._decimation.set_option(
                rs.option.filter_magnitude,
                int(self._filters["decimation_magnitude"]),
            )
            depth_frame = self._decimation.process(depth_frame)
        if self._filters.get("spatial_enabled", True):
            self._spatial.set_option(
                rs.option.filter_magnitude,
                int(self._filters.get("spatial_magnitude", 2)),
            )
            depth_frame = self._spatial.process(depth_frame)
        if self._filters.get("temporal_enabled", True):
            depth_frame = self._temporal.process(depth_frame)
        if self._filters.get("hole_fill_enabled", True):
            depth_frame = self._hole_fill.process(depth_frame)
        return depth_frame

    def get_options(self) -> dict[str, Any]:
        """Current capture options (session + live hardware when connected)."""
        assert rs is not None
        depth_block: dict[str, Any] | None = None
        if self._depth_sensor is not None and self._running:
            try:
                depth_block = read_depth_options(rs, self._depth_sensor)
            except RuntimeError:
                depth_block = None
        sensor = self._depth_sensor if self._running else None
        supported = depth_option_support(rs, sensor)
        if depth_block is None:
            ae = self._depth_session.get("auto_exposure", True)
            depth_block = {
                "visual_preset": {
                    "value": self._depth_session.get("visual_preset"),
                    "choices": list(D405_FALLBACK_PRESETS),
                    "supported": supported["visual_preset"],
                },
                "auto_exposure": ae,
                "exposure_us": {
                    "value": self._depth_session.get("exposure_us"),
                    "range": None,
                    "editable": supported["exposure"] and not ae,
                    "supported": supported["exposure"],
                },
                "gain": {
                    "value": self._depth_session.get("gain"),
                    "range": None,
                    "supported": supported["gain"],
                },
                "laser_power": {"value": None, "range": None, "supported": supported["laser"]},
                "emitter_enabled": {"value": None, "supported": supported["emitter"]},
            }
        return {
            "connected": self._running,
            "supported": supported,
            "depth": depth_block,
            "filters": dict(self._filters),
            "notes": [
                "D405 RGB and depth share the stereo module — exposure and gain affect both.",
                "D405 has no IR dot projector — laser and emitter controls are unavailable.",
                "Settings persist for this web server session only.",
            ],
        }

    def patch_options(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply option changes live; stores session values for reconnect."""
        assert rs is not None
        depth_updates = updates.get("depth") or {}
        for key in (
            "visual_preset",
            "auto_exposure",
            "exposure_us",
            "gain",
            "laser_power",
            "emitter_enabled",
        ):
            if key in depth_updates:
                self._depth_session[key] = depth_updates[key]

        filter_updates = updates.get("filters") or {}
        for key, val in filter_updates.items():
            if key in self._filters:
                self._filters[key] = val

        if self._depth_sensor is not None and self._running and depth_updates:
            apply_depth_options(rs, self._depth_sensor, depth_updates)

        return self.get_options()

    def get_camera_telemetry(self) -> dict[str, Any]:
        if not self._running or self._depth_sensor is None:
            return {"connected": False, "temperatures_c": {}, "fan_rpm": None}
        assert rs is not None
        try:
            data = read_camera_telemetry(rs, self._depth_sensor)
        except RuntimeError:
            data = {"temperatures_c": {}, "fan_rpm": None}
        data["connected"] = True
        return data

    def _configure_sensor_options(self, device: Any) -> None:
        """Match RealSense Viewer defaults where practical (AE, gain)."""
        assert rs is not None
        for sensor in device.sensors:
            try:
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            except RuntimeError:
                pass
            try:
                if sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, 16.0)
            except RuntimeError:
                pass
            try:
                if sensor.supports(rs.option.auto_exposure_priority):
                    sensor.set_option(rs.option.auto_exposure_priority, 1.0)
            except RuntimeError:
                pass

    def prepare_turntable_capture(
        self,
        *,
        warmup_frames: int | None = None,
        gain: float | None = None,
    ) -> dict[str, Any]:
        """AE warmup, lock exposure, then tune for dim or bright scenes."""
        from scanner.config import (
            TURNTABLE_AE_WARMUP_FRAMES,
            TURNTABLE_GAIN,
            TURNTABLE_GAIN_BRIGHT,
            TURNTABLE_HIGHLIGHT_FRAC_MAX,
            TURNTABLE_HOLE_FILL,
            TURNTABLE_RGB_MEAN_MAX,
            TURNTABLE_RGB_MEAN_MIN,
            TURNTABLE_RGB_TARGET,
            TURNTABLE_TEMPORAL_FILTER,
            TURNTABLE_VISUAL_PRESET,
        )
        from scanner.rgb_video_camera import lock_rgb_exposure
        from scanner.rs_options import apply_depth_options

        assert rs is not None
        wf = int(warmup_frames if warmup_frames is not None else TURNTABLE_AE_WARMUP_FRAMES)

        self.patch_options(
            {
                "depth": {"visual_preset": TURNTABLE_VISUAL_PRESET},
                "filters": {
                    "hole_fill_enabled": TURNTABLE_HOLE_FILL,
                    "temporal_enabled": TURNTABLE_TEMPORAL_FILTER,
                    "spatial_enabled": True,
                    "decimation_enabled": False,
                },
            }
        )

        def _rgb_metrics(rgb: np.ndarray | None) -> tuple[float, float]:
            if rgb is None:
                return 0.0, 0.0
            gray = rgb if rgb.ndim == 2 else np.mean(rgb, axis=2)
            return float(np.mean(gray)), float(np.mean(gray >= 245.0))

        # Quick probe: use lower gain when room lights already brighten the scene.
        probe_rgb: np.ndarray | None = None
        for _ in range(min(12, wf)):
            pair = self.read_frame(block=True)
            if pair is not None:
                probe_rgb, _ = pair
        probe_mean, _ = _rgb_metrics(probe_rgb)
        g = float(gain if gain is not None else TURNTABLE_GAIN)
        if gain is None and probe_mean > 140.0:
            g = TURNTABLE_GAIN_BRIGHT

        self.patch_options(
            {
                "depth": {
                    "visual_preset": TURNTABLE_VISUAL_PRESET,
                    "auto_exposure": True,
                    "gain": g,
                }
            }
        )

        last_rgb: np.ndarray | None = probe_rgb
        for _ in range(max(0, wf - 12)):
            pair = self.read_frame(block=True)
            if pair is not None:
                last_rgb, _ = pair

        exp_us = lock_rgb_exposure(self._device) if self._device is not None else None
        rgb_mean, highlight_frac = _rgb_metrics(last_rgb)
        used_gain = g

        if self._depth_sensor is None:
            return {
                "warmup_frames": wf,
                "exposure_us": exp_us,
                "gain": used_gain,
                "rgb_mean": round(rgb_mean, 1),
                "highlight_frac": round(highlight_frac, 4),
                "auto_exposure_locked": True,
            }

        er = self._depth_sensor.get_option_range(rs.option.exposure)
        gr = self._depth_sensor.get_option_range(rs.option.gain)

        def _set_manual(exp: float, gain_val: float) -> None:
            nonlocal exp_us, used_gain
            apply_depth_options(
                rs,
                self._depth_sensor,
                {
                    "auto_exposure": False,
                    "exposure_us": exp,
                    "gain": gain_val,
                },
            )
            exp_us = exp
            used_gain = gain_val

        for _ in range(20):
            need_boost = rgb_mean < TURNTABLE_RGB_MEAN_MIN
            need_cut = (
                rgb_mean > TURNTABLE_RGB_MEAN_MAX
                or highlight_frac > TURNTABLE_HIGHLIGHT_FRAC_MAX
            )
            if not need_boost and not need_cut:
                break

            base_exp = float(exp_us if exp_us is not None else er.default)
            if need_boost:
                target = TURNTABLE_RGB_TARGET
                new_exp = min(
                    float(er.max),
                    max(float(er.min), base_exp * (target / max(rgb_mean, 1.0))),
                )
                new_gain = min(float(gr.max), max(float(gr.min), used_gain * 1.35))
            elif highlight_frac > TURNTABLE_HIGHLIGHT_FRAC_MAX:
                # White cube can clip while mean stays below TURNTABLE_RGB_MEAN_MAX.
                over = highlight_frac / max(TURNTABLE_HIGHLIGHT_FRAC_MAX, 1e-6)
                scale = 0.72 ** min(4.0, over)
                new_exp = max(float(er.min), base_exp * scale)
                new_gain = max(float(gr.min), used_gain * scale)
            else:
                target = min(TURNTABLE_RGB_TARGET, TURNTABLE_RGB_MEAN_MAX)
                scale = target / max(rgb_mean, 1.0)
                new_exp = max(float(er.min), base_exp * scale)
                new_gain = max(float(gr.min), used_gain * 0.88)

            _set_manual(new_exp, new_gain)
            for _ in range(10):
                pair = self.read_frame(block=True)
                if pair is not None:
                    last_rgb, _ = pair
            rgb_mean, highlight_frac = _rgb_metrics(last_rgb)

        return {
            "warmup_frames": wf,
            "exposure_us": exp_us,
            "gain": used_gain,
            "rgb_mean": round(rgb_mean, 1),
            "highlight_frac": round(highlight_frac, 4),
            "auto_exposure_locked": True,
        }

    def prepare_video_spin_capture(
        self,
        *,
        warmup_frames: int = 18,
        gain: float | None = None,
    ) -> dict[str, Any]:
        """Fast shutter for continuous rotation: keep AE on for ~30 fps delivery.

        Turntable step capture locks long exposure for depth SNR; that caps aligned
        RGB-D at ~6 fps on the D405 and worsens motion blur during a spin.
        """
        from scanner.config import TURNTABLE_GAIN, TURNTABLE_GAIN_BRIGHT, TURNTABLE_RGB_MEAN_MIN
        from scanner.rs_options import apply_depth_options

        g0 = float(gain if gain is not None else TURNTABLE_GAIN_BRIGHT)
        self.patch_options(
            {
                "depth": {
                    "visual_preset": "short_range",
                    "auto_exposure": True,
                    "gain": g0,
                },
                "filters": {
                    "hole_fill_enabled": False,
                    "temporal_enabled": False,
                    "spatial_enabled": False,
                    "decimation_enabled": False,
                },
            }
        )

        last_rgb: np.ndarray | None = None
        for _ in range(max(4, int(warmup_frames))):
            pair = self.read_frame(block=True, apply_filters=False)
            if pair is not None:
                last_rgb, _ = pair

        rgb_mean = 0.0
        if last_rgb is not None:
            gray = last_rgb if last_rgb.ndim == 2 else np.mean(last_rgb, axis=2)
            rgb_mean = float(np.mean(gray))

        if (
            rgb_mean < TURNTABLE_RGB_MEAN_MIN
            and self._depth_sensor is not None
            and rs is not None
        ):
            boost = max(g0, float(TURNTABLE_GAIN))
            apply_depth_options(
                rs,
                self._depth_sensor,
                {"auto_exposure": True, "gain": boost},
            )
            for _ in range(12):
                pair = self.read_frame(block=True, apply_filters=False)
                if pair is not None:
                    last_rgb, _ = pair
            if last_rgb is not None:
                gray = last_rgb if last_rgb.ndim == 2 else np.mean(last_rgb, axis=2)
                rgb_mean = float(np.mean(gray))
            g0 = boost

        exp_us: float | None = None
        if self._depth_sensor is not None and self._depth_sensor.supports(rs.option.exposure):
            try:
                exp_us = float(self._depth_sensor.get_option(rs.option.exposure))
            except RuntimeError:
                exp_us = None

        return {
            "warmup_frames": warmup_frames,
            "exposure_us": exp_us,
            "gain": g0,
            "rgb_mean": round(rgb_mean, 1),
            "auto_exposure_locked": False,
            "mode": "video_spin",
        }

    def stop(self) -> None:
        self._release()
        if self._lock_held:
            release_camera_lock()
            self._lock_held = False

    def __del__(self) -> None:
        """Safety-net: release USB handles if stop() was never called."""
        try:
            self._release()
            if self._lock_held:
                release_camera_lock()
                self._lock_held = False
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._running

    def read_frame(
        self, *, block: bool = True, apply_filters: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._running or self._pipeline is None or self._align is None:
            return None
        try:
            if block:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            else:
                frames = self._pipeline.poll_for_frames()
                if not frames:
                    return None
        except RuntimeError:
            return None

        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        use_filters = (
            self._filters.get("spatial_enabled", True)
            if apply_filters is None
            else apply_filters
        )
        if use_filters and self._decimation is not None and self._spatial is not None:
            depth_frame = self._apply_depth_filters(depth_frame)

        depth_raw = np.asanyarray(depth_frame.get_data())
        rgb = np.asanyarray(color_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        depth_mm = np.zeros(depth_raw.shape, dtype=np.uint16)
        valid = depth_raw > 0
        depth_mm[valid] = (depth_m[valid] * 1000.0).astype(np.uint16)
        return rgb, depth_mm

