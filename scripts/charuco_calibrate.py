#!/usr/bin/env python3
"""ChArUco-board intrinsic calibration for the Intel RealSense D405.

Usage
-----
    python scripts/charuco_calibrate.py

Workflow
--------
    With turntable + ANTHROPIC_API_KEY (PID mode):
    1. Camera warms up, turntable homes tilt axis.
    2. Claude observes each settled frame and returns JSON commands:
       capture | hold | move (rotate_deg incremental, tilt_deg absolute).
    3. Turntable only moves when Claude says move — no scripted sweep.
    4. SPACE — manual distance shot when prompted.
    5. Auto-calibrates once min_frames captured.

    Without Claude: legacy auto-capture on board motion (manual turntable).

Controls (OpenCV window)
------------------------
    SPACE   - manual capture / distance shot
    c       - calibrate immediately (once enough frames captured)
    p       - save printable board image
    q / ESC - quit without calibrating

The board
---------
Board (pre-configured for your 6.89" x 5.71" purchased board):
  11 x 9 squares, 15 mm square side, 11 mm marker side, DICT_5X5_100.

If you use a different board, override with --cols, --rows,
--square-mm, --marker-mm.  Wrong physical dimensions cause calibration
errors - measure with calipers if unsure.

calib.io boards (even row count) may need --legacy to match OpenCV's
pre-4.6 corner ordering.  11 x 9 has an odd row count so --legacy is
NOT normally needed, but add it if board detection looks wrong.

Output
------
calibration/realsense_d405_<timestamp>.json  - intrinsics + distortion + RMS
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not found. Install with: pip install pyrealsense2", file=sys.stderr)
    sys.exit(1)

try:
    from scanner.turntable import TILT_MAX, TILT_MIN, RevopointTurntable
    _HAVE_TURNTABLE = True
except Exception:
    TILT_MIN = -30.0
    TILT_MAX = 30.0
    _HAVE_TURNTABLE = False

try:
    import anthropic as _anthropic
    _HAVE_ANTHROPIC = True
except ImportError:
    _HAVE_ANTHROPIC = False


# ---------------------------------------------------------------------------
# Claude PID controller — brain outputs commands; turntable is the actuator
# ---------------------------------------------------------------------------

@dataclass
class CalibrationCommand:
    """One decision from Claude. Actuator validates and executes."""

    action: str          # "capture" | "hold" | "move"
    rotate_deg: float = 0.0
    tilt_deg: float | None = None   # absolute tilt target; None = no tilt change
    reason: str = ""

    def summary(self) -> str:
        if self.action == "move":
            parts = []
            if abs(self.rotate_deg) >= 0.5:
                parts.append(f"rot {self.rotate_deg:+.0f}°")
            if self.tilt_deg is not None:
                parts.append(f"tilt→{self.tilt_deg:+.0f}°")
            move = ", ".join(parts) if parts else "no-op"
            return f"MOVE {move}: {self.reason}"
        return f"{self.action.upper()}: {self.reason}"


def _parse_calibration_command(text: str) -> CalibrationCommand | None:
    """Extract JSON command from Claude's reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    action = str(data.get("action", "hold")).lower().strip()
    if action not in ("capture", "hold", "move"):
        action = "hold"

    rotate = float(data.get("rotate_deg", 0) or 0)
    tilt_raw = data.get("tilt_deg")
    tilt = float(tilt_raw) if tilt_raw is not None else None

    return CalibrationCommand(
        action=action,
        rotate_deg=rotate,
        tilt_deg=tilt,
        reason=str(data.get("reason", ""))[:120],
    )


def _board_ready_for_capture(
    *,
    board_detected: bool,
    n_corners: int,
    incidence_deg: float | None,
    sharpness: float,
    min_sharpness: float,
    min_corners: int = 12,
    max_incidence: float = 38.0,
) -> bool:
    if not board_detected:
        return False
    if n_corners < min_corners:
        return False
    if sharpness < min_sharpness:
        return False
    if incidence_deg is not None and incidence_deg > max_incidence:
        return False
    return True


class _PidSupervisor:
    """Actuator-side guard: stops rotate loops, forces tilt + capture cadence."""

    _TILT_LADDER = (-22.0, -12.0, 0.0, 12.0, 22.0)
    MAX_ROTATE_ONLY = 1          # max rotate-only moves between captures
    MAX_CUMULATIVE_ROTATE = 30.0 # degrees since last capture before force capture/tilt

    def __init__(self) -> None:
        self.moves_since_capture = 0
        self.consecutive_rotate_only = 0
        self.cumulative_rotate_deg = 0.0
        self.coverage_at_last_capture = 0.0
        self.tilts_visited: set[float] = {0.0}
        self._ladder_idx = 0

    def on_capture(self, coverage_pct: float, tilt_deg: float) -> None:
        self.moves_since_capture = 0
        self.consecutive_rotate_only = 0
        self.cumulative_rotate_deg = 0.0
        self.coverage_at_last_capture = coverage_pct
        self.tilts_visited.add(round(tilt_deg, 1))

    def on_move(self, cmd: CalibrationCommand, tilt_before: float) -> None:
        if cmd.action != "move":
            return
        self.moves_since_capture += 1
        tilt_changed = (
            cmd.tilt_deg is not None and abs(cmd.tilt_deg - tilt_before) > 0.5
        )
        if tilt_changed:
            self.consecutive_rotate_only = 0
            self.cumulative_rotate_deg = 0.0
            if cmd.tilt_deg is not None:
                self.tilts_visited.add(round(cmd.tilt_deg, 1))
        elif abs(cmd.rotate_deg) >= 0.5:
            self.consecutive_rotate_only += 1
            self.cumulative_rotate_deg += abs(cmd.rotate_deg)

    def _next_tilt(self, current: float) -> float | None:
        for t in self._TILT_LADDER:
            if abs(t - current) > 4.0 and round(t, 1) not in self.tilts_visited:
                return t
        for t in self._TILT_LADDER:
            if abs(t - current) > 4.0:
                return t
        return None

    def override(
        self,
        cmd: CalibrationCommand,
        *,
        board_ok: bool,
        coverage_pct: float,
        tilt_deg: float,
    ) -> CalibrationCommand:
        if cmd.action != "move" or not board_ok:
            return cmd

        coverage_stuck = (
            self.moves_since_capture >= 2
            and coverage_pct <= self.coverage_at_last_capture + 3.0
        )
        too_much_rotate = (
            self.consecutive_rotate_only >= self.MAX_ROTATE_ONLY
            or self.cumulative_rotate_deg >= self.MAX_CUMULATIVE_ROTATE
            or coverage_stuck
        )

        if too_much_rotate:
            if cmd.tilt_deg is None and abs(cmd.rotate_deg) >= 0.5:
                next_tilt = self._next_tilt(tilt_deg)
                if next_tilt is not None:
                    return CalibrationCommand(
                        "move",
                        rotate_deg=0.0,
                        tilt_deg=next_tilt,
                        reason=f"supervisor: tilt bed to {next_tilt:+.0f}° (rotate not helping)",
                    )
                return CalibrationCommand(
                    "capture",
                    reason="supervisor: good board — capture now",
                )
        return cmd

    def tilt_after_capture(self, current_tilt: float) -> CalibrationCommand | None:
        """Queue the next bed tilt — called automatically after each capture."""
        next_tilt = self._next_tilt(current_tilt)
        if next_tilt is None:
            return None
        return CalibrationCommand(
            "move",
            rotate_deg=0.0,
            tilt_deg=next_tilt,
            reason=f"supervisor: next bed angle {next_tilt:+.0f}°",
        )

    def context_line(self) -> str:
        return (
            f"Moves since capture: {self.moves_since_capture}. "
            f"Consecutive rotate-only: {self.consecutive_rotate_only}. "
            f"Cumulative rotate: {self.cumulative_rotate_deg:.0f}°. "
            f"Bed tilts visited: {sorted(int(t) for t in self.tilts_visited)}. "
            "Bed tilt advances automatically after each capture. "
        )


class _ViewpointTracker:
    """3D viewpoint diversity — better calibration metric than 2D grid alone."""

    def __init__(self) -> None:
        self._incidences: list[float] = []
        self._centroids: list[tuple[float, float]] = []

    def add(self, incidence_deg: float | None, corners: np.ndarray) -> None:
        if incidence_deg is not None:
            self._incidences.append(incidence_deg)
        cen = corners.mean(axis=0).flatten()
        self._centroids.append((float(cen[0]), float(cen[1])))

    def score_pct(self, image_w: int) -> float:
        if len(self._centroids) < 2:
            return 0.0
        xs = [p[0] for p in self._centroids]
        cen_span_pct = (max(xs) - min(xs)) / max(image_w, 1) * 100.0
        inc_span = (
            max(self._incidences) - min(self._incidences)
            if len(self._incidences) >= 2 else 0.0
        )
        cen_score = min(50.0, cen_span_pct / 20.0 * 50.0)
        inc_score = min(50.0, inc_span / 20.0 * 50.0)
        return cen_score + inc_score


class ClaudeCalibrationController:
    """PID brain: observe frame → emit one JSON command per settle cycle."""

    _SYSTEM = (
        "You control a Revopoint dual-axis turntable for ChArUco camera calibration.\n"
        "Reply with ONLY a JSON object (no markdown):\n"
        '{"action":"capture"|"hold"|"move","rotate_deg":0,"tilt_deg":null,"reason":"..."}\n\n'
        "Axes:\n"
        "- rotate_deg: INCREMENTAL turntable spin (+CCW). Changes yaw only.\n"
        "- tilt_deg: ABSOLUTE bed angle (-30 to +30). Changes pitch — shifts board in frame.\n"
        "  THIS is how you get viewpoint diversity. Rotation alone often does NOT raise coverage.\n\n"
        "Priority order:\n"
        "1. CAPTURE if board has ≥12 corners, incidence <38°, sharp.\n"
        "2. Bed tilt is changed AUTOMATICALLY after each capture — do not repeat tilt moves.\n"
        "3. Only rotate 10–15° if board is NOT detected or badly off-center.\n"
        "4. NEVER rotate when board is already good — use capture or hold.\n\n"
        "Goal: 15 CAPTURED frames with varied bed tilt and rotation — NOT 75% coverage.\n"
        "viewpoint_pct tracks real 3D diversity. coverage_pct is a 2D grid; ignore if stuck.\n"
        "If supervisor says rotate is failing, use tilt_deg on the next move.\n"
        "Never move and capture in the same response."
    )
    _MIN_INTERVAL_S = 3.0
    _IMG_W = 320
    _IMG_H = 240
    _JPEG_Q = 60
    _MAX_ROTATE_DEG = 25.0

    def __init__(self, api_key: str) -> None:
        if not _HAVE_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic>=0.40.0")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._last_call: float = 0.0
        self._api_in_flight = False
        self._lock = threading.Lock()
        self._pending: CalibrationCommand | None = None
        self.last_command: CalibrationCommand | None = None
        self.call_count = 0
        self.status_msg = "Claude: idle"

    @property
    def busy(self) -> bool:
        return self._api_in_flight

    def request_decision(
        self,
        bgr: np.ndarray,
        *,
        n_captured: int,
        min_frames: int,
        board_detected: bool,
        n_corners: int,
        coverage_pct: float,
        incidence_deg: float | None,
        tilt_deg: float,
        sharpness: float,
        turntable_moving: bool,
        viewpoint_pct: float,
        supervisor_context: str,
    ) -> bool:
        """Queue an async API call. Returns True if a new request was started."""
        if turntable_moving or self._api_in_flight:
            return False
        now = time.time()
        if now - self._last_call < self._MIN_INTERVAL_S:
            return False

        small = cv2.resize(bgr, (self._IMG_W, self._IMG_H), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self._JPEG_Q])
        if not ok:
            return False

        self._last_call = now
        self._api_in_flight = True
        self.status_msg = "Claude: thinking..."

        img_b64 = base64.b64encode(buf.tobytes()).decode()
        angle_str = f"{incidence_deg:.0f}°" if incidence_deg is not None else "unknown"
        user_text = (
            f"Captured {n_captured}/{min_frames}. "
            f"viewpoint_pct {viewpoint_pct:.0f}% (goal: vary tilt between captures). "
            f"grid_coverage {coverage_pct:.0f}% (2D only — do not chase). "
            f"Bed tilt {tilt_deg:+.0f}°. Sharpness {sharpness:.0f}. "
            f"{supervisor_context}"
            f"Board {'detected' if board_detected else 'NOT DETECTED'}"
            + (f", {n_corners} corners, incidence {angle_str}." if board_detected else ".")
        )

        threading.Thread(
            target=self._api_worker,
            args=(img_b64, user_text),
            daemon=True,
            name="claude-pid",
        ).start()
        return True

    def _api_worker(self, img_b64: str, user_text: str) -> None:
        try:
            resp = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=120,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": user_text},
                ]}],
            )
            text = resp.content[0].text.strip() if resp.content else ""
            cmd = _parse_calibration_command(text)
            if cmd is None:
                print(f"[claude] unparseable response: {text[:80]!r}", file=sys.stderr)
                cmd = CalibrationCommand("hold", reason="parse error")
        except Exception as exc:
            print(f"[claude] API error: {exc}", file=sys.stderr)
            cmd = CalibrationCommand("hold", reason="API error")

        cmd = self._sanitize(cmd)
        self.call_count += 1
        self.last_command = cmd
        print(f"[claude] #{self.call_count} {cmd.summary()}")

        with self._lock:
            self._pending = cmd
            self._api_in_flight = False
            self.status_msg = cmd.summary()

    def _sanitize(self, cmd: CalibrationCommand) -> CalibrationCommand:
        """Plant limits — clamp moves before they reach the actuator."""
        if cmd.action == "move":
            rot = max(-self._MAX_ROTATE_DEG, min(self._MAX_ROTATE_DEG, cmd.rotate_deg))
            tilt = cmd.tilt_deg
            if tilt is not None:
                tilt = max(TILT_MIN, min(TILT_MAX, tilt))
            # Tiny combined move → hold (saves wear + API churn)
            if abs(rot) < 3.0 and (tilt is None):
                return CalibrationCommand("hold", reason=cmd.reason or "move too small")
            cmd = CalibrationCommand("move", rotate_deg=rot, tilt_deg=tilt, reason=cmd.reason)
        elif cmd.action == "capture":
            cmd = CalibrationCommand("capture", reason=cmd.reason)
        else:
            cmd = CalibrationCommand("hold", reason=cmd.reason)
        return cmd

    def poll(self) -> CalibrationCommand | None:
        """Return the latest command if the API worker finished."""
        with self._lock:
            if self._pending is None:
                return None
            cmd = self._pending
            self._pending = None
            return cmd


# ---------------------------------------------------------------------------
# Board defaults  (11x9 purchased board, 6.89"x5.71" physical size)
# ---------------------------------------------------------------------------
# Derived from board dimensions:
#   width  6.89" = 175.1 mm -> (175.1 - 10 mm margin) / 11 cols = 15.0 mm/square
#   height 5.71" = 145.0 mm -> (145.0 - 10 mm margin) /  9 rows = 15.0 mm/square
#   marker size = 73 % of square = 11 mm  (calib.io standard for 15 mm squares)
DEFAULT_COLS = 11
DEFAULT_ROWS = 9
DEFAULT_SQUARE_MM = 15.0
DEFAULT_MARKER_MM = 11.0
# 11x9 board needs 49 ArUco IDs; DICT_5X5_100 (100 IDs) is sufficient.
DEFAULT_DICT = cv2.aruco.DICT_5X5_100
DEFAULT_MIN_FRAMES = 15   # minimum captured frames before calibration


# ---------------------------------------------------------------------------
# IMU helpers
# ---------------------------------------------------------------------------

def _accel_to_angles(ax: float, ay: float, az: float) -> tuple[float, float, float]:
    """Return (pitch_deg, roll_deg, elevation_deg) from raw accelerometer data.

    The D405 IMU coordinate convention (camera pointing away from you):
      +X = right,  +Y = down,  +Z = away from the scene (into the camera body)

    pitch  - rotation around X (nose up/down);  0 deg = level
    roll   - rotation around Z (camera tilting left/right);  0 deg = level
    elevation - angle the optical axis (-Z) makes with the horizontal plane
    """
    g = math.sqrt(ax * ax + ay * ay + az * az)
    if g < 0.5:          # sensor not ready / near-zero gravity (shouldn't happen)
        return 0.0, 0.0, 0.0

    # Pitch: rotation around X axis
    pitch = math.degrees(math.atan2(-az, math.sqrt(ax * ax + ay * ay)))
    # Roll: rotation around Z axis
    roll = math.degrees(math.atan2(ax, ay))
    # Elevation of the optical axis from the horizontal plane
    # Optical axis is -Z in camera coords; gravity component along Z tells tilt.
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, -az / g))))
    return pitch, roll, elevation


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def _try_reset_device() -> None:
    """Hardware-reset any connected RealSense to free it from a previous session.

    Safe to call even if the camera isn't locked; the device re-enumerates in
    ~2 seconds after the reset.
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if devices.size() == 0:
            print("[camera] No RealSense device found for reset")
            return
        dev = devices[0]
        name = dev.get_info(rs.camera_info.name)
        print(f"[camera] Hardware-resetting {name} to release any existing session...")
        dev.hardware_reset()
        time.sleep(2.5)          # wait for USB re-enumeration
        print("[camera] Device reset complete")
    except Exception as exc:
        print(f"[camera] Reset skipped ({exc})")


def _build_pipeline(
    width: int, height: int, fps: int
) -> tuple[rs.pipeline, rs.pipeline_profile]:
    """Start a depth+color pipeline and return (pipeline, profile).

    The D405 requires depth to be active for color frames to flow.
    Not every resolution/fps combo is supported; we try a ranked list and
    pick the first profile that actually delivers a live frame.
    If the camera is busy ('already streaming'), a hardware reset is attempted
    before trying again.
    """
    # Ranked profiles: requested first, then D405-friendly fallbacks.
    # 848x480 is the most reliable mode on the D405.
    candidates: list[tuple[int, int, int]] = list(dict.fromkeys([
        (width, height, fps),
        (848, 480, 30),
        (848, 480, 15),
        (640, 480, 30),
        (640, 480, 15),
        (480, 270, 30),
    ]))

    reset_done = False

    for w, h, f in candidates:
        pipeline = rs.pipeline()
        config = rs.config()
        # D405 requires depth alongside color; color-only silently drops frames.
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)

        print(f"[camera] Trying {w}x{h} @ {f} fps...")
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            err = str(exc).lower()
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            if not reset_done and ("already streaming" in err
                                   or "device is already" in err
                                   or "failed to set power state" in err):
                print(f"[camera] Camera busy - resetting device...")
                _try_reset_device()
                reset_done = True
                # retry the same profile after reset
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                try:
                    profile = pipeline.start(config)
                except RuntimeError:
                    try:
                        pipeline.stop()
                    except RuntimeError:
                        pass
                    continue
            else:
                print(f"[camera]   start failed: {exc}")
                continue

        # Confirm frames actually flow (some profiles start OK but deliver nothing)
        print(f"[camera]   pipeline started - waiting for first frame...")
        got_frame = False
        for _ in range(3):
            try:
                pipeline.wait_for_frames(timeout_ms=3000)
                got_frame = True
                break
            except RuntimeError:
                pass

        if got_frame:
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            usb = device.get_info(rs.camera_info.usb_type_descriptor)
            print(f"[camera] Connected: {name}  [{serial}]  USB {usb}  ({w}x{h} @ {f} fps)")
            break

        print(f"[camera]   no frames at {w}x{h} @ {f} fps - trying next profile")
        try:
            pipeline.stop()
        except RuntimeError:
            pass
    else:
        raise RuntimeError(
            "No working stream profile found for the D405. "
            "Unplug/replug the USB cable (use a USB 3 port), close "
            "RealSense Viewer, and retry."
        )

    # Drain warmup frames so auto-exposure settles before board detection.
    print("[camera] Warming up...")
    for i in range(10):
        try:
            pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            break
    print("[camera] Ready")

    return pipeline, profile


def _read_color_frame(
    pipeline: rs.pipeline,
) -> np.ndarray | None:
    """Return a BGR frame, or None on timeout."""
    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        print("[camera] wait_for_frames timed out")
        return None
    cf = frames.get_color_frame()
    if not cf:
        print("[camera] frameset arrived but get_color_frame() returned invalid")
        return None
    return np.asanyarray(cf.get_data()).copy()


# ---------------------------------------------------------------------------
# Post-warmup camera lock
# ---------------------------------------------------------------------------

def _find_color_sensor(device: rs.device) -> rs.sensor | None:
    """Return the color (RGB) sensor, or None if not found.

    The D405 does not expose a standard color_sensor via first_color_sensor().
    Iterate all sensors and identify the RGB one by its support for auto white
    balance — the depth/stereo sensor does not have that option.
    """
    # Preferred: SDK helper (works on most D4xx models)
    try:
        return device.first_color_sensor()
    except RuntimeError:
        pass

    # Fallback: iterate sensors and find the one with white-balance control
    for sensor in device.sensors:
        try:
            if sensor.supports(rs.option.enable_auto_white_balance):
                return sensor
        except Exception:
            continue

    # Last resort: any sensor that has auto-exposure (depth sensor also has
    # this, but it won't have white balance — so this path only triggers if
    # there's a single sensor that handles both, which some D405 firmware does)
    for sensor in device.sensors:
        try:
            if sensor.supports(rs.option.enable_auto_exposure):
                return sensor
        except Exception:
            continue

    return None


def _lock_camera_settings(profile: rs.pipeline_profile) -> None:
    """Lock exposure and white balance after auto-exposure has settled.

    Frame-to-frame brightness swings from the auto-exposure loop cause the
    ArUco adaptive threshold to oscillate, which is the primary reason the
    board "flickers" in and out of detection.  Calling this once after warmup
    fixes the brightness at whatever value the AE settled on.
    """
    try:
        device = profile.get_device()
        color_sensor = _find_color_sensor(device)

        if color_sensor is None:
            print("[camera] Color sensor not found — exposure lock skipped")
            return

        # Read the AE-settled exposure before disabling AE, then re-apply it
        # so the driver doesn't reset to a default when AE is turned off.
        settled_exposure: float | None = None
        if color_sensor.supports(rs.option.enable_auto_exposure):
            if color_sensor.supports(rs.option.exposure):
                settled_exposure = color_sensor.get_option(rs.option.exposure)
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if settled_exposure is not None and color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, settled_exposure)
                print(f"[camera] Exposure locked at {settled_exposure:.0f} µs")

        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            print("[camera] White balance locked")

        # Maximum sharpness sharpens the marker edges, which improves the
        # signal-to-noise ratio of the adaptive threshold.
        if color_sensor.supports(rs.option.sharpness):
            rng = color_sensor.get_option_range(rs.option.sharpness)
            color_sensor.set_option(rs.option.sharpness, rng.max)
            print(f"[camera] Sharpness → {rng.max:.0f} (max)")

    except Exception as exc:
        print(f"[camera] Could not lock camera settings ({exc}) — continuing anyway")


# ---------------------------------------------------------------------------
# ChArUco detector
# ---------------------------------------------------------------------------

def _make_board(
    cols: int, rows: int, square_mm: float, marker_mm: float, *, legacy: bool = False
) -> cv2.aruco.CharucoBoard:
    aruco_dict = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    # OpenCV CharucoBoard works in any consistent unit (we pass millimetres)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_mm,
        marker_mm,
        aruco_dict,
    )
    if legacy:
        # calib.io boards (and boards generated before OpenCV 4.6) use the old
        # corner-ordering convention for even-row-count boards.  11x9 has an odd
        # row count so this is normally not needed, but --legacy overrides.
        board.setLegacyPattern(True)
    return board


def _make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    det_params = cv2.aruco.DetectorParameters()

    # Wider adaptive threshold window range — the key knob for robustness across
    # distances and lighting.  Default (3–23 step 10) misses markers when the
    # projected size doesn't align with the default windows; sweeping 3–53 at
    # step 4 costs a few ms but catches nearly all cases.
    det_params.adaptiveThreshWinSizeMin = 3
    det_params.adaptiveThreshWinSizeMax = 53
    det_params.adaptiveThreshWinSizeStep = 4

    # More lenient bit-error correction (default 0.6 → 0.8 accepts one extra flipped bit)
    det_params.errorCorrectionRate = 0.8

    # Sub-pixel corner refinement: stabilises corner positions across frames so
    # repeated detections of the same stationary board yield the same corners.
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det_params.cornerRefinementWinSize = 5
    det_params.cornerRefinementMaxIterations = 30
    det_params.cornerRefinementMinAccuracy = 0.01

    # Slightly smaller minimum marker perimeter — handles board at various distances
    det_params.minMarkerPerimeterRate = 0.02   # default 0.03

    # More lenient polygon fit — helps when markers are perspective-distorted
    # (e.g. board viewed at a steep angle so markers appear as trapezoids)
    det_params.polygonalApproxAccuracyRate = 0.08   # default 0.03

    # Higher perspective-corrected cell resolution → better decoding for angled markers
    det_params.perspectiveRemovePixelPerCell = 8   # default 4

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True   # refine ChArUco corners using marker geometry

    return cv2.aruco.CharucoDetector(board, charuco_params, det_params)


# CLAHE instance — created once, applied every frame.  Redistributing local
# contrast dramatically improves ArUco detection under flat or uneven lighting
# without affecting geometry.
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _detect(
    detector: cv2.aruco.CharucoDetector, gray: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (charuco_corners, charuco_ids) or (None, None).

    Tries detection on both the raw frame and a CLAHE-enhanced copy; returns
    whichever yields more ChArUco corners.  This handles two distinct failure
    modes: (a) over-exposed / flat images where raw works, (b) under-exposed /
    low-contrast images where CLAHE rescues the adaptive threshold.
    """
    def _try(img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            corners, ids, _mc, _mi = detector.detectBoard(img)
        except cv2.error:
            return None, None
        if corners is None or ids is None or len(ids) < 4:
            return None, None
        return corners, ids

    corners, ids = _try(gray)
    n_raw = len(ids) if ids is not None else 0

    enhanced = _CLAHE.apply(gray)
    c2, i2 = _try(enhanced)
    n_enh = len(i2) if i2 is not None else 0

    return (c2, i2) if n_enh > n_raw else (corners, ids)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img: np.ndarray, text: str, y: int, *, color=(255, 255, 255), scale=0.55) -> None:
    cv2.putText(img, text, (10, y), _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), _FONT, scale, color, 1, cv2.LINE_AA)


def _draw_hud(
    vis: np.ndarray,
    *,
    n_captured: int,
    min_frames: int,
    n_corners: int,
    accel: tuple[float, float, float] | None,
    board_detected: bool,
    last_msg: str,
    auto_mode: bool = False,
    auto_countdown: float | None = None,   # seconds until next auto-capture (None = waiting for motion)
    auto_ready: bool = False,              # about to fire this frame
) -> None:
    h, w = vis.shape[:2]

    # Instruction bar at the bottom
    bar_y = h - 10
    cv2.rectangle(vis, (0, h - 30), (w, h), (30, 30, 30), -1)
    if auto_mode:
        _put(vis, "AUTO  |  c=calibrate  p=print board  q=quit", bar_y, scale=0.45)
    else:
        _put(vis, "SPACE=capture  c=calibrate  p=print board  q=quit", bar_y, scale=0.45)

    # Frame counter
    count_color = (0, 200, 0) if n_captured >= min_frames else (0, 180, 255)
    _put(vis, f"Captured: {n_captured}/{min_frames}", 25, color=count_color)

    # Detection status
    if board_detected:
        det_color = (0, 220, 80)
        det_text = f"Board detected  ({n_corners} corners)"
    else:
        det_color = (0, 80, 220)
        det_text = "Board NOT detected"
    _put(vis, det_text, 50, color=det_color)

    # IMU tilt
    if accel is not None:
        pitch, roll, elevation = _accel_to_angles(*accel)
        _put(vis, f"Camera elevation: {elevation:+.1f} deg  pitch: {pitch:+.1f} deg  roll: {roll:+.1f} deg", 75)
    else:
        _put(vis, "IMU: not available", 75, color=(120, 120, 120))

    # Auto-capture countdown bar
    if auto_mode and auto_countdown is not None:
        bar_w = w - 20
        if auto_ready:
            filled = bar_w
            bar_color = (0, 255, 80)
            label = "CAPTURING"
        else:
            filled = int(bar_w * (1.0 - auto_countdown))
            bar_color = (0, 180, 255)
            label = f"Auto in {auto_countdown:.1f}s"
        cv2.rectangle(vis, (10, 90), (10 + bar_w, 108), (60, 60, 60), -1)
        cv2.rectangle(vis, (10, 90), (10 + max(0, filled), 108), bar_color, -1)
        _put(vis, label, 106, scale=0.42)
    elif auto_mode and board_detected:
        _put(vis, "Move board to trigger auto-capture", 100, color=(0, 180, 255), scale=0.48)
    elif auto_mode:
        _put(vis, "Waiting for board ...", 100, color=(120, 120, 120), scale=0.48)

    # Status message (flash)
    if last_msg:
        y = 125 if auto_mode else 105
        _put(vis, last_msg, y, color=(0, 255, 255), scale=0.65)

    # Warmup overlay (first few frames)
    if hasattr(vis, '_warmup'):
        pass  # handled by caller overlay


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _run_calibration(
    board: cv2.aruco.CharucoBoard,
    captured: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    for corners, ids in captured:
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and img_pts is not None and len(obj_pts) >= 4:
            obj_points.append(obj_pts)
            img_points.append(img_pts)

    if len(obj_points) < 4:
        print("Not enough valid frames for calibration (need at least 4).")
        return None

    print(f"Calibrating from {len(obj_points)} valid frames ...")
    flags = cv2.CALIB_RATIONAL_MODEL  # 8-coefficient rational model; good for wide lenses
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags
    )
    print(f"  RMS reprojection error: {rms:.4f} px")
    if rms > 1.5:
        print("  WARNING: RMS > 1.5 px - try more frames or larger board movements.")

    result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "rms_px": float(rms),
        "image_size": list(image_size),
        "n_frames": len(obj_points),
        "flags": int(flags),
        "model": "rational",
        "units": "pixels (camera_matrix), millimetres (board)",
    }
    return result


def _save_calibration(
    result: dict[str, Any],
    board_cfg: dict[str, Any],
    out_dir: Path,
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"realsense_d405_{ts}.json"
    payload = {"board": board_cfg, "calibration": result}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Board image
# ---------------------------------------------------------------------------

def _save_board_image(board: cv2.aruco.CharucoBoard, out_dir: Path, dpi: int = 150) -> Path:
    """Render the ChArUco board to a PNG at the given DPI."""
    # Estimate pixel size: board physical size in mm x dpi / 25.4
    board_size = board.getChessboardSize()
    sq = board.getSquareLength()
    px_w = int(board_size[0] * sq / 25.4 * dpi)
    px_h = int(board_size[1] * sq / 25.4 * dpi)
    # Clamp to a sensible range
    px_w = max(400, min(px_w, 3000))
    px_h = max(300, min(px_h, 3000))
    img = board.generateImage((px_w, px_h), marginSize=20, borderBits=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "charuco_board.png"
    cv2.imwrite(str(path), img)
    print(f"Board image saved -> {path}  ({px_w}x{px_h} px at ~{dpi} dpi)")
    return path


# ---------------------------------------------------------------------------
# Diversity hint (encourages varied viewpoints)
# ---------------------------------------------------------------------------

class _DiversityTracker:
    """Tracks how varied the captured frames are (position + tilt)."""

    def __init__(self) -> None:
        self._positions: list[tuple[float, float]] = []  # board centroid in image
        self._elevations: list[float] = []

    def add(
        self,
        corners: np.ndarray,
        accel: tuple[float, float, float] | None,
    ) -> None:
        centroid = corners.mean(axis=0).flatten()
        self._positions.append((float(centroid[0]), float(centroid[1])))
        if accel is not None:
            _, _, elev = _accel_to_angles(*accel)
            self._elevations.append(elev)

    def hint(self, image_w: int) -> str:
        n = len(self._positions)
        if n < 2:
            return "Move the board to a new position before capturing"
        xs = [p[0] for p in self._positions]
        span_x = max(xs) - min(xs)
        span_pct = span_x / image_w * 100
        elev_span = (
            max(self._elevations) - min(self._elevations) if self._elevations else 0.0
        )
        parts = []
        if span_pct < 30:
            parts.append("slide board left/right more")
        if elev_span < 15 and self._elevations:
            parts.append("tilt camera or board at different angles")
        if not parts:
            return "Good diversity! Keep going."
        return "Hint: " + "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Coverage grid – tracks which image regions have been hit by board corners.
# Used to score diversity and decide whether a new capture adds value.
# ---------------------------------------------------------------------------

class _CoverageGrid:
    """4×4 grid that records how many ChArUco corners landed in each cell."""

    ROWS = 4
    COLS = 4

    def __init__(self, w: int, h: int) -> None:
        self._w = w
        self._h = h
        self._counts = np.zeros((self.ROWS, self.COLS), dtype=np.int32)

    def _cell(self, px: float, py: float) -> tuple[int, int]:
        c = min(int(px / self._w * self.COLS), self.COLS - 1)
        r = min(int(py / self._h * self.ROWS), self.ROWS - 1)
        return r, c

    def add(self, corners: np.ndarray) -> None:
        """Record all corners from one capture."""
        for pt in corners.reshape(-1, 2):
            r, c = self._cell(float(pt[0]), float(pt[1]))
            self._counts[r, c] += 1

    def new_cells(self, corners: np.ndarray) -> int:
        """How many currently-empty cells would this frame cover?"""
        seen: set[tuple[int, int]] = set()
        for pt in corners.reshape(-1, 2):
            r, c = self._cell(float(pt[0]), float(pt[1]))
            if self._counts[r, c] == 0:
                seen.add((r, c))
        return len(seen)

    @property
    def coverage_pct(self) -> float:
        return float(np.count_nonzero(self._counts)) / (self.ROWS * self.COLS) * 100.0

    def draw(self, vis: np.ndarray, x0: int, y0: int, cell_px: int = 18) -> None:
        """Render the grid as a small overlay at (x0, y0)."""
        label = f"Grid {self.coverage_pct:.0f}%"
        cv2.putText(vis, label, (x0, y0 - 4), _FONT, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (x0, y0 - 4), _FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x1 = x0 + c * cell_px
                y1 = y0 + r * cell_px
                x2 = x1 + cell_px - 2
                y2 = y1 + cell_px - 2
                count = int(self._counts[r, c])
                if count > 0:
                    intensity = min(255, 80 + count * 25)
                    fill_color = (0, intensity, 0)
                else:
                    fill_color = (35, 35, 35)
                cv2.rectangle(vis, (x1, y1), (x2, y2), fill_color, -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (90, 90, 90), 1)


def _sharpness(gray: np.ndarray) -> float:
    """Return Laplacian variance as a focus measure (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Turntable actuator — PID plant; only moves when Claude commands it
# ---------------------------------------------------------------------------

class _TurntableActuator:
    """Executes CalibrationCommand moves. No scripted plan."""

    _ROTATE_WAIT_S = 2.5
    _TILT_WAIT_S = 4.0
    _SETTLE_AFTER_MOVE_S = 2.0

    def __init__(self, tt: "RevopointTurntable") -> None:
        self._tt = tt
        self.state: str = "homing"   # homing | moving | idle
        self.status_msg: str = "Turntable: homing..."
        self.tilt_deg: float = 0.0

    @property
    def is_moving(self) -> bool:
        return self.state in ("homing", "moving")

    def disconnect(self) -> None:
        try:
            self._tt.disconnect()
        except Exception:
            pass

    def home_async(self) -> None:
        threading.Thread(target=self._do_home, daemon=True, name="tt-home").start()

    def _do_home(self) -> None:
        try:
            self.state = "homing"
            self.status_msg = "Turntable: homing tilt..."
            self._tt.configure_speeds()
            self._tt.tilt_to_zero(wait_s=self._TILT_WAIT_S)
            self.tilt_deg = 0.0
            self.state = "idle"
            self.status_msg = "Turntable: ready — waiting for Claude"
        except Exception as exc:
            self.state = "idle"
            self.status_msg = f"Turntable home error: {exc}"

    def execute(self, cmd: CalibrationCommand) -> None:
        """Run a move command in the background."""
        if cmd.action != "move":
            return
        threading.Thread(
            target=self._do_move,
            args=(cmd,),
            daemon=True,
            name="tt-move",
        ).start()

    def settle_seconds(self, cmd: CalibrationCommand) -> float:
        """How long to wait after a move before observing again."""
        s = self._SETTLE_AFTER_MOVE_S
        if cmd.tilt_deg is not None and abs(cmd.tilt_deg - self.tilt_deg) > 0.5:
            s = max(s, self._TILT_WAIT_S)
        if abs(cmd.rotate_deg) > 0.5:
            s = max(s, self._ROTATE_WAIT_S)
        return s

    def _do_move(self, cmd: CalibrationCommand) -> None:
        self.state = "moving"
        try:
            if cmd.tilt_deg is not None and abs(cmd.tilt_deg - self.tilt_deg) > 0.5:
                prev = self.tilt_deg
                self.status_msg = f"Turntable: bed tilt {prev:+.0f}° → {cmd.tilt_deg:+.0f}°"
                print(f"[turntable] bed tilt {prev:+.1f}° → {cmd.tilt_deg:+.1f}°")
                self._tt.set_tilt(cmd.tilt_deg, wait_s=self._TILT_WAIT_S)
                self.tilt_deg = cmd.tilt_deg
            if abs(cmd.rotate_deg) >= 0.5:
                self.status_msg = f"Turntable: rotate {cmd.rotate_deg:+.0f}°"
                print(f"[turntable] rotate {cmd.rotate_deg:+.1f}°")
                self._tt.rotate_step(cmd.rotate_deg, wait_s=self._ROTATE_WAIT_S)
            self.state = "idle"
            self.status_msg = "Turntable: settled — waiting for Claude"
        except Exception as exc:
            self.state = "idle"
            self.status_msg = f"Turntable move error: {exc}"
            print(f"[turntable] ERROR: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Angle guidance panel
# ---------------------------------------------------------------------------

def _draw_angle_guide(
    vis: np.ndarray,
    incidence_deg: float | None,
    tilt_deg: float | None,
    n_captured: int,
) -> None:
    """Lower-right panel showing camera-to-board incidence angle with guidance.

    incidence_deg = 0   → camera perfectly perpendicular to board (ideal)
    incidence_deg = 90  → camera looking along the board face (invisible)
    target: ≤ 35°
    """
    h, w = vis.shape[:2]
    panel_w, panel_h = 208, 90
    px = w - panel_w - 6
    py = h - panel_h - 36   # sits just above the bottom status bars

    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, vis, 0.32, 0, vis)
    cv2.rectangle(vis, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    def _t(text: str, x: int, y: int, color=(170, 170, 170), scale=0.37) -> None:
        cv2.putText(vis, text, (x, y), _FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)

    lx = px + 7
    ly = py + 15
    _t("Camera angle guide", lx, ly, color=(200, 200, 200), scale=0.40)
    ly += 17

    if incidence_deg is None:
        _t("Aim board toward camera to", lx, ly, color=(0, 175, 255))
        ly += 15
        _t("measure incidence angle.", lx, ly, color=(0, 175, 255))
        ly += 15
        tilt_str = f"  tilt={tilt_deg:+.0f}deg" if tilt_deg is not None else ""
        _t(f"Target: cam ~75deg down{tilt_str}", lx, ly, color=(110, 110, 110), scale=0.34)
        return

    # Quality rating
    if incidence_deg <= 30:
        qcolor = (0, 215, 60)
        quality = "GOOD"
        # Rough camera-elevation estimate.
        # At tilt=0 (flat board): incidence = 90 - cam_elev → cam_elev = 90 - incidence.
        # Turntable sign: negative tilt means board tilted toward camera, which
        # reduces incidence for the same cam_elev, so subtract tilt to compensate.
        t = tilt_deg if tilt_deg is not None else 0.0
        cam_elev = max(0.0, min(90.0, 90.0 - incidence_deg - t))
        action = f"cam ~{cam_elev:.0f}deg from horizontal"
    elif incidence_deg <= 45:
        qcolor = (20, 195, 255)
        quality = "MARGINAL"
        action = "Tilt camera a bit more downward"
    elif incidence_deg <= 60:
        qcolor = (20, 100, 255)
        quality = "TOO STEEP"
        action = "Tilt camera more downward"
    else:
        qcolor = (0, 55, 215)
        quality = "CAMERA TOO SHALLOW"
        action = "Point camera much more downward"

    _t(f"Incidence: {incidence_deg:.1f}deg", lx, ly, color=(210, 210, 210), scale=0.42)
    ly += 16

    # Horizontal bar: 0° (left, ideal) → 90° (right, parallel)
    bar_x, bar_y = lx, ly
    bar_w, bar_h = panel_w - 14, 8
    pct = min(incidence_deg / 90.0, 1.0)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), qcolor, -1)
    # Green tick at the 35° target
    tx = bar_x + int(bar_w * 35.0 / 90.0)
    cv2.line(vis, (tx, bar_y - 3), (tx, bar_y + bar_h + 3), (0, 215, 60), 2)
    _t("35", tx - 7, bar_y - 5, color=(0, 180, 60), scale=0.28)
    ly += bar_h + 12

    _t(quality, lx, ly, color=qcolor, scale=0.44)
    ly += 16
    action_color = (120, 195, 120) if incidence_deg <= 30 else (170, 170, 170)
    _t(action, lx, ly, color=action_color, scale=0.34)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> int:
    board = _make_board(args.cols, args.rows, args.square_mm, args.marker_mm, legacy=args.legacy)
    detector = _make_detector(board)
    out_dir = Path(args.output)

    board_cfg = {
        "cols": args.cols,
        "rows": args.rows,
        "square_mm": args.square_mm,
        "marker_mm": args.marker_mm,
        "dictionary": "DICT_5X5_100",
        "legacy_pattern": args.legacy,
    }

    WIN = "ChArUco calibration - RealSense D405"

    def _status_frame(msg: str, color: tuple = (200, 200, 200)) -> np.ndarray:
        """Black frame with a centered status message - shown before camera is live."""
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(f, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(f, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        return f

    # Open the window immediately - before the camera even starts
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WIN, _status_frame("Saving board image..."))
    cv2.waitKey(1)

    _save_board_image(board, out_dir)

    cv2.imshow(WIN, _status_frame("Connecting to RealSense D405..."))
    cv2.waitKey(1)

    try:
        pipeline, profile = _build_pipeline(
            args.width, args.height, args.fps
        )
    except Exception as exc:
        err = str(exc)
        print(f"[camera] ERROR: {err}", file=sys.stderr)
        cv2.imshow(WIN, _status_frame(f"Camera error: {err[:70]}", color=(0, 60, 220)))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return 1

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    image_size = (intr.width, intr.height)

    K_factory = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ])
    print(f"Connected - {image_size[0]}x{image_size[1]}  "
          f"fx={intr.fx:.0f} fy={intr.fy:.0f}  "
          f"cx={intr.ppx:.0f} cy={intr.ppy:.0f}")
    print(f"Board: {args.cols}x{args.rows} @ {args.square_mm}mm  |  "
          f"auto={'ON' if args.auto else 'OFF'}  |  "
          f"SPACE=distance shot  q=quit\n")

    captured: list[tuple[np.ndarray, np.ndarray]] = []
    diversity = _DiversityTracker()
    coverage = _CoverageGrid(image_size[0], image_size[1])
    viewpoints = _ViewpointTracker()
    last_msg = ""
    last_msg_time = 0.0
    last_accel: tuple[float, float, float] | None = None  # D405 has no IMU; stays None

    # Minimum Laplacian variance to accept a capture (rejects motion blur).
    # Typical values: sharp=300-2000, soft-blur=60-150, heavy-blur<40.
    # 60 is intentionally lenient — the exposure lock means blurry frames come
    # only from actual motion, not brightness swings.
    MIN_SHARPNESS = 60.0
    # Minimum new coverage cells a frame must add when turntable is driving.
    # Set to 0 to capture every position regardless of redundancy.
    MIN_NEW_CELLS = 1

    # Auto-capture state
    auto_mode: bool = args.auto
    auto_delay: float = args.auto_delay
    _auto_last_centroid: np.ndarray | None = None
    _auto_trigger_start: float | None = None

    # Auto-calibration: fires when min_frames reached
    _autocal_start: float | None = None  # time when min_frames was first hit
    AUTOCAL_DELAY = 4.0                  # seconds before auto-calibrating

    # Warmup counter: skip board detection for the first N frames so
    # auto-exposure has fully settled before we lock it and start detecting.
    # 20 frames @ 30 fps = 0.67 s; @ 15 fps = 1.3 s — enough for AE to settle.
    WARMUP_FRAMES = 20
    _warmup_count = 0
    _settings_locked = False
    _last_incidence: float | None = None   # incidence angle from last successful solvePnP
    _last_incidence_time: float = 0.0      # wall-clock time of that measurement

    # ---- Claude PID brain ----
    controller: ClaudeCalibrationController | None = None
    _api_key = getattr(args, "api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
    if _api_key:
        if not _HAVE_ANTHROPIC:
            print("[claude] anthropic package not installed — AI guidance disabled. "
                  "Run: pip install anthropic>=0.40.0", file=sys.stderr)
        else:
            try:
                controller = ClaudeCalibrationController(_api_key)
                print("[claude] PID control enabled (Haiku, ~$0.0003/call)")
            except Exception as _adv_exc:
                print(f"[claude] Failed to init controller: {_adv_exc}", file=sys.stderr)

    # ---- Turntable actuator ----
    actuator: _TurntableActuator | None = None
    if _HAVE_TURNTABLE and not getattr(args, "no_turntable", False):
        ble_addr = getattr(args, "ble_address", None)
        try:
            cv2.imshow(WIN, _status_frame("Connecting to turntable via BLE..."))
            cv2.waitKey(1)
            _tt = RevopointTurntable()
            _tt.connect(ble_addr)
            actuator = _TurntableActuator(_tt)
            actuator.home_async()
            print("[turntable] Connected — homing, then Claude drives moves")
        except Exception as _tt_exc:
            print(f"[turntable] Not available ({_tt_exc}) - manual mode")
            actuator = None

    # PID loop state (Claude + turntable)
    pid_mode = controller is not None and actuator is not None
    supervisor = _PidSupervisor() if pid_mode else None
    _pid_settle_until: float = time.time() + 5.0   # after homing
    _pid_waiting_claude = False

    print(f"\nCamera live - auto-capture {'ON' if auto_mode else 'OFF'}"
          f"{' + Claude PID' if pid_mode else ''}"
          f"{' + turntable' if actuator else ''}.")
    print(f"SPACE = distance shot  |  q = quit\n")

    def _centroid(c: np.ndarray) -> np.ndarray:
        return c.mean(axis=0).flatten()

    def _do_capture(c: np.ndarray, i: np.ndarray, sharp: float = 0.0) -> None:
        nonlocal last_msg, last_msg_time, _auto_last_centroid, _auto_trigger_start
        nonlocal _pid_settle_until, _pid_waiting_claude, _last_incidence
        captured.append((c.copy(), i.copy()))
        diversity.add(c, last_accel)
        coverage.add(c)
        viewpoints.add(_last_incidence, c)
        hint = diversity.hint(image_size[0])
        vp = viewpoints.score_pct(image_size[0])
        last_msg = (
            f"Captured #{len(captured)}  vp={vp:.0f}%  grid={coverage.coverage_pct:.0f}%  - {hint}"
        )
        last_msg_time = time.time()
        _auto_last_centroid = _centroid(c)
        _auto_trigger_start = None
        print(
            f"  [frame {len(captured)}] sharp={sharp:.0f}  "
            f"vp={vp:.0f}%  grid={coverage.coverage_pct:.0f}%  {hint}"
        )
        if supervisor is not None and actuator is not None:
            supervisor.on_capture(coverage.coverage_pct, actuator.tilt_deg)
            tilt_cmd = supervisor.tilt_after_capture(actuator.tilt_deg)
            if tilt_cmd is not None:
                tilt_before = actuator.tilt_deg
                print(f"[supervisor] {tilt_cmd.summary()}")
                actuator.execute(tilt_cmd)
                supervisor.on_move(tilt_cmd, tilt_before)
                _pid_settle_until = time.time() + actuator.settle_seconds(tilt_cmd) + 1.0
                _pid_waiting_claude = False
                return
        if pid_mode:
            _pid_settle_until = time.time() + 1.5
            _pid_waiting_claude = False

    try:
        while True:
            bgr = _read_color_frame(pipeline)
            if bgr is None:
                cv2.imshow(WIN, _status_frame("Waiting for camera frame...", color=(0, 180, 255)))
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue

            _warmup_count += 1
            warming_up = _warmup_count <= WARMUP_FRAMES

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Skip detection during warmup so auto-exposure can settle first
            if warming_up:
                corners, ids = None, None
                frame_sharp = 0.0
            else:
                # Lock exposure/WB/sharpness on the first post-warmup frame
                if not _settings_locked:
                    _lock_camera_settings(profile)
                    _settings_locked = True
                corners, ids = _detect(detector, gray)
                frame_sharp = _sharpness(gray)
            board_detected = corners is not None

            vis = bgr.copy()

            # Warmup overlay - shown for the first WARMUP_FRAMES frames
            if warming_up:
                overlay = vis.copy()
                cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
                pct = int(_warmup_count / WARMUP_FRAMES * 100)
                bar_w = vis.shape[1] - 40
                cv2.rectangle(vis, (20, vis.shape[0]//2 - 8), (20 + bar_w, vis.shape[0]//2 + 8), (60,60,60), -1)
                cv2.rectangle(vis, (20, vis.shape[0]//2 - 8), (20 + int(bar_w * pct / 100), vis.shape[0]//2 + 8), (0,180,255), -1)
                _put(vis, f"Auto-exposure settling ... {pct}%", vis.shape[0]//2 + 25, scale=0.55)
                cv2.imshow(WIN, vis)
                cv2.waitKey(1)
                continue

            n_corners_now = len(corners) if board_detected and corners is not None else 0

            if board_detected:
                assert corners is not None and ids is not None
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 220, 80))
                try:
                    obj_pts, img_pts = board.matchImagePoints(corners, ids)
                    if obj_pts is not None and len(obj_pts) >= 4:
                        dist_coeffs = np.array(intr.coeffs)
                        ok, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts, K_factory, dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE
                        )
                        if ok:
                            cv2.drawFrameAxes(vis, K_factory, dist_coeffs, rvec, tvec, args.square_mm * 2)
                            # Incidence angle: angle between camera optical axis (+Z)
                            # and board face normal (third column of rotation matrix).
                            # 0° = camera perpendicular to board (best); 90° = parallel.
                            R_mat, _ = cv2.Rodrigues(rvec)
                            board_normal_z = float(abs(R_mat[2, 2]))   # |n · Z_cam|
                            _last_incidence = math.degrees(math.acos(min(board_normal_z, 1.0)))
                            _last_incidence_time = time.time()
                except cv2.error:
                    pass

            # ------------------------------------------------------------------
            # PID control loop — Claude brain, turntable actuator
            # ------------------------------------------------------------------
            auto_countdown: float | None = None
            auto_ready = False
            tt_moving = actuator is not None and actuator.is_moving
            now = time.time()

            if pid_mode and controller is not None and actuator is not None:
                if tt_moving:
                    _pid_waiting_claude = False
                elif now < _pid_settle_until:
                    pass  # mechanical settle — do not query Claude yet
                else:
                    cmd = controller.poll()
                    if cmd is not None:
                        _pid_waiting_claude = False
                        board_ok = _board_ready_for_capture(
                            board_detected=board_detected,
                            n_corners=n_corners_now,
                            incidence_deg=_last_incidence,
                            sharpness=frame_sharp,
                            min_sharpness=MIN_SHARPNESS,
                        )
                        if supervisor is not None:
                            original = cmd.summary()
                            cmd = supervisor.override(
                                cmd,
                                board_ok=board_ok,
                                coverage_pct=coverage.coverage_pct,
                                tilt_deg=actuator.tilt_deg,
                            )
                            if cmd.summary() != original:
                                print(f"[supervisor] {cmd.summary()}")

                        last_msg = f"Claude: {cmd.summary()}"
                        last_msg_time = now
                        if cmd.action == "capture":
                            if board_detected and corners is not None and ids is not None:
                                if frame_sharp < MIN_SHARPNESS:
                                    last_msg = f"Too blurry (sharp={frame_sharp:.0f}) — hold"
                                else:
                                    auto_ready = True
                                    _do_capture(corners, ids, frame_sharp)
                            else:
                                last_msg = "Claude: capture — board not detected"
                        elif cmd.action == "move":
                            tilt_before = actuator.tilt_deg
                            actuator.execute(cmd)
                            if supervisor is not None:
                                supervisor.on_move(cmd, tilt_before)
                            _pid_settle_until = now + actuator.settle_seconds(cmd)
                        else:
                            _pid_settle_until = now + 2.0
                    elif not controller.busy and not _pid_waiting_claude:
                        sup_ctx = supervisor.context_line() if supervisor else ""
                        if controller.request_decision(
                            bgr,
                            n_captured=len(captured),
                            min_frames=args.min_frames,
                            board_detected=board_detected,
                            n_corners=n_corners_now,
                            coverage_pct=coverage.coverage_pct,
                            incidence_deg=_last_incidence,
                            tilt_deg=actuator.tilt_deg,
                            sharpness=frame_sharp,
                            turntable_moving=tt_moving,
                            viewpoint_pct=viewpoints.score_pct(image_size[0]),
                            supervisor_context=sup_ctx,
                        ):
                            _pid_waiting_claude = True

            # ------------------------------------------------------------------
            # Legacy auto-capture (no Claude PID — manual or OpenCV-only)
            # ------------------------------------------------------------------
            elif auto_mode and board_detected and corners is not None and ids is not None:
                cen = _centroid(corners)
                dist = (
                    float(np.linalg.norm(cen - _auto_last_centroid))
                    if _auto_last_centroid is not None else 999.0
                )
                timer_running = _auto_trigger_start is not None
                reset_threshold = args.auto_min_move * 0.5 if timer_running else args.auto_min_move
                moved_enough = dist >= reset_threshold or _auto_last_centroid is None

                if moved_enough:
                    if _auto_trigger_start is None:
                        _auto_trigger_start = now
                    elapsed = now - _auto_trigger_start
                    remaining = max(0.0, auto_delay - elapsed)
                    auto_countdown = remaining / auto_delay
                    if remaining <= 0.0:
                        if frame_sharp < MIN_SHARPNESS:
                            last_msg = f"Too blurry (sharp={frame_sharp:.0f}) - waiting"
                            last_msg_time = now
                            _auto_trigger_start = None
                        else:
                            auto_ready = True
                            _do_capture(corners, ids, frame_sharp)
                else:
                    _auto_trigger_start = None
            elif auto_mode and not board_detected:
                _auto_trigger_start = None

            # ------------------------------------------------------------------
            # Auto-calibration: fires AUTOCAL_DELAY seconds after min_frames hit
            # ------------------------------------------------------------------
            n_cap = len(captured)
            autocal_countdown: float | None = None

            if n_cap >= args.min_frames:
                if _autocal_start is None:
                    _autocal_start = now
                autocal_remaining = max(0.0, AUTOCAL_DELAY - (now - _autocal_start))
                autocal_countdown = autocal_remaining
                if autocal_remaining <= 0.0:
                    break   # exit loop -> calibrate
            else:
                _autocal_start = None   # reset if somehow frames drop (shouldn't happen)

            # Distance-shot nudge: shown when one frame away from target
            if n_cap == args.min_frames - 1 and not last_msg:
                last_msg = "Almost done! Lift board ~5cm closer + SPACE"
                last_msg_time = now

            # Expire flash message after 3 seconds
            if now - last_msg_time > 3.0:
                last_msg = ""

            _draw_hud(
                vis,
                n_captured=n_cap,
                min_frames=args.min_frames,
                n_corners=len(corners) if board_detected and corners is not None else 0,
                accel=last_accel,
                board_detected=board_detected,
                last_msg=last_msg,
                auto_mode=auto_mode,
                auto_countdown=auto_countdown,
                auto_ready=auto_ready,
            )

            # Coverage grid overlay — top-right corner
            grid_cell = 18
            grid_x = vis.shape[1] - _CoverageGrid.COLS * grid_cell - 10
            grid_y = 14
            coverage.draw(vis, grid_x, grid_y, cell_px=grid_cell)

            # Live sharpness readout (top-right, below grid)
            sharp_y = grid_y + _CoverageGrid.ROWS * grid_cell + 16
            sharp_color = (0, 220, 80) if frame_sharp >= MIN_SHARPNESS else (0, 80, 220)
            sharp_txt = f"Sharp {frame_sharp:.0f}"
            cv2.putText(vis, sharp_txt, (grid_x, sharp_y), _FONT, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, sharp_txt, (grid_x, sharp_y), _FONT, 0.38, sharp_color, 1, cv2.LINE_AA)
            vp_txt = f"Viewpoint {viewpoints.score_pct(image_size[0]):.0f}%"
            cv2.putText(vis, vp_txt, (grid_x, sharp_y + 16), _FONT, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, vp_txt, (grid_x, sharp_y + 16), _FONT, 0.38, (0, 220, 180), 1, cv2.LINE_AA)

            # Expire incidence reading after 1.5 s without a fresh detection
            if _last_incidence is not None and (now - _last_incidence_time) > 1.5:
                _last_incidence = None

            # Camera angle guidance panel (lower-right)
            tt_tilt = actuator.tilt_deg if actuator is not None else None
            _draw_angle_guide(vis, _last_incidence, tt_tilt, n_cap)

            # Auto-calibration countdown overlay (bottom-right)
            if autocal_countdown is not None:
                txt = f"Auto-calibrating in {autocal_countdown:.1f}s  |  SPACE=more shots  c=now"
                _put(vis, txt, vis.shape[0] - 40, color=(0, 255, 150), scale=0.48)

            # PID / turntable status (bottom of frame)
            if pid_mode and controller is not None:
                pid_color = (0, 200, 255) if controller.busy else (180, 255, 100)
                _put(vis, controller.status_msg, vis.shape[0] - 36, color=pid_color, scale=0.40)
            if actuator is not None:
                tt_color = (0, 200, 255) if actuator.is_moving else (140, 220, 140)
                _put(vis, actuator.status_msg, vis.shape[0] - 18, color=tt_color, scale=0.40)

            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF

            # SPACE - manual/distance shot (resets auto-cal countdown)
            if key == ord(" "):
                if not board_detected or corners is None or ids is None:
                    last_msg = "Board not detected - nothing captured"
                    last_msg_time = time.time()
                else:
                    _do_capture(corners, ids, frame_sharp)
                    _autocal_start = None  # reset countdown so user can add more shots

            # c - calibrate immediately
            elif key in (ord("c"), ord("C")):
                if len(captured) < args.min_frames:
                    last_msg = f"Need {args.min_frames - len(captured)} more frames"
                    last_msg_time = time.time()
                else:
                    break

            # p - show/save board image
            elif key in (ord("p"), ord("P")):
                path = _save_board_image(board, out_dir)
                last_msg = f"Board saved -> {path.name}"
                last_msg_time = time.time()

            # q / ESC - quit
            elif key in (ord("q"), ord("Q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if actuator is not None:
            actuator.disconnect()

    if not captured:
        print("No frames captured - exiting without calibrating.")
        return 0

    if len(captured) < args.min_frames:
        print(
            f"Only {len(captured)} frames captured (need {args.min_frames}). "
            "Calibrating anyway ..."
        )

    result = _run_calibration(board, captured, image_size)
    if result is None:
        return 1

    # Print comparison with factory intrinsics
    K_cal = np.array(result["camera_matrix"])
    print("\nComparison (calibrated vs factory):")
    print(f"  fx  {K_cal[0,0]:8.2f}  vs  {K_factory[0,0]:8.2f}  (delta {K_cal[0,0]-K_factory[0,0]:+.2f})")
    print(f"  fy  {K_cal[1,1]:8.2f}  vs  {K_factory[1,1]:8.2f}  (delta {K_cal[1,1]-K_factory[1,1]:+.2f})")
    print(f"  cx  {K_cal[0,2]:8.2f}  vs  {K_factory[0,2]:8.2f}  (delta {K_cal[0,2]-K_factory[0,2]:+.2f})")
    print(f"  cy  {K_cal[1,2]:8.2f}  vs  {K_factory[1,2]:8.2f}  (delta {K_cal[1,2]-K_factory[1,2]:+.2f})")
    dist = result["dist_coeffs"]
    print(f"  distortion coeffs: {[f'{v:.4f}' for v in dist]}")

    path = _save_calibration(result, board_cfg, out_dir)

    print("\nDone. Scanner capture auto-loads the newest calibration/*.json:")
    print("  python -m scanner capture --mode turntable --name my_object")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS, help=f"Squares in X (default {DEFAULT_COLS})")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Squares in Y (default {DEFAULT_ROWS})")
    ap.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM, dest="square_mm",
                    help=f"Physical square side in mm (default {DEFAULT_SQUARE_MM})")
    ap.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM, dest="marker_mm",
                    help=f"Physical ArUco marker side in mm (default {DEFAULT_MARKER_MM})")
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, dest="min_frames",
                    help=f"Min frames before calibration is allowed (default {DEFAULT_MIN_FRAMES})")
    ap.add_argument("--output", default="calibration",
                    help="Directory to save calibration JSON and board image (default: calibration/)")
    ap.add_argument("--width", type=int, default=640, help="Camera width (default 640)")
    ap.add_argument("--height", type=int, default=480, help="Camera height (default 480)")
    ap.add_argument("--fps", type=int, default=30, help="Camera FPS (default 30)")
    ap.add_argument(
        "--no-auto", action="store_false", dest="auto",
        help="Disable auto-capture (manual SPACE-only mode).",
    )
    ap.set_defaults(auto=True)
    ap.add_argument(
        "--auto-delay", type=float, default=2.0, dest="auto_delay",
        help="Seconds to wait after board motion before auto-capturing (default 2.0).",
    )
    ap.add_argument(
        "--auto-min-move", type=float, default=20.0, dest="auto_min_move",
        help=(
            "Minimum board centroid movement in pixels to trigger a new auto-capture "
            "(prevents duplicate shots, default 20)."
        ),
    )
    ap.add_argument(
        "--legacy", action="store_true", default=False,
        help=(
            "Use the pre-OpenCV-4.6 ChArUco corner ordering (setLegacyPattern). "
            "Required for boards generated by calib.io or older tools with an even "
            "row count.  Not needed for 11x9 (odd rows), but add it if the board "
            "is not detected correctly."
        ),
    )
    ap.add_argument(
        "--no-turntable", action="store_true", default=False, dest="no_turntable",
        help="Disable BLE turntable control (manual-only mode).",
    )
    ap.add_argument(
        "--ble-address", default=None, dest="ble_address",
        help=(
            "BLE address of the Revopoint turntable (e.g. XX:XX:XX:XX:XX:XX). "
            "Defaults to the address saved in scanner/config.py."
        ),
    )
    ap.add_argument(
        "--api-key", default=None, dest="api_key",
        help=(
            "Anthropic API key to enable Claude AI-guided capture decisions. "
            "Falls back to the ANTHROPIC_API_KEY environment variable."
        ),
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.marker_mm >= args.square_mm:
        print("Error: --marker-mm must be smaller than --square-mm", file=sys.stderr)
        return 1
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
