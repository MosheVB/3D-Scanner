"""Shared BLE turntable control for the web server (manual moves between scans)."""

from __future__ import annotations

import threading
from typing import Any

from scanner.config import (
    TURNTABLE_BLE_ADDRESS,
    TURNTABLE_HOME_WAIT_S,
    TURNTABLE_ROTATE_WAIT_S,
    TURNTABLE_TILT_WAIT_S,
)
from scanner.jobs import JobManager
from scanner.turntable import TILT_MAX, TILT_MIN, RevopointTurntable


class TurntableService:
    """Manual turntable control; blocked while a scan job owns the BLE link."""

    def __init__(self, jobs: JobManager) -> None:
        self._jobs = jobs
        self._lock = threading.Lock()
        self._tt: RevopointTurntable | None = None
        self._last_tilt: float = 0.0
        self._rotation_deg: float = 0.0

    def _scan_active(self) -> bool:
        return self._jobs.active_job() is not None

    def _busy(self) -> None:
        if self._scan_active():
            raise RuntimeError("Turntable is in use by an active scan job")

    def is_connected(self) -> bool:
        with self._lock:
            return self._tt is not None and self._tt.is_connected()

    def connect(
        self,
        address: str | None = None,
        *,
        scan_timeout: float = 10.0,
    ) -> None:
        self._busy()
        with self._lock:
            if self._tt is not None and self._tt.is_connected():
                return
            self._tt = RevopointTurntable()
            self._tt.connect(
                address or TURNTABLE_BLE_ADDRESS or None,
                scan_timeout=scan_timeout,
            )
            self._tt.configure_speeds()

    def disconnect(self) -> None:
        with self._lock:
            if self._tt is not None:
                self._tt.disconnect()
                self._tt = None

    def _require(self) -> RevopointTurntable:
        self._busy()
        with self._lock:
            if self._tt is None or not self._tt.is_connected():
                raise RuntimeError("Turntable not connected")
            return self._tt

    def rotate(self, degrees: float) -> None:
        tt = self._require()
        tt.rotate_step(float(degrees), wait_s=TURNTABLE_ROTATE_WAIT_S)
        self._rotation_deg = (self._rotation_deg + float(degrees)) % 360.0

    def tilt(self, *, angle: float | None = None, delta: float | None = None) -> None:
        tt = self._require()
        if angle is not None:
            target = float(angle)
        elif delta is not None:
            target = self._last_tilt + float(delta)
        else:
            raise ValueError("Provide angle (absolute) or delta (relative)")
        target = max(TILT_MIN, min(TILT_MAX, target))
        tt.set_tilt(target, wait_s=TURNTABLE_TILT_WAIT_S)
        self._last_tilt = target

    def home(self) -> None:
        tt = self._require()
        tt.home()
        self._last_tilt = 0.0
        self._rotation_deg = 0.0

    def emergency_stop(self) -> None:
        tt = self._require()
        tt.emergency_stop()

    def status_dict(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected(),
            "last_tilt_deg": self._last_tilt,
            "rotation_deg": self._rotation_deg,
            "scan_active": self._scan_active(),
        }
