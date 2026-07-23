"""Revopoint dual-axis turntable BLE controller.

Protocol: serial-over-BLE via service 0xFFE0 / characteristic 0xFFE1.
Commands are ASCII, ALL CAPS, start with '+', end with ';'.

Confirmed commands (from reverse-engineering the Revopoint protocol):

  Tilt axis (absolute position, -30 to +30 degrees):
    +CR,TILTVALUE=X;    set absolute tilt angle
    +CR,TILTSPEED=X;    set tilt speed (6.62 = max, larger = slower)
    +CR,TOZERO;         return tilt to boot-time zero
    +CR,STOP;           stop tilt immediately
    +QR,TILTANGLE;      query current tilt angle
    +QR,TILTSPEED;      query current tilt speed

  Rotation axis (incremental steps):
    +CT,TURNANGLE=X;    rotate by X degrees (positive or negative)
    +CT,TURNSPEED=X;    set rotation speed (35.64 = max, larger = slower)
    +CT,TOZERO;         rotate back to boot-time zero
    +CT,STOP;           stop rotation immediately
    +QT,CHANGEANGLE;    query current rotation angle
    +QT,TURNSPEED;      query current rotation speed

References:
  https://github.com/opensourcemanufacturing/Revopoint-Dual-Turntable
  https://github.com/SphaeroX/Revopoint-Dual-Axis-Turntable
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Callable

from bleak import BleakClient, BleakScanner

from scanner.config import (
    TURNTABLE_BLE_ADDRESS,
    TURNTABLE_CHAR_UUID,
    TURNTABLE_HOME_WAIT_S,
    TURNTABLE_ROTATE_SPEED,
    TURNTABLE_ROTATE_WAIT_S,
    TURNTABLE_SERVICE_UUID,
    TURNTABLE_TILT_SPEED,
    TURNTABLE_TILT_WAIT_S,
)

log = logging.getLogger(__name__)

# Revopoint turntables advertise names containing one of these strings (lower-case)
_NAME_HINTS = ("revo_dual", "revo_tt", "revopoint", "turntable", "tt_")

TILT_MIN = -30.0
TILT_MAX = 30.0

_ANGLE_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_angle(resp: str | None) -> float | None:
    """Pull the numeric value out of a device reply like '+DATA=60.0;'."""
    if not resp:
        return None
    m = _ANGLE_RE.search(resp)
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Internal async implementation
# ---------------------------------------------------------------------------

class _AsyncTurntable:
    """Low-level async BLE driver.  Lives in a dedicated event loop thread."""

    def __init__(self, client: BleakClient) -> None:
        self._client = client
        self._responses: list[str] = []
        self._notify_callback: Callable[[str], None] | None = None

    @classmethod
    async def connect(cls, address: str | None, timeout: float) -> "_AsyncTurntable":
        if address is None:
            address = await _scan_for_turntable(timeout)
        print(f"Connecting to turntable {address} …")
        # timeout goes to the constructor in bleak 3.x
        client = BleakClient(address, timeout=timeout)
        await client.connect()
        tt = cls(client)
        try:
            await client.start_notify(TURNTABLE_CHAR_UUID, tt._on_notify)
        except Exception:
            # Some firmware variants don't support notify — graceful fallback
            log.debug("Notify not available; will poll read instead.")
        print("Turntable connected.")
        return tt

    def _on_notify(self, _char: object, data: bytearray) -> None:
        text = data.decode("utf-8", errors="replace").strip()
        if text:
            log.debug("TT ← %s", text)
            self._responses.append(text)
            if self._notify_callback:
                self._notify_callback(text)

    async def send(self, cmd: str) -> None:
        log.debug("TT → %s", cmd)
        await self._client.write_gatt_char(
            TURNTABLE_CHAR_UUID, cmd.encode("utf-8"), response=False
        )

    async def query(self, cmd: str, timeout: float = 2.0) -> str | None:
        self._responses.clear()
        await self.send(cmd)
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._responses:
                return self._responses[-1]
            await asyncio.sleep(0.05)
        # Fallback: try a direct read
        try:
            raw = await self._client.read_gatt_char(TURNTABLE_CHAR_UUID)
            return raw.decode("utf-8", errors="replace").strip() or None
        except Exception:
            return None

    async def disconnect(self) -> None:
        if self._client.is_connected:
            try:
                await self._client.stop_notify(TURNTABLE_CHAR_UUID)
            except Exception:
                pass
            await self._client.disconnect()
            print("Turntable disconnected.")


async def _scan_for_turntable(timeout: float) -> str:
    print(f"Scanning for Revopoint turntable ({timeout:.0f} s) …")
    devices = await BleakScanner.discover(timeout=timeout)
    for d in sorted(devices, key=lambda x: -(x.rssi or -999)):
        name = (d.name or "").lower()
        if any(hint in name for hint in _NAME_HINTS):
            print(f"  Found: {d.name!r}  [{d.address}]  RSSI {d.rssi}")
            return d.address
    raise RuntimeError(
        "No Revopoint turntable found.\n"
        "  • Make sure it is powered on.\n"
        "  • Disconnect it from Revo Scan or any other app first.\n"
        "  • Pass --ble-address XX:XX:XX:XX:XX:XX if auto-scan misses it."
    )


# ---------------------------------------------------------------------------
# Synchronous public API (wraps the async driver via a background loop)
# ---------------------------------------------------------------------------

class RevopointTurntable:
    """Synchronous controller for the Revopoint dual-axis BLE turntable.

    Uses a dedicated background asyncio event loop so the BLE connection stays
    alive across calls without requiring the caller to manage async/await.

    Usage::

        tt = RevopointTurntable()
        tt.connect()          # auto-scan, or pass address="AA:BB:CC:DD:EE:FF"
        tt.set_tilt_speed()
        tt.set_rotate_speed()
        tt.set_tilt(-30)      # move tilt axis, blocks until settled
        tt.rotate_step(15)    # incremental rotation, blocks until settled
        tt.home()             # return both axes to zero
        tt.disconnect()
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="turntable-ble"
        )
        self._thread.start()
        self._tt: _AsyncTurntable | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float = 30.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, address: str | None = None, *, scan_timeout: float = 10.0) -> None:
        """Connect to the turntable.

        Address resolution order:
          1. *address* argument (explicit override)
          2. ``TURNTABLE_BLE_ADDRESS`` from config (saved from last successful scan)
          3. BLE auto-scan (finds by name hint or service UUID)
        """
        resolved = address or TURNTABLE_BLE_ADDRESS or None
        self._tt = self._run(
            _AsyncTurntable.connect(resolved, scan_timeout), timeout=scan_timeout + 5
        )

    def disconnect(self) -> None:
        if self._tt:
            self._run(self._tt.disconnect())
            self._tt = None
        self._loop.call_soon_threadsafe(self._loop.stop)

    def is_connected(self) -> bool:
        return self._tt is not None and self._tt._client.is_connected

    # ------------------------------------------------------------------
    # Raw command
    # ------------------------------------------------------------------

    def send(self, cmd: str) -> None:
        """Send a raw protocol command, e.g. '+CT,TURNANGLE=15;'."""
        assert self._tt, "Not connected — call connect() first."
        self._run(self._tt.send(cmd))

    def query(self, cmd: str, timeout: float = 2.0) -> str | None:
        """Send a query command and return the device's response string."""
        assert self._tt, "Not connected — call connect() first."
        return self._run(self._tt.query(cmd, timeout), timeout=timeout + 2)

    # ------------------------------------------------------------------
    # Speed configuration
    # ------------------------------------------------------------------

    def set_tilt_speed(self, speed: float = TURNTABLE_TILT_SPEED) -> None:
        """Set tilt speed.  6.62 = fastest; larger values = slower."""
        self.send(f"+CR,TILTSPEED={speed:.3f};")
        time.sleep(0.15)

    def set_rotate_speed(self, speed: float = TURNTABLE_ROTATE_SPEED) -> None:
        """Set rotation speed.  35.64 = fastest; larger values = slower."""
        self.send(f"+CT,TURNSPEED={speed:.3f};")
        time.sleep(0.15)

    # ------------------------------------------------------------------
    # Tilt axis
    # ------------------------------------------------------------------

    def set_tilt(self, angle: float, *, wait_s: float = TURNTABLE_TILT_WAIT_S) -> None:
        """Move tilt axis to absolute *angle* (clamped to ±30°) and wait."""
        angle = max(TILT_MIN, min(TILT_MAX, float(angle)))
        self.send(f"+CR,TILTVALUE={angle:.4f};")
        time.sleep(wait_s)

    def tilt_to_zero(self, *, wait_s: float = TURNTABLE_TILT_WAIT_S) -> None:
        self.send("+CR,TOZERO;")
        time.sleep(wait_s)

    def stop_tilt(self) -> None:
        self.send("+CR,STOP;")

    def query_tilt_angle(self) -> str | None:
        return self.query("+QR,TILTANGLE;")

    # ------------------------------------------------------------------
    # Rotation axis
    # ------------------------------------------------------------------

    def rotate_step(
        self, degrees: float, *, wait_s: float = TURNTABLE_ROTATE_WAIT_S
    ) -> None:
        """Rotate by *degrees* (incremental, + or −) then wait."""
        self.send(f"+CT,TURNANGLE={degrees:.4f};")
        time.sleep(wait_s)

    def rotate_to_zero(self, *, wait_s: float = TURNTABLE_HOME_WAIT_S) -> None:
        self.send("+CT,TOZERO;")
        time.sleep(wait_s)

    def stop_rotation(self) -> None:
        self.send("+CT,STOP;")

    def query_rotation_angle(self) -> str | None:
        return self.query("+QT,CHANGEANGLE;")

    def query_rotation_angle_deg(self) -> float | None:
        return _parse_angle(self.query_rotation_angle())

    # ------------------------------------------------------------------
    # Motion settling (poll the live angle instead of guessing a wait)
    # ------------------------------------------------------------------

    def _settle(
        self,
        query_fn: Callable[[], str | None],
        *,
        target_deg: float | None,
        tol_deg: float,
        timeout_s: float,
        poll_s: float,
        settle_reads: int,
    ) -> float | None:
        """Block until the reported angle reaches *target_deg* or stops moving.

        The Revopoint turntable moves slowly (~10 deg/s) and reports its live
        angle, so a fixed sleep either wastes time or captures mid-motion. We
        poll until the target is hit, or (fallback) until the angle has moved
        and then held steady for *settle_reads* polls. Falls back to a short
        sleep only if the device never answers a query.
        """
        t0 = time.monotonic()
        start = _parse_angle(query_fn())
        last = start
        stable = 0
        moved = False
        any_reading = start is not None
        while time.monotonic() - t0 < timeout_s:
            a = _parse_angle(query_fn())
            if a is not None:
                any_reading = True
                if start is not None and abs(a - start) > tol_deg:
                    moved = True
                if target_deg is not None and abs(a - target_deg) <= tol_deg:
                    return a
                if last is not None and abs(a - last) <= 0.3:
                    stable += 1
                    if stable >= settle_reads and moved:
                        return a
                else:
                    stable = 0
                last = a
            time.sleep(poll_s)
        if not any_reading:
            time.sleep(min(3.0, timeout_s))
        return last

    def settle_rotation(
        self,
        *,
        target_deg: float | None = None,
        tol_deg: float = 1.0,
        timeout_s: float = 30.0,
        poll_s: float = 0.25,
        settle_reads: int = 2,
    ) -> float | None:
        """Wait until rotation reaches *target_deg* (cumulative) or stops."""
        return self._settle(
            self.query_rotation_angle,
            target_deg=target_deg, tol_deg=tol_deg, timeout_s=timeout_s,
            poll_s=poll_s, settle_reads=settle_reads,
        )

    def settle_tilt(
        self,
        *,
        target_deg: float | None = None,
        tol_deg: float = 0.7,
        timeout_s: float = 20.0,
        poll_s: float = 0.25,
        settle_reads: int = 2,
    ) -> float | None:
        """Wait until tilt reaches *target_deg* (absolute) or stops."""
        return self._settle(
            self.query_tilt_angle,
            target_deg=target_deg, tol_deg=tol_deg, timeout_s=timeout_s,
            poll_s=poll_s, settle_reads=settle_reads,
        )

    # ------------------------------------------------------------------
    # Combined helpers
    # ------------------------------------------------------------------

    def configure_speeds(
        self,
        *,
        tilt_speed: float = TURNTABLE_TILT_SPEED,
        rotate_speed: float = TURNTABLE_ROTATE_SPEED,
    ) -> None:
        self.set_tilt_speed(tilt_speed)
        self.set_rotate_speed(rotate_speed)

    def home(self) -> None:
        """Return both axes to their boot-time zero positions."""
        print("Homing turntable …")
        self.tilt_to_zero(wait_s=TURNTABLE_TILT_WAIT_S)
        self.rotate_to_zero(wait_s=TURNTABLE_HOME_WAIT_S)
        print("Turntable homed.")

    def emergency_stop(self) -> None:
        self.send("+CR,STOP;")
        self.send("+CT,STOP;")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RevopointTurntable":
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
