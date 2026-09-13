#!/usr/bin/env python3
"""Interactive turntable test — no camera required.

Connects to the Revopoint BLE turntable and runs a menu-driven set of tests
so you can verify every command works before committing to a full scan.

Usage:
    python scripts/turntable_test.py
    python scripts/turntable_test.py --address XX:XX:XX:XX:XX:XX

Menu options
────────────
  1  Tilt sweep    — move through -30, -15, 0, +15, +30 ° with pauses
  2  Rotate 360°   — full single revolution in 24 × 15° steps
  3  Manual tilt   — type an angle, moves there
  4  Manual rotate — type a step size, rotates once
  5  Query angles  — ask the device for current tilt and rotation angle
  6  Home          — return both axes to zero
  7  Stop both     — emergency stop
  Q  Quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.turntable import RevopointTurntable
from scanner.config import (
    TURNTABLE_TILT_LEVELS,
    TURNTABLE_ROTATE_STEP_DEG,
    TURNTABLE_ROTATE_WAIT_S,
    TURNTABLE_TILT_WAIT_S,
)


def _header(text: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")


def test_tilt_sweep(tt: RevopointTurntable) -> None:
    _header("Tilt sweep")
    for angle in TURNTABLE_TILT_LEVELS:
        print(f"  → tilt {angle:+.0f}°  (waiting {TURNTABLE_TILT_WAIT_S} s) …")
        tt.set_tilt(angle)
    print("  → returning to 0°")
    tt.tilt_to_zero()
    print("  Done.")


def test_rotate_360(tt: RevopointTurntable) -> None:
    _header("Full 360° rotation")
    steps = round(360 / TURNTABLE_ROTATE_STEP_DEG)
    print(f"  {steps} steps × {TURNTABLE_ROTATE_STEP_DEG}° each, "
          f"{TURNTABLE_ROTATE_WAIT_S} s wait per step")
    for i in range(steps):
        angle = (i + 1) * TURNTABLE_ROTATE_STEP_DEG
        print(f"  → step {i + 1}/{steps}  cumulative {angle:.0f}°")
        tt.rotate_step(TURNTABLE_ROTATE_STEP_DEG)
    print("  Returning to zero …")
    tt.rotate_to_zero()
    print("  Done.")


def test_manual_tilt(tt: RevopointTurntable) -> None:
    _header("Manual tilt")
    raw = input("  Enter target tilt angle (−30 to +30): ").strip()
    try:
        angle = float(raw)
    except ValueError:
        print("  Invalid number — skipping.")
        return
    print(f"  → tilting to {angle:+.1f}° …")
    tt.set_tilt(angle)
    print("  Done.")


def test_manual_rotate(tt: RevopointTurntable) -> None:
    _header("Manual rotate step")
    raw = input("  Enter rotation step in degrees (negative = reverse): ").strip()
    try:
        deg = float(raw)
    except ValueError:
        print("  Invalid number — skipping.")
        return
    print(f"  → rotating {deg:+.1f}° …")
    tt.rotate_step(deg)
    print("  Done.")


def test_query(tt: RevopointTurntable) -> None:
    _header("Query current angles")
    tilt = tt.query_tilt_angle()
    rot = tt.query_rotation_angle()
    print(f"  Tilt    : {tilt or '(no response)'}")
    print(f"  Rotation: {rot or '(no response)'}")


def test_home(tt: RevopointTurntable) -> None:
    _header("Home both axes")
    tt.home()


def test_estop(tt: RevopointTurntable) -> None:
    _header("Emergency stop")
    tt.emergency_stop()
    print("  Both axes stopped.")


MENU = """
┌──────────────────────────────────────┐
│  Revopoint Turntable Test Menu       │
├──────────────────────────────────────┤
│  1  Tilt sweep  (-30 → 0 → +30)     │
│  2  Rotate full 360°                 │
│  3  Manual tilt (enter angle)        │
│  4  Manual rotate step               │
│  5  Query current angles             │
│  6  Home both axes                   │
│  7  Emergency stop                   │
│  Q  Quit                             │
└──────────────────────────────────────┘
"""

HANDLERS = {
    "1": test_tilt_sweep,
    "2": test_rotate_360,
    "3": test_manual_tilt,
    "4": test_manual_rotate,
    "5": test_query,
    "6": test_home,
    "7": test_estop,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--address", "-a", default=None, metavar="XX:XX:XX:XX:XX:XX",
        help="BLE address (uses saved config default if omitted)",
    )
    p.add_argument(
        "--scan-timeout", type=float, default=10.0, metavar="SEC",
        help="BLE scan timeout if no address is given (default 10 s)",
    )
    args = p.parse_args()

    tt = RevopointTurntable()
    try:
        tt.connect(args.address, scan_timeout=args.scan_timeout)
        tt.configure_speeds()
    except Exception as exc:
        print(f"Could not connect: {exc}", file=sys.stderr)
        sys.exit(1)

    print(MENU)
    try:
        while True:
            choice = input("Choice: ").strip().upper()
            if choice == "Q":
                break
            handler = HANDLERS.get(choice)
            if handler is None:
                print(f"  Unknown option '{choice}'")
                print(MENU)
                continue
            try:
                handler(tt)
            except Exception as exc:
                print(f"  Error: {exc}")
    finally:
        print("\nDisconnecting …")
        tt.disconnect()
        print("Bye.")


if __name__ == "__main__":
    main()
