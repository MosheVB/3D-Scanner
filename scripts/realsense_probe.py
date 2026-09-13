#!/usr/bin/env python3
"""List RealSense USB devices without starting a stream (diagnostics)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    try:
        from scanner.realsense_camera import list_realsense_devices
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print("On macOS: pip install pyrealsense2-macosx", file=sys.stderr)
        return 1

    try:
        devices = list_realsense_devices()
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        if "macOS blocked USB" not in str(exc):
            print(
                "\nTry:\n"
                "  1. Quit RealSense Viewer and any camera app\n"
                "  2. Unplug D405 USB, wait 5 seconds, replug (USB 3 port)\n"
                "  3. Run this script again (on Mac: ./scripts/realsense_sudo.sh scripts/realsense_probe.py)\n",
                file=sys.stderr,
            )
        # pyrealsense2-macosx can segfault during interpreter shutdown after a failed open
        os._exit(1)

    if not devices:
        print("No RealSense devices found.")
        return 1

    print(f"Found {len(devices)} device(s):")
    for d in devices:
        print(f"  {d['name']}  serial={d['serial']}  usb={d['usb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
