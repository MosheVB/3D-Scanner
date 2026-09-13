#!/usr/bin/env python3
"""Live RealSense D405 RGB + depth preview. Press 'q' to quit."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scanner.realsense_camera import RealSenseD405


def _colorize(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((depth_mm.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(
        np.uint8
    )
    out = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    out[depth_mm == 0] = 0
    return out


def _run_preview() -> int:
    cam = RealSenseD405()
    try:
        cam.start()
    except Exception as exc:
        print(f"Could not open RealSense D405: {exc}", file=sys.stderr)
        print("Run: python scripts/realsense_probe.py", file=sys.stderr)
        cam.stop()
        return 1

    print(f"Connected: {cam.label}")
    print(f"Resolution: {cam.resolution[0]}×{cam.resolution[1]} @ {cam.fps} fps")
    print("Press 'q' in the preview window to quit.")

    try:
        while cam.is_running():
            pair = cam.read_frame(block=True)
            if pair is None:
                continue
            rgb, depth_mm = pair
            depth_vis = _colorize(depth_mm)
            stacked = np.hstack([rgb, depth_vis])
            cv2.imshow("RealSense D405 — RGB | depth (q=quit)", stacked)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    # pyrealsense2-macosx can segfault in the parent after a failed native open
    if platform.system() == "Darwin" and "--worker" not in sys.argv:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]]
        result = subprocess.run(cmd, check=False)
        code = result.returncode
        if code in (-11, 139):
            print(
                "RealSense preview crashed (USB/power). "
                "Quit RealSense Viewer, unplug/replug D405, run scripts/realsense_probe.py.",
                file=sys.stderr,
            )
            return 1
        return code

    code = _run_preview()
    if code != 0 and platform.system() == "Darwin":
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
