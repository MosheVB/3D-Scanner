"""Non-interactive capture: save N synced frames then exit."""

from __future__ import annotations

import sys
from pathlib import Path

from scanner.config import DEFAULT_CALIBRATION_DIR, DEFAULT_SCANS_DIR
from scanner.intrinsics import resolve_intrinsics
from scanner.realsense_camera import RealSenseD405
from scanner.session import ScanMode, add_frame, create_session, session_root


def run_smoke_capture(
    *,
    frames: int = 3,
    name: str = "smoke",
    mode: ScanMode = "turntable",
    output_dir: Path = DEFAULT_SCANS_DIR,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
) -> int:
    if frames < 1:
        print("frames must be >= 1", file=sys.stderr)
        return 1

    cam = RealSenseD405()
    try:
        cam.start()
    except RuntimeError as exc:
        print(f"Could not open RealSense D405: {exc}", file=sys.stderr)
        return 1

    intrinsics, intrinsics_src = resolve_intrinsics(
        cam.intrinsics,
        cam.resolution,
        use_calibration=use_calibration,
        calibration_dir=calibration_dir or DEFAULT_CALIBRATION_DIR,
    )
    print(f"Camera: {cam.label}  {cam.resolution[0]}×{cam.resolution[1]} @ {cam.fps} fps")
    print(f"Intrinsics: {intrinsics_src}")

    session = create_session(output_dir, name, mode, cam.resolution, intrinsics)
    root = session_root(session)
    print(f"Session: {root}")
    print(f"Capturing {frames} frame(s) …")

    saved = 0
    try:
        while cam.is_running() and saved < frames:
            pair = cam.read_frame(block=True)
            if pair is None:
                continue
            rgb, depth_mm = pair
            add_frame(session, rgb, depth_mm)
            saved += 1
            print(f"  saved frame {saved}/{frames}")
    finally:
        cam.stop()

    print(f"Done. {saved} frame(s) in {root}")
    if saved >= 2:
        print(f"Process: python -m scanner process {root}")
        print(f"Measure: python -m scanner measure {root}")
    return 0 if saved == frames else 1
