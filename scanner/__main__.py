#!/usr/bin/env python3
"""CLI: python -m scanner capture|process ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanner.config import DEFAULT_CALIBRATION_DIR, DEFAULT_SCANS_DIR
from scanner.platform_io import configure_stdio_utf8
from scanner.api_client import DEFAULT_API_URL


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        prog="scanner",
        description="Portable 3D scanner — RealSense D405 capture and mesh export",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Live capture session (SPACE / P / Q)")
    cap.add_argument(
        "--mode",
        choices=("turntable", "handheld"),
        default="turntable",
        help="Turntable: rotate object between shots; handheld: move camera",
    )
    cap.add_argument("--name", default="scan", help="Session name suffix")
    cap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCANS_DIR,
        help="Directory for scan sessions",
    )
    cap.add_argument(
        "--process",
        action="store_true",
        help="Automatically reconstruct mesh when capture ends",
    )
    cap.add_argument(
        "--no-calibration",
        action="store_true",
        help="Use factory intrinsics instead of the newest calibration/*.json",
    )
    cap.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Folder with realsense_d405_*.json (default: calibration/)",
    )

    proc = sub.add_parser("process", help="Fuse frames and export OBJ mesh")
    proc.add_argument("session", type=Path, help="Path to scan session folder")
    proc.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Output mesh path (.obj or .stl)",
    )
    proc.add_argument(
        "--no-icp",
        action="store_true",
        help="Handheld without odometry: merge frames without ICP",
    )
    proc.add_argument(
        "--odometry",
        action="store_true",
        help="Handheld only: fuse with RGB-D odometry instead of ICP",
    )
    proc.add_argument(
        "--no-dot-track",
        action="store_true",
        help="Turntable: use BLE/manifest rotation instead of platter dot tracking",
    )
    proc.add_argument(
        "--refine-pivot",
        action="store_true",
        help="Turntable: search pivot X/Z via OBB face parallelism (slow)",
    )
    proc.add_argument(
        "--no-turntable-pose",
        action="store_true",
        help="Turntable: skip analytical poses (use --odometry or ICP instead)",
    )

    meas = sub.add_parser("measure", help="Print axis-aligned bbox dimensions (mm)")
    meas.add_argument("session", type=Path, help="Path to scan session folder")
    meas.add_argument(
        "--source",
        choices=("cloud", "mesh", "frames"),
        default="cloud",
        help="cloud/mesh: from scan_object.ply or scan_mesh.obj; frames: fuse raw frames",
    )
    meas.add_argument(
        "--reprocess",
        action="store_true",
        help="Run process before measuring (if outputs missing)",
    )
    meas.add_argument(
        "--no-icp",
        action="store_true",
        help="Skip ICP when fusing --source frames",
    )
    meas.add_argument(
        "--expected",
        metavar="L,W,T",
        default=None,
        help="Expected dimensions in mm (any order); compare sorted extents",
    )
    meas.add_argument(
        "--tolerance",
        type=float,
        default=15.0,
        metavar="PCT",
        help="Max percent error per axis for PASS (default 15)",
    )

    smoke = sub.add_parser(
        "smoke-capture",
        help="Save N frames without GUI (for tests / headless)",
    )
    smoke.add_argument("--frames", type=int, default=3, help="Number of frames to save")
    smoke.add_argument("--name", default="smoke", help="Session name suffix")
    smoke.add_argument(
        "--mode",
        choices=("turntable", "handheld"),
        default="turntable",
    )
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_SCANS_DIR)
    smoke.add_argument(
        "--no-calibration",
        action="store_true",
        help="Use factory intrinsics instead of the newest calibration/*.json",
    )
    smoke.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Folder with realsense_d405_*.json (default: calibration/)",
    )

    tt = sub.add_parser(
        "turntable-scan",
        help="Automated BLE turntable scan (tilt sweep × full rotation)",
    )
    tt.add_argument("--name", default="scan", help="Session name suffix")
    tt.add_argument("--output-dir", type=Path, default=DEFAULT_SCANS_DIR)
    tt.add_argument(
        "--ble-address",
        default=None,
        metavar="XX:XX:XX:XX:XX:XX",
        help="BLE address of the turntable (auto-scan if omitted)",
    )
    tt.add_argument(
        "--tilt-levels",
        default=None,
        metavar="ANGLES",
        help="Comma-separated tilt angles in degrees, e.g. -30,-15,0,15,30",
    )
    tt.add_argument(
        "--rotate-step",
        type=float,
        default=None,
        metavar="DEG",
        help="Rotation step between frames in degrees (default 15)",
    )
    tt.add_argument(
        "--rotate-wait",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait after each rotation step (default 2.0)",
    )
    tt.add_argument(
        "--tilt-wait",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to wait after each tilt change (default 4.0)",
    )
    tt.add_argument(
        "--process",
        action="store_true",
        help="Automatically reconstruct mesh when scan ends",
    )
    tt.add_argument(
        "--scan-timeout",
        type=float,
        default=10.0,
        metavar="SEC",
        help="BLE scan timeout when searching for turntable (default 10)",
    )
    tt.add_argument(
        "--no-calibration",
        action="store_true",
        help="Use factory intrinsics instead of the newest calibration/*.json",
    )
    tt.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Folder with realsense_d405_*.json (default: calibration/)",
    )
    tt.add_argument(
        "--direct",
        action="store_true",
        help="Run scan locally with RealSense (no web server)",
    )
    tt.add_argument(
        "--video",
        action="store_true",
        help="Continuous 360 deg movie capture (one spin per tilt) instead of step shots",
    )
    tt.add_argument(
        "--video-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Max seconds to record during each video rotation (default: auto from speed)",
    )
    tt.add_argument(
        "--video-speed",
        type=float,
        default=None,
        metavar="SPEED",
        help="Turntable TURNSPEED for video spin (larger=slower; default 280)",
    )
    tt.add_argument(
        "--video-fps",
        type=int,
        default=None,
        metavar="FPS",
        help="Aligned RGB-D fps during video spin (default 30; 0=auto-probe max)",
    )
    tt.add_argument(
        "--video-pose-step",
        type=float,
        default=None,
        metavar="DEG",
        help="Keep best frame per this many degrees (default: same as --rotate-step)",
    )
    tt.add_argument(
        "--no-auto-mask",
        action="store_true",
        help="Skip motion-preview mask (fuse full frame; not recommended)",
    )
    tt.add_argument(
        "--api-url",
        default="http://127.0.0.1:8765",
        help="Scanner web server base URL (default http://127.0.0.1:8765)",
    )

    mbuild = sub.add_parser(
        "mask-build",
        help="Build motion-union mask from saved session frames",
    )
    mbuild.add_argument("session", type=Path, help="Session folder")
    mbuild.add_argument("--max-frames", type=int, default=24)
    mbuild.add_argument("--stride", type=int, default=None)

    web = sub.add_parser(
        "web",
        help="Web server: live view + API (port 8765). Restart: scripts/start_web_server.ps1",
    )
    web.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 — all interfaces)",
    )
    web.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port (default 8765)",
    )

    agent = sub.add_parser(
        "agent",
        help="Haiku vision agent: camera API + desktop mouse (Windows)",
    )
    agent.add_argument(
        "--goal",
        choices=("setup", "d405_tune", "viewer_calibrate", "turntable_test"),
        default="d405_tune",
        help="Task preset",
    )
    agent.add_argument("--goal-text", default="", help="Override goal description")
    agent.add_argument("--max-steps", type=int, default=40)
    agent.add_argument("--api-url", default=DEFAULT_API_URL)
    agent.add_argument("--save-dir", type=Path, default=None)
    agent.add_argument(
        "--no-desktop",
        action="store_true",
        help="Camera API only (no desktop screenshot/mouse)",
    )
    agent.add_argument("--list-goals", action="store_true")

    autotune = sub.add_parser(
        "autotune",
        help="Autonomous D405 tuning — live camera, no mouse (runs until Ctrl+C)",
    )
    autotune.add_argument("--live-dir", type=Path, default=None)
    autotune.add_argument("--max-configs", type=int, default=24)
    autotune.add_argument("--gui", action="store_true", help="OpenCV preview window")
    autotune.add_argument("--duration", type=float, default=0.0)
    autotune.add_argument("--haiku-every", type=int, default=8)

    args = parser.parse_args(argv)

    if args.command == "capture":
        from scanner.capture import run_capture

        return run_capture(
            mode=args.mode,
            name=args.name,
            output_dir=args.output_dir,
            auto_process=args.process,
            use_calibration=not args.no_calibration,
            calibration_dir=args.calibration_dir,
        )
    if args.command == "process":
        from scanner.process import process_session

        process_session(
            args.session,
            output_mesh=args.mesh,
            use_odometry=args.odometry,
            use_icp=not args.no_icp,
            use_turntable_pose=not args.no_turntable_pose,
            refine_pivot_xz=args.refine_pivot,
            use_dot_track=not args.no_dot_track,
        )
        return 0
    if args.command == "measure":
        from scanner.measure import measure_session

        measure_session(
            args.session,
            source=args.source,
            reprocess=args.reprocess,
            use_icp=not args.no_icp,
            expected_mm=args.expected,
            tolerance_pct=args.tolerance,
        )
        return 0
    if args.command == "smoke-capture":
        from scanner.smoke_capture import run_smoke_capture

        return run_smoke_capture(
            frames=args.frames,
            name=args.name,
            mode=args.mode,
            output_dir=args.output_dir,
            use_calibration=not args.no_calibration,
            calibration_dir=args.calibration_dir,
        )
    if args.command == "turntable-scan":
        from scanner.config import (
            TURNTABLE_TILT_LEVELS,
            TURNTABLE_ROTATE_STEP_DEG,
            TURNTABLE_ROTATE_WAIT_S,
            TURNTABLE_TILT_WAIT_S,
            TURNTABLE_VIDEO_ROTATE_SPEED,
            TURNTABLE_VIDEO_CAPTURE_FPS,
        )
        from scanner.mask_rgb_video import estimate_rotation_record_s
        from scanner.turntable_capture import TurntableScanConfig, run_turntable_scan_direct

        tilt_levels: list[float] | None = None
        if args.tilt_levels:
            try:
                tilt_levels = [float(v) for v in args.tilt_levels.split(",")]
            except ValueError:
                print(
                    f"--tilt-levels must be comma-separated numbers, got: {args.tilt_levels!r}",
                    file=sys.stderr,
                )
                return 1

        cfg = TurntableScanConfig(
            tilt_levels=tilt_levels or list(TURNTABLE_TILT_LEVELS),
            rotate_step_deg=args.rotate_step or TURNTABLE_ROTATE_STEP_DEG,
            rotate_wait_s=args.rotate_wait or TURNTABLE_ROTATE_WAIT_S,
            tilt_wait_s=args.tilt_wait or TURNTABLE_TILT_WAIT_S,
            ble_address=args.ble_address,
            scan_timeout=args.scan_timeout,
            auto_mask=not args.no_auto_mask,
            capture_mode="video" if args.video else "step",
        )
        if args.video:
            cfg.auto_mask = False
            cfg.video_rotate_speed = (
                args.video_speed if args.video_speed is not None else TURNTABLE_VIDEO_ROTATE_SPEED
            )
            cfg.video_pose_step_deg = (
                args.video_pose_step
                if args.video_pose_step is not None
                else (args.rotate_step or TURNTABLE_ROTATE_STEP_DEG)
            )
            cfg.video_capture_timeout_s = (
                args.video_timeout
                if args.video_timeout is not None
                else estimate_rotation_record_s(
                    cfg.video_rotate_speed, rotation_deg=cfg.video_rotation_deg
                )
            )
            if args.video_fps is not None:
                cfg.video_capture_fps = int(args.video_fps)

        if args.direct:
            return run_turntable_scan_direct(
                cfg=cfg,
                name=args.name,
                output_dir=args.output_dir,
                auto_process=args.process,
                use_calibration=not args.no_calibration,
                calibration_dir=args.calibration_dir,
            )

        from scanner.api_client import run_turntable_scan_via_api

        return run_turntable_scan_via_api(
            name=args.name,
            output_dir=args.output_dir,
            auto_process=args.process,
            use_calibration=not args.no_calibration,
            calibration_dir=args.calibration_dir,
            tilt_levels=list(cfg.tilt_levels),
            rotate_step_deg=cfg.rotate_step_deg,
            rotate_wait_s=cfg.rotate_wait_s,
            tilt_wait_s=cfg.tilt_wait_s,
            ble_address=cfg.ble_address,
            scan_timeout=cfg.scan_timeout,
            base_url=args.api_url,
        )
    if args.command == "mask-build":
        from scanner.mask_from_session import build_mask_from_session

        return build_mask_from_session(
            args.session,
            max_frames=args.max_frames,
            stride=args.stride,
        )
    if args.command == "web":
        from scanner.web import run_web_server

        return run_web_server(host=args.host, port=args.port)
    if args.command == "agent":
        from scanner.remote_agent import main as agent_main

        if args.list_goals:
            return agent_main(["--list-goals"])
        agent_argv = [
            "--goal", args.goal,
            "--max-steps", str(args.max_steps),
            "--api-url", args.api_url,
        ]
        if args.goal_text:
            agent_argv.extend(["--goal-text", args.goal_text])
        if args.save_dir:
            agent_argv.extend(["--save-dir", str(args.save_dir)])
        if args.no_desktop:
            agent_argv.append("--no-desktop")
        return agent_main(agent_argv)
    if args.command == "autotune":
        from scanner.d405_autotune import main as autotune_main

        at_argv: list[str] = []
        if args.live_dir:
            at_argv.extend(["--live-dir", str(args.live_dir)])
        if args.max_configs != 24:
            at_argv.extend(["--max-configs", str(args.max_configs)])
        if args.gui:
            at_argv.append("--gui")
        if args.duration:
            at_argv.extend(["--duration", str(args.duration)])
        if args.haiku_every != 8:
            at_argv.extend(["--haiku-every", str(args.haiku_every)])
        return autotune_main(at_argv if at_argv else None)
    return 1


if __name__ == "__main__":
    sys.exit(main())
