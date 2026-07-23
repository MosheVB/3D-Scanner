"""Automated 3D scan with the Revopoint dual-axis BLE turntable.

CLI: ``python -m scanner turntable-scan`` (web API) or ``--direct`` (local camera).
The web server calls :func:`execute_turntable_scan` with its owned camera instance.

Fusion uses known turntable poses (tilt + rotation per frame), not ICP or ORB.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import open3d as o3d

from scanner.config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_SCANS_DIR,
    MASK_PREVIEW_ROTATE_STEP_DEG,
    MASK_PREVIEW_ROTATE_WAIT_S,
    MASK_PREVIEW_STEPS,
    MASK_PREVIEW_TILT_DEG,
    RGB_HEIGHT,
    RGB_WIDTH,
    TURNTABLE_BLE_ADDRESS,
    TURNTABLE_CAPTURE_FPS,
    TURNTABLE_VIDEO_CAPTURE_FPS,
    TURNTABLE_CAPTURE_HEIGHT,
    TURNTABLE_CAPTURE_WIDTH,
    TURNTABLE_HOME_WAIT_S,
    TURNTABLE_ROTATE_SPEED,
    TURNTABLE_ROTATE_STEP_DEG,
    TURNTABLE_VIDEO_ROTATE_SPEED,
    TURNTABLE_ROTATE_WAIT_S,
    TURNTABLE_TILT_LEVELS,
    TURNTABLE_TILT_WAIT_S,
)
from scanner.intrinsics import resolve_intrinsics
from scanner.mask_rgb_preview import RgbVideoMaskConfig, run_rgb_video_mask_preview
from scanner.mask_preview import MaskPreviewConfig, run_mask_preview
from scanner.mask_rgb_video import estimate_rotation_record_s
from scanner.mask_video_capture import VideoCaptureConfig, capture_rotation_video, probe_best_fps
from scanner.video_frame_select import select_best_frames_per_degree
from scanner.realsense_camera import RealSenseD405
from scanner.session import add_frame, create_session, session_root
from scanner.session_mask import (
    load_include_mask,
    load_session_mask,
    mask_depth_for_session,
)
from scanner.turntable import TILT_MAX, TILT_MIN, RevopointTurntable
from scanner.turntable_fusion import TurntableScanMeta, TurntableVoxelFusion, save_turntable_scan_meta
from scanner.turntable_pose import (
    analytical_geometry_from_frame0,
    object_center_from_seed,
    object_extent_from_seed,
    save_turntable_geometry,
)


class FrameSource(Protocol):
    def read_frame(
        self, *, block: bool = True, apply_filters: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None: ...


@dataclass
class TurntableScanConfig:
    """Parameters for an automated turntable scan pass."""

    tilt_levels: list[float] = field(default_factory=lambda: list(TURNTABLE_TILT_LEVELS))
    rotate_step_deg: float = TURNTABLE_ROTATE_STEP_DEG
    rotate_wait_s: float = TURNTABLE_ROTATE_WAIT_S
    tilt_wait_s: float = TURNTABLE_TILT_WAIT_S
    ble_address: str | None = field(default_factory=lambda: TURNTABLE_BLE_ADDRESS or None)
    scan_timeout: float = 10.0
    auto_mask: bool = True
    mask_preview_steps: int = MASK_PREVIEW_STEPS
    mask_preview_rotate_step_deg: float = MASK_PREVIEW_ROTATE_STEP_DEG
    mask_preview_rotate_wait_s: float = MASK_PREVIEW_ROTATE_WAIT_S
    mask_preview_tilt_deg: float = MASK_PREVIEW_TILT_DEG
    mask_use_rgb_video: bool = True
    live_voxel_fusion: bool = True
    # "step" = stop-and-shoot; "video" = one continuous 360 spin per tilt level
    capture_mode: str = "step"
    video_rotation_deg: float = 360.0
    video_capture_timeout_s: float = 60.0
    video_min_frames: int = 24
    video_max_frames: int = 4500
    video_rotate_speed: float = TURNTABLE_VIDEO_ROTATE_SPEED
    video_pose_step_deg: float = TURNTABLE_ROTATE_STEP_DEG
    video_capture_fps: int = TURNTABLE_VIDEO_CAPTURE_FPS  # 0 = auto-probe max aligned RGB-D

    @property
    def steps_per_revolution(self) -> int:
        return max(1, round(360.0 / self.rotate_step_deg))

    @property
    def total_frames(self) -> int:
        if self.capture_mode == "video":
            return len(self.tilt_levels)
        return len(self.tilt_levels) * self.steps_per_revolution


def _hud(
    frame: np.ndarray,
    lines: list[str],
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.6,
) -> np.ndarray:
    out = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return out


def _colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip(
        (depth_mm.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255
    ).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def _grab_frame(
    cam: FrameSource, *, timeout_s: float = 3.0
) -> tuple[np.ndarray, np.ndarray] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pair = cam.read_frame(block=False)
        if pair is not None:
            return pair
        time.sleep(0.02)
    return None


def _execute_turntable_video_scan(
    frame_source: FrameSource,
    *,
    realsense: RealSenseD405,
    cfg: TurntableScanConfig,
    session,
    root: Path,
    tt: RevopointTurntable,
    intrinsics: list[list[float]],
    on_progress: Callable[[int, int, str], None] | None,
    on_frame: Callable[[np.ndarray, np.ndarray], None] | None,
    cancel_event: threading.Event | None,
    show_preview: bool,
    log: Callable[[str], None],
) -> int:
    """Continuous rotation capture: poll RGB-D while turntable spins 360 deg."""
    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    win = "3D Scanner - turntable video"
    if show_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    steps_per_rev = cfg.video_min_frames
    effective_step_deg = 360.0 / max(1, steps_per_rev)

    try:
        for tilt_idx, tilt_angle in enumerate(cfg.tilt_levels):
            if _cancelled():
                break

            msg = (
                f"Video tilt {tilt_idx + 1}/{len(cfg.tilt_levels)} "
                f"-> {tilt_angle:+.1f} deg (spin {cfg.video_rotation_deg:.0f} deg)"
            )
            log(msg)
            if on_progress is not None:
                on_progress(tilt_idx, len(cfg.tilt_levels), msg)

            log("  Still reference frames (pre-spin bg)...")
            for ref_i in range(2):
                pair = None
                while pair is None:
                    if _cancelled():
                        break
                    pair = frame_source.read_frame(block=True, apply_filters=False)
                if pair is None:
                    break
                rgb_ref, depth_ref = pair
                cv2.imwrite(str(root / f"video_bg_ref_{ref_i}_rgb.png"), rgb_ref)
                cv2.imwrite(str(root / f"video_bg_ref_{ref_i}_depth.png"), depth_ref)
            time.sleep(0.2)

            vcfg = VideoCaptureConfig(
                rotation_deg=cfg.video_rotation_deg,
                capture_timeout_s=cfg.video_capture_timeout_s,
                min_frames=cfg.video_min_frames,
                max_frames=cfg.video_max_frames,
                rotate_speed=cfg.video_rotate_speed,
                tilt_deg=tilt_angle,
                tilt_wait_s=cfg.tilt_wait_s,
                home_after=False,
            )
            rgb_frames, depth_frames, fps, frame_times, spin_t0 = capture_rotation_video(
                frame_source, tt, vcfg, log=log
            )
            n_raw = len(rgb_frames)
            spin_t1 = frame_times[-1] if frame_times else spin_t0 + 1.0
            spin_duration = max(spin_t1 - spin_t0, 1e-3)

            picks = select_best_frames_per_degree(
                rgb_frames,
                depth_frames,
                frame_times,
                spin_t0=spin_t0,
                spin_duration=spin_duration,
                rotation_deg_total=cfg.video_rotation_deg,
                pose_step_deg=cfg.video_pose_step_deg,
            )
            n_bins = max(1, int(round(cfg.video_rotation_deg / cfg.video_pose_step_deg)))
            if len(picks) < cfg.video_min_frames // 2:
                log(
                    f"  Warning: only {len(picks)}/{n_bins} degree bins filled "
                    f"(step {cfg.video_pose_step_deg:.1f} deg)"
                )

            saved = 0
            for pick in picks:
                if show_preview and saved % max(1, len(picks) // 30) == 0:
                    depth_vis = _colorize_depth(pick.depth_mm)
                    depth_strip = cv2.resize(depth_vis, (RGB_WIDTH, 160))
                    rgb_panel = cv2.resize(pick.rgb, (RGB_WIDTH, RGB_HEIGHT - 160))
                    left = np.vstack([rgb_panel, depth_strip])
                    hud = _hud(
                        left,
                        [
                            f"Video {tilt_angle:+.0f} deg  saved {saved + 1}/{len(picks)}",
                            f"rot {pick.rotation_deg:.1f} deg  q={pick.quality:.0f}",
                            "Q=quit",
                        ],
                    )
                    cv2.imshow(win, hud)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        if key == 27:
                            tt.emergency_stop()
                        break

                add_frame(
                    session,
                    pick.rgb,
                    pick.depth_mm,
                    tilt_deg=tilt_angle,
                    rotation_deg=pick.rotation_deg,
                )
                if on_frame is not None:
                    on_frame(pick.rgb, pick.depth_mm)
                frame_idx += 1
                saved += 1

            n = saved
            steps_per_rev = n
            effective_step_deg = cfg.video_pose_step_deg

            log(
                f"  Saved {n}/{n_raw} video frames "
                f"(best per {cfg.video_pose_step_deg:.1f} deg, {n_bins} bins) @ {fps:.1f} fps "
                f"(speed {cfg.video_rotate_speed:.1f})"
            )

            if _cancelled():
                break

        scan_meta = TurntableScanMeta(
            tilt_levels=list(cfg.tilt_levels),
            rotate_step_deg=effective_step_deg,
            steps_per_revolution=steps_per_rev,
            capture_mode="video",
        )
        save_turntable_scan_meta(root / "turntable.json", scan_meta)

        if not _cancelled():
            log("Returning turntable to zero ...")
            tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
            tt.rotate_to_zero(wait_s=TURNTABLE_HOME_WAIT_S)
    finally:
        if show_preview:
            cv2.destroyAllWindows()

    return frame_idx


def execute_turntable_scan(
    frame_source: FrameSource,
    *,
    realsense: RealSenseD405,
    cfg: TurntableScanConfig | None = None,
    name: str = "scan",
    output_dir: Path = DEFAULT_SCANS_DIR,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_frame: Callable[[np.ndarray, np.ndarray], None] | None = None,
    cancel_event: threading.Event | None = None,
    show_preview: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path | None:
    """Run scan using an existing camera. Returns session path or None on failure."""
    if cfg is None:
        cfg = TurntableScanConfig()
    if log is None:
        log = print

    for t in cfg.tilt_levels:
        if not (TILT_MIN <= t <= TILT_MAX):
            raise ValueError(f"Tilt level {t}° is out of range [{TILT_MIN}, {TILT_MAX}].")

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _progress(current: int, total: int, message: str) -> None:
        if on_progress is not None:
            on_progress(current, total, message)

    tt = RevopointTurntable()
    try:
        tt.connect(cfg.ble_address, scan_timeout=cfg.scan_timeout)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to turntable: {exc}") from exc

    intrinsics, intrinsics_src = resolve_intrinsics(
        realsense.intrinsics,
        realsense.resolution,
        use_calibration=use_calibration,
        calibration_dir=calibration_dir or DEFAULT_CALIBRATION_DIR,
    )
    log(
        f"Camera: {realsense.label}  "
        f"{realsense.resolution[0]}x{realsense.resolution[1]} @ {realsense.fps} fps"
    )
    log(f"Intrinsics: {intrinsics_src}")

    session = create_session(output_dir, name, "turntable", realsense.resolution, intrinsics)
    root = session_root(session)
    log(f"Session: {root}")

    if cfg.capture_mode != "video":
        scan_meta = TurntableScanMeta(
            tilt_levels=list(cfg.tilt_levels),
            rotate_step_deg=cfg.rotate_step_deg,
            steps_per_revolution=cfg.steps_per_revolution,
            capture_mode="step",
        )
        save_turntable_scan_meta(root / "turntable.json", scan_meta)

    tt.configure_speeds()

    mask_spec = load_session_mask(root)
    if cfg.capture_mode != "video" and cfg.auto_mask and mask_spec is None:
        if cfg.mask_use_rgb_video:
            mask_spec = run_rgb_video_mask_preview(
                frame_source,
                tt,
                root,
                RgbVideoMaskConfig(tilt_wait_s=cfg.tilt_wait_s),
                log=log,
            )
        else:
            preview_cfg = MaskPreviewConfig(
                steps=cfg.mask_preview_steps,
                rotate_step_deg=cfg.mask_preview_rotate_step_deg,
                rotate_wait_s=cfg.mask_preview_rotate_wait_s,
                tilt_deg=cfg.mask_preview_tilt_deg,
                tilt_wait_s=cfg.tilt_wait_s,
                home_after=True,
            )
            mask_spec = run_mask_preview(
                frame_source, tt, root, preview_cfg, log=log
            )

    include_mask: np.ndarray | None = None
    if mask_spec is not None:
        include_mask = load_include_mask(root, mask_spec)

    fusion: TurntableVoxelFusion | None = None
    geom_ref_tilt = 0.0 if 0.0 in cfg.tilt_levels else cfg.tilt_levels[0]

    win = "3D Scanner - turntable scan"
    if show_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    aborted = False
    total = cfg.total_frames

    try:
        if cfg.capture_mode == "video":
            log(
                f"Video capture mode: {cfg.video_rotation_deg:.0f} deg continuous spin, "
                f"timeout {cfg.video_capture_timeout_s:.0f}s, "
                f"speed {cfg.video_rotate_speed:.0f}, "
                f"camera {realsense.fps} fps, "
                f"{len(cfg.tilt_levels)} tilt level(s)"
            )
            frame_idx = _execute_turntable_video_scan(
                frame_source,
                realsense=realsense,
                cfg=cfg,
                session=session,
                root=root,
                tt=tt,
                intrinsics=intrinsics,
                on_progress=_progress,
                on_frame=on_frame,
                cancel_event=cancel_event,
                show_preview=show_preview,
                log=log,
            )
        else:
            for tilt_idx, tilt_angle in enumerate(cfg.tilt_levels):
                if _cancelled():
                    aborted = True
                    break

                msg = f"Tilt {tilt_idx + 1}/{len(cfg.tilt_levels)} -> {tilt_angle:+.1f} deg"
                log(msg)
                _progress(frame_idx, total, msg)
                tt.set_tilt(tilt_angle, wait_s=cfg.tilt_wait_s)

                for rot_idx in range(cfg.steps_per_revolution):
                    if _cancelled():
                        aborted = True
                        break

                    rotation_angle = rot_idx * cfg.rotate_step_deg
                    status = (
                        f"Tilt {tilt_angle:+.0f} deg  "
                        f"Rot {rotation_angle:.0f} deg  "
                        f"({frame_idx}/{total})"
                    )
                    _progress(frame_idx, total, status)

                    if rot_idx > 0:
                        tt.rotate_step(cfg.rotate_step_deg, wait_s=cfg.rotate_wait_s)

                    pair = _grab_frame(frame_source)
                    if pair is None:
                        log("  WARNING: no camera frame - skipping position.")
                        continue
                    rgb, depth_mm = pair

                    if mask_spec is not None and include_mask is not None:
                        depth_mm = mask_depth_for_session(
                            root, depth_mm, spec=mask_spec, include=include_mask
                        )

                    if (
                        cfg.live_voxel_fusion
                        and include_mask is not None
                        and fusion is None
                        and abs(tilt_angle - geom_ref_tilt) < 0.1
                        and abs(rotation_angle) < 0.1
                    ):
                        try:
                            geom = analytical_geometry_from_frame0(
                                depth_mm,
                                include_mask,
                                intrinsics,
                                tilt_ref_deg=tilt_angle,
                                rot_ref_deg=rotation_angle,
                            )
                            extent = object_extent_from_seed(
                                depth_mm, include_mask, intrinsics
                            )
                            center = object_center_from_seed(
                                depth_mm, include_mask, intrinsics
                            )
                            save_turntable_geometry(root / "turntable_geometry.json", geom)
                            fusion = TurntableVoxelFusion(
                                geom, intrinsics, center_m=center, extent_m=extent
                            )
                            log(
                                "Analytical turntable geometry "
                                f"(pivot_m={[round(x, 3) for x in geom.pivot_m]})"
                            )
                        except ValueError as exc:
                            log(f"Geometry calibration skipped ({exc}).")

                    if fusion is not None and include_mask is not None:
                        fusion.integrate(
                            rgb,
                            depth_mm,
                            include_mask,
                            tilt_deg=tilt_angle,
                            rotation_deg=rotation_angle,
                        )

                    if show_preview:
                        depth_vis = _colorize_depth(depth_mm)
                        depth_strip = cv2.resize(depth_vis, (RGB_WIDTH, 160))
                        rgb_panel = cv2.resize(rgb, (RGB_WIDTH, RGB_HEIGHT - 160))
                        left = np.vstack([rgb_panel, depth_strip])
                        hud_lines = [
                            status,
                            f"Tilt levels: {cfg.tilt_levels}",
                            "S=skip  Q=quit  ESC=e-stop",
                        ]
                        if fusion is not None:
                            hud_lines.append(f"Voxels: {fusion.voxel_count}")
                        cv2.imshow(win, _hud(left, hud_lines))
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            tt.emergency_stop()
                            aborted = True
                            break
                        if key in (ord("q"), ord("Q")):
                            aborted = True
                            break
                        if key in (ord("s"), ord("S")):
                            continue

                    add_frame(
                        session,
                        rgb,
                        depth_mm,
                        tilt_deg=tilt_angle,
                        rotation_deg=rotation_angle,
                    )
                    if on_frame is not None:
                        on_frame(rgb, depth_mm)
                    frame_idx += 1
                    _progress(frame_idx, total, status)

                if aborted:
                    break

        if cfg.capture_mode != "video":
            if fusion is not None and fusion.voxel_count > 0:
                cloud = fusion.to_point_cloud()
                ply_path = root / "scan_object.ply"
                o3d.io.write_point_cloud(str(ply_path), cloud, print_progress=False)
                log(f"Live posed fusion: {fusion.voxel_count} voxels -> {ply_path.name}")

            if not aborted:
                log("Returning turntable to zero ...")
                tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
                tt.rotate_to_zero(wait_s=TURNTABLE_HOME_WAIT_S)
            elif not _cancelled():
                tt.tilt_to_zero(wait_s=cfg.tilt_wait_s)
                tt.rotate_to_zero(wait_s=TURNTABLE_HOME_WAIT_S)

    finally:
        if show_preview:
            cv2.destroyAllWindows()
        tt.disconnect()

    log(f"Scan complete: {frame_idx} frames saved to {root}")

    if auto_process and frame_idx >= 2:
        from scanner.runner import process_session_safe

        log("Auto-processing (turntable posed fusion) ...")
        process_session_safe(root, prefer_icp=False)
    elif frame_idx >= 2:
        log(f"Run: python -m scanner process {root}")

    if frame_idx < 2:
        return None
    return root


def run_turntable_scan_direct(
    *,
    cfg: TurntableScanConfig | None = None,
    name: str = "scan",
    output_dir: Path = DEFAULT_SCANS_DIR,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
    auto_mask: bool | None = None,
) -> int:
    """Run turntable scan with a local RealSense (no web server)."""
    if cfg is None:
        cfg = TurntableScanConfig()
    if auto_mask is not None:
        cfg.auto_mask = auto_mask

    cam: RealSenseD405 | None = None
    try:
        if cfg.capture_mode == "video":
            want_fps = int(cfg.video_capture_fps)
            if want_fps <= 0:
                probed_fps, cam = probe_best_fps(
                    width=TURNTABLE_CAPTURE_WIDTH,
                    height=TURNTABLE_CAPTURE_HEIGHT,
                    lock_holder="turntable_scan_cli",
                )
                print(f"Video capture: using max aligned RGB-D profile {probed_fps} fps")
            else:
                cam = RealSenseD405(
                    width=TURNTABLE_CAPTURE_WIDTH,
                    height=TURNTABLE_CAPTURE_HEIGHT,
                    fps=want_fps,
                    lock_holder="turntable_scan_cli",
                )
                cam.start()
                print(f"Video capture: using {want_fps} fps")
        else:
            cam = RealSenseD405(
                width=TURNTABLE_CAPTURE_WIDTH,
                height=TURNTABLE_CAPTURE_HEIGHT,
                fps=int(TURNTABLE_CAPTURE_FPS),
                lock_holder="turntable_scan_cli",
            )
            cam.start()

        if cfg.capture_mode == "video":
            meta = cam.prepare_video_spin_capture()
            print(
                f"Video capture: AE unlocked for motion "
                f"(warmup rgb mean {meta.get('rgb_mean', 0):.0f})"
            )
        else:
            cam.prepare_turntable_capture()
    except RuntimeError as exc:
        print(f"Could not open RealSense D405: {exc}", file=sys.stderr)
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
        return 1

    class _FrameSource:
        def read_frame(self, *, block: bool = True, apply_filters: bool | None = None):
            return cam.read_frame(block=block, apply_filters=apply_filters)

    try:
        root = execute_turntable_scan(
            _FrameSource(),
            realsense=cam,
            cfg=cfg,
            name=name,
            output_dir=output_dir,
            auto_process=auto_process,
            use_calibration=use_calibration,
            calibration_dir=calibration_dir,
        )
    except Exception as exc:
        print(f"Turntable scan failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cam.stop()

    if root is None:
        return 1
    return 0


def run_turntable_scan(
    *,
    cfg: TurntableScanConfig | None = None,
    name: str = "scan",
    output_dir: Path = DEFAULT_SCANS_DIR,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
) -> int:
    """Legacy entry - delegates to web API."""
    from scanner.api_client import run_turntable_scan_via_api

    if cfg is None:
        cfg = TurntableScanConfig()

    print(
        "turntable-scan uses the web server API (camera is owned by the server).",
        file=sys.stderr,
    )
    return run_turntable_scan_via_api(
        name=name,
        output_dir=output_dir,
        auto_process=auto_process,
        use_calibration=use_calibration,
        calibration_dir=calibration_dir,
        tilt_levels=list(cfg.tilt_levels),
        rotate_step_deg=cfg.rotate_step_deg,
        rotate_wait_s=cfg.rotate_wait_s,
        tilt_wait_s=cfg.tilt_wait_s,
        ble_address=cfg.ble_address,
        scan_timeout=cfg.scan_timeout,
    )
