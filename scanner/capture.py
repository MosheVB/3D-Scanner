"""Interactive RGB-D capture for turntable or handheld scanning."""

from __future__ import annotations

import sys
from pathlib import Path

# IDE / terminal: python scanner/capture.py
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scanner.config import (
    CLICK_PATCH_RADIUS,
    CLICK_SEARCH_MAX_RADIUS,
    D405_MIN_DEPTH_MM,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_SCANS_DIR,
    DISTANCE_OPTIMAL_MAX_MM,
    DISTANCE_OPTIMAL_MIN_MM,
    DISTANCE_TOO_CLOSE_MM,
    DISTANCE_TOO_FAR_MM,
    HUD_CLICK_EXCLUDE_TOP_PX,
    LEFT_DEPTH_STRIP_H,
    LEFT_RGB_PANEL_H,
    LIVE_DEPTH_BAND_MM,
    RGB_HEIGHT,
    RGB_WIDTH,
    TURNTABLE_ROTATE_STEP_DEG,
)
from scanner.live_preview import LivePointCloudView
from scanner.intrinsics import resolve_intrinsics
from scanner.realsense_camera import RealSenseD405
from scanner.runner import process_session_safe
from scanner.session import ScanMode, add_frame, create_session, session_root

_CONTROLS_LINE = "Click object = start 3D build | SPACE save | P mesh | Q quit"


def _colorize_depth(
    depth_mm: np.ndarray,
    *,
    focus_mm: float | None = None,
    band_mm: float = LIVE_DEPTH_BAND_MM,
) -> np.ndarray:
    """Colormap depth; when focus is set, scale to close-up band around the click."""
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)
    if focus_mm is not None:
        lo = max(0.0, focus_mm - band_mm)
        hi = focus_mm + band_mm
    else:
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((depth_mm.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def _map_window_click_to_uv(x: int, y: int) -> tuple[int, int] | None:
    """Map click on left column (RGB area) to image pixel (u, v)."""
    if y < 0 or y >= LEFT_RGB_PANEL_H:
        return None
    if 0 <= x < RGB_WIDTH:
        return x, y
    return None


def _median_depth_patch(
    depth_mm: np.ndarray, u: int, v: int, radius: int
) -> float | None:
    h, w = depth_mm.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    r = int(radius)
    patch = depth_mm[max(0, v - r) : min(h, v + r + 1), max(0, u - r) : min(w, u + r + 1)]
    valid = patch[patch > 0]
    if valid.size < 3:
        return None
    return float(np.median(valid))


def _sample_depth_mm(depth_mm: np.ndarray, u: int, v: int) -> tuple[float | None, int]:
    """Depth at (u,v): prefer valid pixels nearest the click, then median in growing patches."""
    h, w = depth_mm.shape[:2]
    u0 = int(np.clip(u, 0, w - 1))
    v0 = int(np.clip(v, 0, h - 1))

    best_dist: float | None = None
    best_r2 = float("inf")
    for radius in range(CLICK_PATCH_RADIUS, CLICK_SEARCH_MAX_RADIUS + 1, 2):
        r = int(radius)
        v_lo, v_hi = max(0, v0 - r), min(h, v0 + r + 1)
        u_lo, u_hi = max(0, u0 - r), min(w, u0 + r + 1)
        patch = depth_mm[v_lo:v_hi, u_lo:u_hi]
        ys, xs = np.where(patch > 0)
        if ys.size == 0:
            continue
        dists = patch[ys, xs].astype(np.float64)
        du = (u_lo + xs) - u0
        dv = (v_lo + ys) - v0
        r2 = du * du + dv * dv
        j = int(np.argmin(r2))
        if r2[j] < best_r2:
            best_r2 = float(r2[j])
            best_dist = float(dists[j])
        median = _median_depth_patch(depth_mm, u0, v0, r)
        if median is not None:
            return median, r

    if best_dist is not None:
        return best_dist, CLICK_SEARCH_MAX_RADIUS
    return None, 0


def _center_depth_mm(depth_mm: np.ndarray, half_size: int = 48) -> float | None:
    """Depth at image center (live guide when clicks miss holes)."""
    h, w = depth_mm.shape[:2]
    cy, cx = h // 2, w // 2
    return _median_depth_patch(depth_mm, cx, cy, half_size)


def _distance_guidance(dist_mm: float) -> str:
    if dist_mm < DISTANCE_TOO_CLOSE_MM:
        return "too close"
    if dist_mm > DISTANCE_TOO_FAR_MM:
        return "too far"
    if DISTANCE_OPTIMAL_MIN_MM <= dist_mm <= DISTANCE_OPTIMAL_MAX_MM:
        return "optimal"
    if dist_mm < DISTANCE_OPTIMAL_MIN_MM:
        return "move back slightly"
    return "move closer slightly"


def _guidance_bgr(guidance: str) -> tuple[int, int, int]:
    if guidance == "optimal":
        return (0, 255, 0)
    if guidance == "no_depth":
        return (255, 0, 255)
    if guidance in ("too close", "too far"):
        return (0, 0, 255)
    return (0, 200, 255)


def _format_distance_line(dist_mm: float, *, prefix: str = "Distance") -> str:
    cm = dist_mm / 10.0
    return f"{prefix}: {dist_mm:.0f} mm ({cm:.1f} cm) — {_distance_guidance(dist_mm)}"


def _draw_click_marker(frame: np.ndarray, u: int, v: int, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    r = 12
    cv2.circle(frame, (u, v), r, color, 1, cv2.LINE_AA)
    cv2.line(frame, (u - r - 4, v), (u + r + 4, v), color, 1, cv2.LINE_AA)
    cv2.line(frame, (u, v - r - 4), (u, v + r + 4), color, 1, cv2.LINE_AA)


class _ClickDistanceState:
    """Mouse click → depth sample; persists last reading for the session."""

    def __init__(
        self,
        *,
        intrinsics: list[list[float]] | None = None,
        live_view: LivePointCloudView | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.live_view = live_view
        self.last_depth_mm: np.ndarray | None = None
        self.click_uv: tuple[int, int] | None = None
        self.distance_mm: float | None = None
        self.no_depth_at_click: bool = False
        self.search_radius: int = 0
        self.on_depth_panel: bool = False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.last_depth_mm is None:
            return
        if x >= RGB_WIDTH:
            print("Click the object on the left (camera) image to start building.")
            return
        if y < HUD_CLICK_EXCLUDE_TOP_PX:
            print("Click below the text on the object.")
            return
        uv = _map_window_click_to_uv(x, y)
        if uv is None:
            return
        u, v = uv
        self.on_depth_panel = False
        # Always move reticle to the click (even when stereo has holes up close)
        self.click_uv = (u, v)
        dist, radius = _sample_depth_mm(self.last_depth_mm, u, v)
        if dist is None:
            self.distance_mm = None
            self.no_depth_at_click = True
            self.search_radius = 0
            center = _center_depth_mm(self.last_depth_mm)
            print(
                f"No depth at click (D405 min ~{D405_MIN_DEPTH_MM} mm — too close, "
                "or object surface has no return). Back up or try Center line."
            )
            if center is not None:
                print(_format_distance_line(center, prefix="Center"))
            return
        self.no_depth_at_click = False
        self.distance_mm = dist
        self.search_radius = radius
        if self.live_view is not None and self.intrinsics is not None:
            self.live_view.set_focus(dist, self.intrinsics, (u, v))
        line = _format_distance_line(dist)
        if radius > CLICK_PATCH_RADIUS:
            line += f" (filled {radius}px)"
        print(line)

    def distance_hud_line(self) -> str | None:
        if self.no_depth_at_click and self.click_uv is not None:
            return f"Click: no depth — back up (D405 min ~{D405_MIN_DEPTH_MM} mm) or try Center"
        if self.distance_mm is None:
            return None
        return _format_distance_line(self.distance_mm)

    def marker_color(self) -> tuple[int, int, int] | None:
        if self.click_uv is None:
            return None
        if self.no_depth_at_click:
            return _guidance_bgr("no_depth")
        if self.distance_mm is None:
            return None
        return _guidance_bgr(_distance_guidance(self.distance_mm))


def _preview_image(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    *,
    mode: ScanMode,
    frame_count: int,
    mode_hint: str,
    click_state: _ClickDistanceState,
    live_view: LivePointCloudView,
    status_line: str | None = None,
) -> np.ndarray:
    rgb_vis = rgb.copy()
    if click_state.click_uv is not None:
        color = click_state.marker_color()
        if color is not None:
            _draw_click_marker(rgb_vis, click_state.click_uv[0], click_state.click_uv[1], color)

    lines = [
        f"Mode: {mode} | Frames: {frame_count}",
        mode_hint,
        _CONTROLS_LINE,
    ]
    center = _center_depth_mm(depth_mm)
    if center is not None:
        lines.append(_format_distance_line(center, prefix="Center"))
    dist_line = click_state.distance_hud_line()
    if dist_line:
        lines.append(dist_line)
    if live_view.active:
        lines.append(f"3D object model: {live_view.point_count} voxels")
    if status_line:
        lines.append(status_line)
    hud = _draw_hud(rgb_vis, lines)
    focus = live_view.focus_mm if live_view.active else click_state.distance_mm
    depth_vis = _colorize_depth(depth_mm, focus_mm=focus)
    depth_strip = cv2.resize(depth_vis, (RGB_WIDTH, LEFT_DEPTH_STRIP_H))
    if focus is not None:
        cv2.putText(
            depth_strip,
            f"Depth {focus - LIVE_DEPTH_BAND_MM:.0f}-{focus + LIVE_DEPTH_BAND_MM:.0f} mm",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    rgb_panel = cv2.resize(hud, (RGB_WIDTH, LEFT_RGB_PANEL_H))
    left = np.vstack([rgb_panel, depth_strip])
    pcl_panel = live_view.render_panel_bgr(RGB_WIDTH, RGB_HEIGHT)
    return np.hstack([left, pcl_panel])


def _draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return out


def run_capture(
    *,
    mode: ScanMode = "turntable",
    name: str = "scan",
    output_dir: Path = DEFAULT_SCANS_DIR,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
) -> int:
    cam = RealSenseD405()
    try:
        cam.start()
    except RuntimeError as exc:
        print(
            f"Could not open RealSense D405: {exc}\n"
            "  • Close RealSense Viewer or any other app using the camera.\n"
            "  • Unplug, wait 5 s, replug into a USB 3 port.",
            file=sys.stderr,
        )
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
    print(
        "Controls: click object = start 3D build | SPACE = save frame | P = mesh | Q = quit"
    )
    print(
        f"Close-up target: {DISTANCE_OPTIMAL_MIN_MM}–{DISTANCE_OPTIMAL_MAX_MM} mm "
        f"(D405 range ~{D405_MIN_DEPTH_MM}–400 mm)"
    )

    if mode == "turntable":
        mode_hint = "Rotate object ~15-30° between captures"
    else:
        mode_hint = (
            "Close-up 8+ frames: object fills frame; "
            f"~{DISTANCE_OPTIMAL_MIN_MM//10}–{DISTANCE_OPTIMAL_MAX_MM//10} cm on dark mat"
        )

    window = "3D Scanner — capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    live_view = LivePointCloudView()
    if mode == "turntable":
        live_view.enable_analytical_turntable(rotate_step_deg=TURNTABLE_ROTATE_STEP_DEG)
    click_state = _ClickDistanceState(intrinsics=intrinsics, live_view=live_view)
    cv2.setMouseCallback(window, click_state.on_mouse)
    last_rgb: np.ndarray | None = None
    last_depth_mm: np.ndarray | None = None

    try:
        while cam.is_running():
            pair = cam.read_frame(block=False)
            if pair is not None:
                last_rgb, last_depth_mm = pair
                click_state.last_depth_mm = last_depth_mm

            if last_rgb is not None and last_depth_mm is not None:
                combined = _preview_image(
                    last_rgb,
                    last_depth_mm,
                    mode=mode,
                    frame_count=len(session.frames),
                    mode_hint=mode_hint,
                    click_state=click_state,
                    live_view=live_view,
                )
                cv2.imshow(window, combined)
            else:
                live_view.poll()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if last_rgb is None or last_depth_mm is None:
                    continue
                add_frame(session, last_rgb, last_depth_mm)
                frame_idx = len(session.frames) - 1
                if live_view.active:
                    if mode == "turntable":
                        live_view.accumulate(
                            last_rgb,
                            last_depth_mm,
                            rotation_deg=frame_idx * TURNTABLE_ROTATE_STEP_DEG,
                        )
                    else:
                        live_view.accumulate(last_rgb, last_depth_mm)
                print(f"Captured frame {frame_idx}")
            elif key in (ord("p"), ord("P")):
                if len(session.frames) < 2:
                    print("Capture at least 2 frames before processing.")
                    continue
                if last_rgb is not None and last_depth_mm is not None:
                    processing_view = _preview_image(
                        last_rgb,
                        last_depth_mm,
                        mode=mode,
                        frame_count=len(session.frames),
                        mode_hint=mode_hint,
                        click_state=click_state,
                        live_view=live_view,
                        status_line="Processing mesh (live preview resumes)...",
                    )
                    cv2.imshow(window, processing_view)
                    cv2.waitKey(1)
                code = process_session_safe(root, prefer_icp=mode != "turntable")
                if code != 0:
                    print(
                        "Processing failed; fix errors above or capture more frames.",
                        file=sys.stderr,
                    )
                else:
                    print("Processing done — continue capturing or press Q to quit.")
    finally:
        live_view.close()
        cam.stop()
        cv2.destroyAllWindows()

    if auto_process and len(session.frames) >= 2:
        process_session_safe(root, prefer_icp=True)
    elif len(session.frames) >= 2:
        print(f"Run: python -m scanner process {root}")

    return 0


if __name__ == "__main__":
    from scanner.__main__ import main

    raise SystemExit(main(["capture", *sys.argv[1:]]))
