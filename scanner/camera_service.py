"""Single-process camera owner: streaming, jobs, voxel preview, turntable API."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from scanner.depth_fusion import center_focus_mm, object_mask_center
from scanner.depth_vis import colorize_depth_mm
from scanner.jobs import Job, JobManager, JobStatus
from scanner.voxel_overlay import LiveVoxelRgbOverlay
from scanner.realsense_camera import RealSenseD405
from scanner.config import DEFAULT_CALIBRATION_DIR
from scanner.scan_mask import ScanMaskState, draw_click_markers
from scanner.intrinsics import resolve_intrinsics
from scanner.host_telemetry import collect_host_telemetry
from scanner.turntable_capture import TurntableScanConfig, execute_turntable_scan
from scanner.turntable_service import TurntableService

_BOUNDARY = b"frame"


def _encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _placeholder_bgr(width: int, height: int, message: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (24, 24, 28)
    cv2.putText(
        img,
        message,
        (16, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def _placeholder_jpeg(message: str, *, width: int = 640, height: int = 480) -> bytes:
    return _encode_jpeg(_placeholder_bgr(width, height, message))


class CameraService:
    """Owns the RealSense pipeline and coordinates stream + scan jobs."""

    def __init__(self) -> None:
        self._cam = RealSenseD405(lock_holder="web")
        self._frame_lock = threading.Lock()
        self._voxel_lock = threading.Lock()
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self.jobs = JobManager()
        self.turntable = TurntableService(self.jobs)
        self.scan_mask = ScanMaskState()
        self._voxel_view = LiveVoxelRgbOverlay(band_mm=40.0)
        self._live_voxel_on_rgb = True
        self._live_voxel_band_mm = 40.0
        self._scan_intrinsics: Sequence[Sequence[float]] | None = None
        self._rgb_jpeg = _placeholder_jpeg("Camera starting…")
        self._depth_jpeg = _placeholder_jpeg("Camera starting…")
        self._voxel_jpeg = _placeholder_jpeg("Voxel model — starts with scan")
        self._fps = 0.0
        self._stream_frames = 0
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._started = False

    @property
    def camera(self) -> RealSenseD405:
        return self._cam

    @property
    def camera_connected(self) -> bool:
        return self._cam.is_running()

    def start(self) -> None:
        if self._started:
            return
        self._cam.start()
        try:
            self._cam.prepare_turntable_capture()
        except Exception:
            pass
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="camera-stream", daemon=True
        )
        self._stream_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        self._cam.stop()
        self.turntable.disconnect()
        with self._voxel_lock:
            self._voxel_view.close()
        self._started = False

    def connect_camera(self) -> None:
        if self._cam.is_running():
            return
        if self.jobs.active_job() is not None:
            raise RuntimeError("Cannot connect camera during an active scan job")
        self._cam.start()
        try:
            self._cam.prepare_turntable_capture()
        except Exception:
            pass
        if not self._stream_thread or not self._stream_thread.is_alive():
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_loop, name="camera-stream", daemon=True
            )
            self._stream_thread.start()
        self._started = True
        self.reapply_mask_clicks()

    def disconnect_camera(self) -> None:
        if self.jobs.active_job() is not None:
            raise RuntimeError("Cannot disconnect camera during an active scan job")
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        self._cam.stop()
        with self._frame_lock:
            self._latest_rgb = None
            self._latest_depth = None
            self._rgb_jpeg = _placeholder_jpeg("Camera disconnected")
            self._depth_jpeg = _placeholder_jpeg("Camera disconnected")
            self._fps = 0.0

    def read_frame(self, *, block: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        # Turntable scan uses block=False; the stream loop already owns the pipeline.
        if not block:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with self._frame_lock:
                    if self._latest_rgb is not None and self._latest_depth is not None:
                        return self._latest_rgb.copy(), self._latest_depth.copy()
                time.sleep(0.02)
            return None
        return self._cam.read_frame(block=block)

    def _mask_intrinsics(self) -> Sequence[Sequence[float]] | None:
        if self._scan_intrinsics is not None:
            return self._scan_intrinsics
        if not self.camera_connected:
            return None
        intrinsics, _ = resolve_intrinsics(
            self._cam.intrinsics,
            self._cam.resolution,
            use_calibration=True,
            calibration_dir=DEFAULT_CALIBRATION_DIR,
        )
        return intrinsics

    def set_mask_clicks(
        self,
        points: Sequence[Sequence[int | float]] | None = None,
        *,
        include: Sequence[Sequence[int | float]] | None = None,
        exclude: Sequence[Sequence[int | float]] | None = None,
        mode: str | None = None,
        use_haiku: bool | None = None,
    ) -> dict[str, Any]:
        with self._frame_lock:
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
        intrinsics = self._mask_intrinsics()
        self.scan_mask.set_clicks(
            points,
            depth,
            intrinsics,
            rgb_bgr=rgb,
            include=include,
            exclude=exclude,
            mode=mode if mode in ("include", "exclude") else None,  # type: ignore[arg-type]
            use_haiku=use_haiku,
        )
        with self._voxel_lock:
            self._sync_voxel_mask()
        with self._frame_lock:
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
        if depth is not None and self._live_voxel_on_rgb:
            self._ensure_live_voxel(depth)
        return self.scan_mask.status_dict()

    def reapply_mask_clicks(self) -> dict[str, Any] | None:
        """Re-run mask logic after camera/depth becomes available."""
        include = self.scan_mask.include_uvs()
        if not include:
            return None
        exclude = self.scan_mask.exclude_uvs()
        status = self.scan_mask.status_dict()
        use_haiku = status.get("use_haiku")
        return self.set_mask_clicks(
            include=[[u, v] for u, v in include],
            exclude=[[u, v] for u, v in exclude],
            use_haiku=use_haiku,
        )

    def append_mask_click(
        self,
        u: int,
        v: int,
        *,
        mode: str = "include",
        use_haiku: bool | None = None,
    ) -> dict[str, Any]:
        """Append one mask click server-side (avoids client sync races)."""
        include = list(self.scan_mask.include_uvs())
        exclude = list(self.scan_mask.exclude_uvs())
        u_i, v_i = int(u), int(v)
        if mode == "exclude":
            if not include:
                raise ValueError("Need at least one include click before exclude")
            if len(exclude) >= 16:
                raise ValueError("At most 16 exclude clicks")
            exclude.append((u_i, v_i))
        else:
            if len(include) >= 16:
                raise ValueError("At most 16 include clicks")
            include.append((u_i, v_i))
        return self.set_mask_clicks(
            include=[[a, b] for a, b in include],
            exclude=[[a, b] for a, b in exclude],
            use_haiku=use_haiku,
        )

    def set_mask_use_haiku(self, enabled: bool) -> dict[str, Any]:
        self.scan_mask.set_use_haiku(enabled)
        if enabled:
            with self._frame_lock:
                depth = self._latest_depth.copy() if self._latest_depth is not None else None
                rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            intrinsics = self._mask_intrinsics()
            if self.scan_mask.ready and depth is not None and intrinsics is not None:
                self.scan_mask.request_haiku_assist(depth, intrinsics, rgb_bgr=rgb)
        return self.scan_mask.status_dict()

    def assist_mask_haiku(self) -> dict[str, Any]:
        with self._frame_lock:
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
        intrinsics = self._mask_intrinsics()
        started = self.scan_mask.request_haiku_assist(depth, intrinsics, rgb_bgr=rgb)
        status = self.scan_mask.status_dict()
        status["haiku_requested"] = started
        return status

    def clear_mask_clicks(self) -> None:
        self.scan_mask.clear()
        with self._voxel_lock:
            self._sync_voxel_mask()

    def mask_status(self) -> dict[str, Any]:
        return self.scan_mask.status_dict()

    def patch_mask_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self.scan_mask.patch_settings(updates)

    def get_camera_options(self) -> dict[str, Any]:
        return self._cam.get_options()

    def patch_camera_options(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self._cam.patch_options(updates)

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "host": collect_host_telemetry(),
            "camera": self._cam.get_camera_telemetry(),
        }

    def _live_mask_builder(self, depth_mm: np.ndarray) -> np.ndarray | None:
        with self._frame_lock:
            rgb = self._latest_rgb
        intrinsics = self._mask_intrinsics()
        if rgb is None or intrinsics is None:
            return None
        focus = self._voxel_view.focus_mm or center_focus_mm(depth_mm)
        seed = self._voxel_view.seed_uv
        if seed is None:
            h, w = depth_mm.shape[:2]
            seed = (w // 2, h // 2)
        return object_mask_center(
            depth_mm,
            center_mm=focus,
            band_mm=self._live_voxel_band_mm,
            seed_u=seed[0],
            seed_v=seed[1],
            rgb_bgr=rgb,
            intrinsics=intrinsics,
        )

    def _sync_voxel_mask(self) -> None:
        if self._live_voxel_on_rgb:
            self._voxel_view.set_mask_fn(self._live_mask_builder)
        elif self.scan_mask.ready:
            self._voxel_view.set_mask_fn(self.scan_mask.build_mask)
        else:
            self._voxel_view.set_mask_fn(None)

    def set_live_voxel_on_rgb(self, enabled: bool) -> dict[str, Any]:
        self._live_voxel_on_rgb = bool(enabled)
        with self._voxel_lock:
            self._sync_voxel_mask()
        with self._frame_lock:
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
        if enabled and depth is not None:
            self._ensure_live_voxel(depth)
        return {"live_voxel_on_rgb": self._live_voxel_on_rgb}

    def _ensure_live_voxel(self, depth_mm: np.ndarray) -> None:
        intrinsics = self._mask_intrinsics()
        if intrinsics is None:
            return
        with self._voxel_lock:
            if self._voxel_view.active:
                return
            self._voxel_view.auto_init_from_center(depth_mm, intrinsics)
            self._sync_voxel_mask()

    def _tick_live_voxel(self, rgb: np.ndarray, depth_mm: np.ndarray) -> None:
        if not self._live_voxel_on_rgb:
            return
        self._ensure_live_voxel(depth_mm)
        with self._voxel_lock:
            if self._voxel_view.active:
                self._voxel_view.accumulate(rgb, depth_mm)
                self._voxel_jpeg = _encode_jpeg(self._voxel_view.render_panel_bgr())

    def rgb_jpeg(self) -> bytes:
        with self._frame_lock:
            return self._rgb_jpeg

    def depth_jpeg(self) -> bytes:
        with self._frame_lock:
            return self._depth_jpeg

    def voxel_jpeg(self) -> bytes:
        with self._voxel_lock:
            return self._voxel_jpeg

    def reset_voxel(self) -> None:
        with self._voxel_lock:
            if self._live_voxel_on_rgb and self._voxel_view.active:
                self._voxel_view.reset_model()
            else:
                self._voxel_view.close()
                self._scan_intrinsics = None
                self._voxel_jpeg = _placeholder_jpeg("Voxel model — starts with scan")
            self._sync_voxel_mask()

    def _init_voxel_for_scan(self, depth_mm: np.ndarray) -> None:
        if self._scan_intrinsics is None:
            return
        with self._voxel_lock:
            if self.scan_mask.ready:
                uvs = self.scan_mask.click_uvs()
                focus = self.scan_mask.focus_mm
                if uvs and focus is not None:
                    self._voxel_view.init_from_scan_mask(
                        self._scan_intrinsics,
                        seed_uv=uvs[0],
                        focus_mm=focus,
                    )
                    self._voxel_view.set_mask_fn(self.scan_mask.build_mask)
                    return
            if not self._voxel_view.active:
                self._voxel_view.auto_init_from_center(depth_mm, self._scan_intrinsics)

    def _update_mask_tracking(self, rgb: np.ndarray, depth_mm: np.ndarray) -> None:
        if not self.scan_mask.ready:
            return
        intrinsics = self._scan_intrinsics or self._mask_intrinsics()
        if intrinsics is None:
            return
        self.scan_mask.update_frame(rgb, depth_mm, intrinsics)

    def _update_voxel_from_frame(self, rgb: np.ndarray, depth_mm: np.ndarray) -> None:
        self._update_mask_tracking(rgb, depth_mm)
        if self._scan_intrinsics is None:
            return
        with self._voxel_lock:
            if not self._voxel_view.active:
                self._init_voxel_for_scan(depth_mm)
            if self._voxel_view.active:
                self._voxel_view.accumulate(rgb, depth_mm)
                panel = self._voxel_view.render_panel_bgr()
                self._voxel_jpeg = _encode_jpeg(panel)

    def status_dict(self) -> dict[str, Any]:
        active = self.jobs.active_job()
        with self._voxel_lock:
            voxel = {
                "active": self._voxel_view.active,
                "points": self._voxel_view.point_count,
                "frames": self._voxel_view.frame_count,
                "live_on_rgb": self._live_voxel_on_rgb,
            }
        return {
            "camera_connected": self.camera_connected,
            "camera": {
                "connected": self.camera_connected,
                "device": self._cam.label if self.camera_connected else None,
                "serial": getattr(self._cam, "_device_serial", ""),
                "resolution": list(self._cam.resolution),
                "fps_target": self._cam.fps,
                "fps_actual": round(self._fps, 1) if self.camera_connected else 0.0,
                "stream_frames": self._stream_frames,
                "running": self.camera_connected,
            },
            "streaming": self._started and not self._stream_stop.is_set() and self.camera_connected,
            "turntable": self.turntable.status_dict(),
            "mask": self.scan_mask.status_dict(),
            "voxel": voxel,
            "active_job": active.to_dict() if active else None,
        }

    def _stream_loop(self) -> None:
        window: list[float] = []
        while not self._stream_stop.is_set():
            if not self._cam.is_running():
                time.sleep(0.05)
                continue
            pair = self._cam.read_frame(block=True)
            if pair is None:
                continue
            rgb, depth_mm = pair
            focus = self.scan_mask.focus_mm
            depth_vis = colorize_depth_mm(
                depth_mm,
                focus_mm=focus,
                use_d405_range=focus is None,
            )
            intrinsics = self._mask_intrinsics()
            if self.scan_mask.ready and intrinsics is not None:
                self.scan_mask.update_frame(rgb, depth_mm, intrinsics)
            if self.scan_mask.needs_camera_depth and intrinsics is not None:
                self.reapply_mask_clicks()
            include_clicks = self.scan_mask.include_uvs()
            exclude_clicks = self.scan_mask.exclude_uvs()
            tracker = self.scan_mask.tracker
            if include_clicks or exclude_clicks:
                rgb_vis = draw_click_markers(
                    rgb,
                    include_clicks,
                    exclude_clicks=exclude_clicks,
                    bounds_tracker=tracker if intrinsics is not None else None,
                    intrinsics=intrinsics,
                )
                depth_vis = draw_click_markers(
                    depth_vis,
                    include_clicks,
                    exclude_clicks=exclude_clicks,
                    bounds_tracker=tracker if intrinsics is not None else None,
                    intrinsics=intrinsics,
                )
            else:
                rgb_vis = rgb

            self._tick_live_voxel(rgb, depth_mm)
            if (
                self._live_voxel_on_rgb
                and self._voxel_view.active
            ):
                with self._voxel_lock:
                    rgb_vis = self._voxel_view.render_on_rgb(rgb_vis, depth_mm)

            rgb_jpeg = _encode_jpeg(rgb_vis)
            depth_jpeg = _encode_jpeg(depth_vis)
            now = time.monotonic()
            window.append(now)
            window = [t for t in window if now - t <= 1.0]
            with self._frame_lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_mm
                self._rgb_jpeg = rgb_jpeg
                self._depth_jpeg = depth_jpeg
                self._fps = float(len(window))
                self._stream_frames += 1

    def submit_turntable_scan(
        self,
        *,
        name: str,
        output_dir: Path,
        auto_process: bool,
        use_calibration: bool,
        calibration_dir: Path | None,
        cfg: TurntableScanConfig,
    ) -> Job:
        if not self.camera_connected:
            raise RuntimeError("Camera is disconnected — connect camera first")
        if self.jobs.active_job() is not None:
            raise RuntimeError("Another job is already running")

        job = self.jobs.create(
            "turntable_scan",
            total=cfg.total_frames,
            message="Queued",
        )

        def _worker() -> None:
            from scanner.intrinsics import resolve_intrinsics

            self.jobs.update(
                job,
                status=JobStatus.RUNNING,
                message="Connecting turntable",
            )
            self.reset_voxel()

            intrinsics, _ = resolve_intrinsics(
                self._cam.intrinsics,
                self._cam.resolution,
                use_calibration=use_calibration,
                calibration_dir=calibration_dir,
            )
            self._scan_intrinsics = intrinsics

            def on_progress(current: int, total: int, message: str) -> None:
                self.jobs.update(
                    job,
                    progress_current=current,
                    progress_total=total,
                    message=message,
                )

            def on_frame(rgb: np.ndarray, depth_mm: np.ndarray) -> None:
                self._update_voxel_from_frame(rgb, depth_mm)

            try:
                root = execute_turntable_scan(
                    self,
                    realsense=self._cam,
                    cfg=cfg,
                    name=name,
                    output_dir=output_dir,
                    auto_process=auto_process,
                    use_calibration=use_calibration,
                    calibration_dir=calibration_dir,
                    on_progress=on_progress,
                    on_frame=on_frame,
                    cancel_event=job.cancel_event,
                    show_preview=False,
                    log=lambda msg: print(msg, flush=True),
                )
            except Exception as exc:
                if job.cancel_event.is_set():
                    self.jobs.update(
                        job,
                        status=JobStatus.CANCELLED,
                        message="Cancelled",
                        error=str(exc) if str(exc) else None,
                    )
                else:
                    self.jobs.update(
                        job,
                        status=JobStatus.FAILED,
                        message="Failed",
                        error=str(exc),
                    )
                return

            if job.cancel_event.is_set():
                self.jobs.update(
                    job,
                    status=JobStatus.CANCELLED,
                    message="Cancelled",
                )
                return

            if root is None:
                self.jobs.update(
                    job,
                    status=JobStatus.FAILED,
                    message="Not enough frames",
                    error="Need at least 2 frames",
                )
                return

            self.jobs.update(
                job,
                status=JobStatus.COMPLETED,
                message="Done",
                result_path=str(root),
                progress_current=job.progress_current,
            )

        threading.Thread(target=_worker, name=f"job-{job.id}", daemon=True).start()
        return job

    def cancel_job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        job.cancel_event.set()
        self.jobs.update(job, message="Cancelling…")
        return job
