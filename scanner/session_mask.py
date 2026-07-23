"""Session-level scan masks — persist and apply before fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scanner.mask_motion import MotionMaskResult, compute_motion_mask, render_motion_diagnostics

try:
    from scanner.mask_rgb_video import RgbVideoMaskResult, render_mean_mad_mask_combo
except ImportError:
    RgbVideoMaskResult = None  # type: ignore[misc, assignment]
    render_mean_mad_mask_combo = None  # type: ignore[assignment]


@dataclass
class SessionMask:
    """2D include corridor + optional depth band (from motion preview or manual)."""

    version: int = 1
    method: str = "motion_preview"
    preview_frames: int = 0
    depth_band_mm: tuple[float, float] = (0.0, 2500.0)
    tau_depth_mm: float = 0.0
    tau_rgb: float = 0.0
    include_png: str = "mask_include.png"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["depth_band_mm"] = [float(self.depth_band_mm[0]), float(self.depth_band_mm[1])]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMask:
        band = data.get("depth_band_mm") or [0.0, 2500.0]
        return cls(
            version=int(data.get("version", 1)),
            method=str(data.get("method", "motion_preview")),
            preview_frames=int(data.get("preview_frames", 0)),
            depth_band_mm=(float(band[0]), float(band[1])),
            tau_depth_mm=float(data.get("tau_depth_mm", 0.0)),
            tau_rgb=float(data.get("tau_rgb", 0.0)),
            include_png=str(data.get("include_png", "mask_include.png")),
            extra=dict(data.get("extra") or {}),
        )


def build_mask_from_preview(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
) -> MotionMaskResult:
    return compute_motion_mask(rgb_frames, depth_frames)


def save_session_mask(
    session_root: Path,
    result: MotionMaskResult,
    rgb_frames: list[np.ndarray],
    *,
    method: str = "motion_preview",
) -> SessionMask:
    """Write include PNG + diagnostic JPEGs; return manifest block."""
    session_root.mkdir(parents=True, exist_ok=True)
    inc_path = session_root / "mask_include.png"
    cv2.imwrite(str(inc_path), (result.include.astype(np.uint8) * 255))

    diag = render_motion_diagnostics(rgb_frames, result)
    cv2.imwrite(str(session_root / "mask_motion_combo.jpg"), diag["combo"])
    cv2.imwrite(str(session_root / "mask_motion_heat.jpg"), diag["motion_heat"])
    cv2.imwrite(str(session_root / "mask_motion_overlay.jpg"), diag["overlay"])

    spec = SessionMask(
        method=method,
        preview_frames=result.preview_frame_count,
        depth_band_mm=result.depth_band_mm,
        tau_depth_mm=result.tau_depth_mm,
        tau_rgb=result.tau_rgb,
        include_png="mask_include.png",
        extra={
            "include_pixels": int(result.include.sum()),
            "static_pixels": int(result.static.sum()),
        },
    )
    (session_root / "mask.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return spec


def save_rgb_video_session_mask(
    session_root: Path,
    result: "RgbVideoMaskResult",
    *,
    combo_bgr: np.ndarray | None = None,
) -> SessionMask:
    """Write mean+MAD video mask for turntable process fusion."""
    session_root.mkdir(parents=True, exist_ok=True)
    inc_path = session_root / "mask_include.png"
    cv2.imwrite(str(inc_path), (result.include.astype(np.uint8) * 255))
    if combo_bgr is not None:
        cv2.imwrite(str(session_root / "mask_rgb_mean_mad_combo.jpg"), combo_bgr)

    spec = SessionMask(
        method=result.method,
        preview_frames=result.preview_frame_count,
        include_png="mask_include.png",
        extra={
            "include_pixels": int(result.include.sum()),
            "tau_mean": result.tau_mean,
            "tau_mad": result.tau_mad,
            "holes_filled": True,
        },
    )
    (session_root / "mask.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return spec


def load_session_mask(session_root: Path) -> SessionMask | None:
    path = session_root / "mask.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionMask.from_dict(data)


def load_include_mask(session_root: Path, spec: SessionMask) -> np.ndarray | None:
    path = session_root / spec.include_png
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 0


def apply_mask_to_depth(
    depth_mm: np.ndarray,
    include: np.ndarray,
    *,
    depth_band_mm: tuple[float, float] | None = None,
) -> np.ndarray:
    """Zero depth outside include mask (and optional depth band)."""
    out = np.zeros_like(depth_mm)
    keep = include & (depth_mm > 0)
    if depth_band_mm is not None:
        lo, hi = depth_band_mm
        keep &= (depth_mm >= lo) & (depth_mm <= hi)
    out[keep] = depth_mm[keep]
    return out


def mask_depth_for_session(
    session_root: Path,
    depth_mm: np.ndarray,
    spec: SessionMask | None = None,
    include: np.ndarray | None = None,
) -> np.ndarray:
    """Apply saved session mask if present; otherwise return depth unchanged."""
    if spec is None:
        spec = load_session_mask(session_root)
    if spec is None:
        return depth_mm
    if include is None:
        include = load_include_mask(session_root, spec)
    if include is None:
        return depth_mm
    if include.shape != depth_mm.shape[:2]:
        include = cv2.resize(
            include.astype(np.uint8),
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return apply_mask_to_depth(
        depth_mm,
        include,
        depth_band_mm=spec.depth_band_mm,
    )
