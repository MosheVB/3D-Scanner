"""Load ChArUco calibration JSON for capture (newest file in calibration/)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scanner.config import DEFAULT_CALIBRATION_DIR

_CALIB_GLOB = "realsense_d405_*.json"


@dataclass(frozen=True)
class CalibrationInfo:
    path: Path
    camera_matrix: list[list[float]]
    image_size: tuple[int, int]
    rms_px: float | None

    @property
    def label(self) -> str:
        rms = f", RMS {self.rms_px:.2f} px" if self.rms_px is not None else ""
        return f"{self.path.name}{rms}"


def find_latest_calibration(cal_dir: Path | None = DEFAULT_CALIBRATION_DIR) -> Path | None:
    """Newest realsense_d405_*.json by modification time."""
    if cal_dir is None or not cal_dir.is_dir():
        return None
    candidates = sorted(cal_dir.glob(_CALIB_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_calibration(path: Path) -> CalibrationInfo:
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("calibration") or data
    k = cal.get("camera_matrix")
    if not k:
        raise ValueError(f"No camera_matrix in {path}")
    size = cal.get("image_size") or [0, 0]
    w, h = int(size[0]), int(size[1])
    rms = cal.get("rms_px")
    return CalibrationInfo(
        path=path,
        camera_matrix=[[float(v) for v in row] for row in k],
        image_size=(w, h),
        rms_px=float(rms) if rms is not None else None,
    )


def load_latest_calibration(
    cal_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> CalibrationInfo | None:
    path = find_latest_calibration(cal_dir)
    if path is None:
        return None
    return load_calibration(path)


def resolve_intrinsics(
    factory_intrinsics: Sequence[Sequence[float]],
    resolution: tuple[int, int],
    *,
    use_calibration: bool = True,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> tuple[list[list[float]], str]:
    """Pick calibrated K or factory K. Returns (intrinsics, source line for logging)."""
    if not use_calibration:
        return [list(map(float, row)) for row in factory_intrinsics], "factory (--no-calibration)"

    try:
        info = load_latest_calibration(calibration_dir)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return (
            [list(map(float, row)) for row in factory_intrinsics],
            f"factory (calibration file unreadable: {exc})",
        )

    if info is None:
        return (
            [list(map(float, row)) for row in factory_intrinsics],
            f"factory (no {_CALIB_GLOB} in {calibration_dir})",
        )

    if info.image_size != resolution:
        return (
            [list(map(float, row)) for row in factory_intrinsics],
            (
                f"factory (calibration {info.image_size[0]}×{info.image_size[1]} "
                f"≠ camera {resolution[0]}×{resolution[1]})"
            ),
        )

    rms = f" (RMS {info.rms_px:.2f} px)" if info.rms_px is not None else ""
    return info.camera_matrix, f"calibration/{info.path.name}{rms}"
