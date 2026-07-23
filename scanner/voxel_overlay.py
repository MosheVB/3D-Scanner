"""Live voxel build drawn on top of the RGB camera image."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from scanner.live_preview import LivePointCloudView
from scanner.voxel_model import ObjectVoxelModel


def project_points_to_uv(
    points_m: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame XYZ (m) -> pixel (u, v), Y-up."""
    if points_m.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    z = points_m[:, 2]
    ok = z > 0.01
    u = np.rint(points_m[ok, 0] * fx / z[ok] + cx).astype(np.int32)
    v = np.rint(-points_m[ok, 1] * fy / z[ok] + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside]


class LiveVoxelRgbOverlay(LivePointCloudView):
    """Integrate voxels and paint coverage / holes on the live RGB feed."""

    def __init__(self, *, band_mm: float = 40.0) -> None:
        super().__init__()
        self._band_mm = float(band_mm)
        self._coverage: np.ndarray | None = None
        self._last_mask: np.ndarray | None = None
        self._rgb_ref: np.ndarray | None = None

    def reset_model(self) -> None:
        if self._model is not None and self._seed_uv is not None and self._focus_mm is not None:
            u, v = self._seed_uv
            k = np.asarray(self._intrinsics, dtype=np.float64)
            z = self._focus_mm / 1000.0
            x = (u - k[0, 2]) * z / k[0, 0]
            y = -(v - k[1, 2]) * z / k[1, 1]
            center = np.array([x, y, z], dtype=np.float64)
            self._model = ObjectVoxelModel(center, extent_m=0.22, voxel_m=0.005)
        self._world_transform = np.eye(4)
        self._frame_count = 0
        self._prev_cloud = None
        self._prev_rgbd = None
        self._o3d_intrinsic = None
        self._bounds_set = False
        self._coverage = None
        self._last_mask = None

    @property
    def coverage_map(self) -> np.ndarray | None:
        return self._coverage

    def accumulate(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray) -> None:
        h, w = depth_mm.shape[:2]
        if self._coverage is None or self._coverage.shape != (h, w):
            self._coverage = np.zeros((h, w), dtype=np.uint16)

        mask = self._frame_mask(depth_mm)
        self._last_mask = mask
        self._rgb_ref = rgb_bgr

        if mask is not None and np.any(mask):
            hit = mask & (depth_mm > 0)
            self._coverage[hit] = np.minimum(
                self._coverage[hit].astype(np.int32) + 1, 65535
            ).astype(np.uint16)

        super().accumulate(rgb_bgr, depth_mm)

    def render_on_rgb(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        show_mask_edge: bool = True,
    ) -> np.ndarray:
        """Composite live RGB with hole / coverage / reprojected voxels."""
        h, w = rgb_bgr.shape[:2]
        out = rgb_bgr.astype(np.float32)
        mask = self._last_mask
        if mask is None:
            mask = self._frame_mask(depth_mm)
        if mask is None:
            mask = np.zeros((h, w), dtype=bool)

        coverage = self._coverage
        if coverage is None or coverage.shape != (h, w):
            coverage = np.zeros((h, w), dtype=np.uint16)

        holes = mask & (depth_mm == 0)
        out[holes] = out[holes] * 0.25 + np.array([0.0, 0.0, 240.0]) * 0.75

        live = mask & (depth_mm > 0) & ~holes
        out[live] = out[live] * 0.55 + np.array([0.0, 160.0, 255.0]) * 0.45

        never = mask & (coverage == 0) & ~holes
        out[never] = out[never] * 0.45 + np.array([220.0, 0.0, 220.0]) * 0.55

        if self._model is not None and self._intrinsics is not None:
            cloud = self._model.to_point_cloud(min_weight=1)
            pts = np.asarray(cloud.points)
            if pts.size:
                u, v = project_points_to_uv(pts, self._intrinsics, width=w, height=h)
                if u.size:
                    out[v, u] = out[v, u] * 0.2 + np.array([255.0, 255.0, 0.0]) * 0.8
                    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        uu = np.clip(u + du, 0, w - 1)
                        vv = np.clip(v + dv, 0, h - 1)
                        out[vv, uu] = out[vv, uu] * 0.35 + np.array([255.0, 255.0, 0.0]) * 0.65

        if show_mask_edge:
            edge = cv2.Canny((mask.astype(np.uint8) * 255), 40, 120) > 0
            out[edge] = np.array([80.0, 255.0, 80.0])

        result = np.clip(out, 0, 255).astype(np.uint8)
        self._draw_hud(result, depth_mm, mask, coverage)
        return result

    def _draw_hud(
        self,
        bgr: np.ndarray,
        depth_mm: np.ndarray,
        mask: np.ndarray,
        coverage: np.ndarray,
    ) -> None:
        h, w = bgr.shape[:2]
        mask_px = int(mask.sum())
        holes = int((mask & (depth_mm == 0)).sum())
        never = int((mask & (coverage == 0)).sum())
        lines = [
            f"Voxels: {self.point_count}  frames: {self.frame_count}  (RGB-D odom)",
            f"Mask {mask_px}px | holes now {holes} | never seen {never}",
            "cyan=voxel  red=depth hole  orange=live depth  magenta=no hits yet",
            "Rotate object or move camera slowly  R=reset",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                bgr,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            y += 20

        leg_y = h - 28
        cv2.rectangle(bgr, (w - 200, leg_y - 8), (w - 8, h - 8), (0, 0, 0), -1)
        for i, (label, color) in enumerate(
            [
                ("hole", (0, 0, 240)),
                ("live", (0, 160, 255)),
                ("voxel", (0, 255, 255)),
            ]
        ):
            x0 = w - 190 + i * 62
            cv2.rectangle(bgr, (x0, leg_y), (x0 + 14, leg_y + 14), color, -1)
            cv2.putText(
                bgr,
                label,
                (x0 + 18, leg_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
