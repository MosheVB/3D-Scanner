"""Track scan object across turntable tilt/rotation after an initial mask seed.

Pipeline:
1. Seed from static 2D mask + depth (first capture frame or preview).
2. Build a camera-frame AABB and ORB keypoints inside the seed region.
3. Each frame: ORB match -> 2D hull; kinematics project AABB corners -> fallback hull.
4. Depth gate: keep pixels with z in [z_near, z_far] where z_far is the max projected
   corner depth at this (tilt, rotation) — masks everything farther than the object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OrbTrackResult:
    """ORB match result for one frame (visualization / debug)."""

    ok: bool
    matches: int = 0
    inliers: int = 0
    affine: np.ndarray | None = None
    match_pts: np.ndarray | None = None  # Nx2 in current frame
    box_pts: np.ndarray | None = None  # 4x2 int, rotated rect from warped mask
    bbox_xywh: tuple[int, int, int, int] | None = None


@dataclass
class ObjectTrackModel:
    """Serializable seed model for offline replay."""

    version: int = 1
    intrinsics: list[list[float]] = field(default_factory=list)
    pivot_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    half_extents_m: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    seed_tilt_deg: float = 0.0
    seed_rot_deg: float = 0.0
    z_margin_mm: float = 25.0
    orb_keypoints: list[list[float]] = field(default_factory=list)
    orb_descriptors: list[list[int]] = field(default_factory=list)
    seed_mask_png: str = "mask_track_seed.png"
    depth_band_mm: list[float] = field(default_factory=lambda: [0.0, 2500.0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectTrackModel:
        return cls(
            version=int(data.get("version", 1)),
            intrinsics=list(data.get("intrinsics") or []),
            pivot_m=list(data.get("pivot_m") or [0.0, 0.0, 0.0]),
            half_extents_m=list(data.get("half_extents_m") or [0.05, 0.05, 0.05]),
            seed_tilt_deg=float(data.get("seed_tilt_deg", 0.0)),
            seed_rot_deg=float(data.get("seed_rot_deg", 0.0)),
            z_margin_mm=float(data.get("z_margin_mm", 25.0)),
            orb_keypoints=list(data.get("orb_keypoints") or []),
            orb_descriptors=list(data.get("orb_descriptors") or []),
            seed_mask_png=str(data.get("seed_mask_png", "mask_track_seed.png")),
            depth_band_mm=list(data.get("depth_band_mm") or [0.0, 2500.0]),
        )


def _rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _unproject_mask(
    depth_mm: np.ndarray,
    include: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 2,
    max_depth_mm: float | None = None,
) -> np.ndarray:
    """Camera-frame points (meters) for masked valid depth."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_mm.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    uu, vv = np.meshgrid(xs, ys)
    z_mm = depth_mm[vv, uu]
    keep = include[vv, uu] & (z_mm > 0)
    if max_depth_mm is not None:
        keep &= z_mm <= max_depth_mm
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)
    z = z_mm[keep].astype(np.float64) / 1000.0
    u = uu[keep].astype(np.float64)
    v = vv[keep].astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _project_points(
    points_m: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv Nx2, z_mm N) for camera-frame points."""
    if points_m.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = points_m[:, 0], points_m[:, 1], points_m[:, 2]
    ok = z > 1e-6
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    u[ok] = fx * x[ok] / z[ok] + cx
    v[ok] = cy - fy * y[ok] / z[ok]
    return np.stack([u, v], axis=1), (z * 1000.0)


def _aabb_corners(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """8 corners of axis-aligned box centered at *center*."""
    offsets = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    return offsets * half[None, :] + center[None, :]


def _pose_points(
    local_pts: np.ndarray,
    pivot: np.ndarray,
    *,
    tilt_deg: float,
    rot_deg: float,
    seed_tilt_deg: float,
    seed_rot_deg: float,
) -> np.ndarray:
    """Transform object-local points to camera frame at (tilt, rot).

    Turntable model: spin about camera Y through pivot, then tilt about camera X.
    Angles are relative to the seed pose.
    """
    rel_tilt = tilt_deg - seed_tilt_deg
    rel_rot = rot_deg - seed_rot_deg
    r = _rot_x(rel_tilt) @ _rot_y(rel_rot)
    centered = local_pts - pivot
    return (r @ centered.T).T + pivot


def _hull_mask(
    uv: np.ndarray,
    shape: tuple[int, int],
    *,
    margin_px: int = 12,
) -> np.ndarray:
    h, w = shape
    if uv.shape[0] < 3:
        return np.zeros((h, w), dtype=bool)
    pts = uv.astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return np.zeros((h, w), dtype=bool)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        return np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    if margin_px > 0:
        k = margin_px | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


class ObjectTracker:
    """Per-frame object mask from seed + ORB + kinematic depth gate."""

    def __init__(self, model: ObjectTrackModel) -> None:
        self.model = model
        self._k = np.array(model.intrinsics, dtype=np.float64)
        self._pivot = np.array(model.pivot_m, dtype=np.float64)
        self._half = np.array(model.half_extents_m, dtype=np.float64)
        self._orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if model.orb_descriptors:
            self._seed_desc = np.array(model.orb_descriptors, dtype=np.uint8)
            self._seed_kp = np.array(model.orb_keypoints, dtype=np.float32)
        else:
            self._seed_desc = None
            self._seed_kp = None
        self._seed_mask_u8: np.ndarray | None = None

    def load_seed_mask(self, session_root: Path) -> None:
        path = session_root / self.model.seed_mask_png
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._seed_mask_u8 = img

    @classmethod
    def from_seed(
        cls,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        include: np.ndarray,
        intrinsics: np.ndarray | list[list[float]],
        *,
        seed_tilt_deg: float = 0.0,
        seed_rot_deg: float = 0.0,
        z_margin_mm: float = 25.0,
    ) -> ObjectTracker:
        k = np.array(intrinsics, dtype=np.float64)
        masked_depth = depth_mm[include & (depth_mm > 0)]
        if masked_depth.size < 80:
            raise ValueError("Too few masked depth pixels in seed.")
        med = float(np.median(masked_depth))
        fg_cut = float(min(np.percentile(masked_depth, 96), max(med + 120.0, 450.0)))
        seed_mask = include & (depth_mm > 0) & (depth_mm <= fg_cut)
        if int(seed_mask.sum()) < 80:
            raise ValueError("Too few foreground seed pixels after depth gate.")
        pts = _unproject_mask(depth_mm, seed_mask, k, stride=2)
        if pts.shape[0] < 80:
            raise ValueError(f"Too few seed points ({pts.shape[0]}); need masked depth.")

        lo = np.percentile(pts, 3, axis=0)
        hi = np.percentile(pts, 97, axis=0)
        center = 0.5 * (lo + hi)
        half = np.maximum(hi - center, 0.02) * 1.15
        # Pivot: bottom-center of AABB (table contact region)
        pivot = center.copy()
        pivot[1] = lo[1]  # lowest Y (toward table in camera frame)

        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        mask_u8 = (seed_mask.astype(np.uint8)) * 255
        orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)
        kp_pts, desc = ObjectTracker._detect_orb_in_mask(orb, gray, mask_u8)
        desc_list: list[list[int]] = []
        kp_list: list[list[float]] = []
        if desc is not None and len(kp_pts) >= 8:
            for i, pt in enumerate(kp_pts):
                kp_list.append([float(pt[0]), float(pt[1])])
                desc_list.append(desc[i].tolist())

        z_vals = pts[:, 2] * 1000.0
        depth_band = [float(np.percentile(z_vals, 2)), float(np.percentile(z_vals, 98))]

        model = ObjectTrackModel(
            intrinsics=k.tolist(),
            pivot_m=pivot.tolist(),
            half_extents_m=half.tolist(),
            seed_tilt_deg=seed_tilt_deg,
            seed_rot_deg=seed_rot_deg,
            z_margin_mm=z_margin_mm,
            orb_keypoints=kp_list,
            orb_descriptors=desc_list,
            depth_band_mm=depth_band,
        )
        tracker = cls(model)
        tracker._seed_mask_u8 = mask_u8.copy()
        return tracker

    @staticmethod
    def _detect_orb_in_mask(
        orb: cv2.ORB | None,
        gray: np.ndarray,
        mask_u8: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        detector = orb if orb is not None else cv2.ORB_create(nfeatures=800)
        kps, desc = detector.detectAndCompute(gray, mask_u8)
        if not kps or desc is None:
            return np.zeros((0, 2), dtype=np.float32), None
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts, desc

    def _kinematic_hull(
        self,
        shape: tuple[int, int],
        *,
        tilt_deg: float,
        rot_deg: float,
    ) -> tuple[np.ndarray, float, float]:
        """Projected AABB hull and conservative depth bounds (mm)."""
        local_corners = _aabb_corners(np.zeros(3), self._half)
        world = _pose_points(
            local_corners + self._pivot,
            self._pivot,
            tilt_deg=tilt_deg,
            rot_deg=rot_deg,
            seed_tilt_deg=self.model.seed_tilt_deg,
            seed_rot_deg=self.model.seed_rot_deg,
        )
        uv, z_mm = _project_points(world, self._k)
        ok = (z_mm > 0) & np.isfinite(uv).all(axis=1)
        uv = uv[ok]
        z_mm = z_mm[ok]
        if uv.shape[0] < 3:
            return np.zeros(shape, dtype=bool), 0.0, 2500.0
        hull = _hull_mask(uv, shape, margin_px=14)
        z_near = float(np.min(z_mm)) - self.model.z_margin_mm
        z_far = float(np.max(z_mm)) + self.model.z_margin_mm
        return hull, max(0.0, z_near), z_far

    def _orb_match(
        self,
        rgb_bgr: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int]:
        """Return (affine 2x3, seed_pts, cur_pts, match_count)."""
        empty = np.zeros((0, 2), dtype=np.float32)
        if self._seed_desc is None or self._seed_kp is None or len(self._seed_kp) < 8:
            return None, empty, empty, 0
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 8:
            return None, empty, empty, 0
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        matches = self._matcher.knnMatch(self._seed_desc, desc, k=2)
        seed_pts: list[np.ndarray] = []
        cur_pts: list[np.ndarray] = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                seed_pts.append(self._seed_kp[m.queryIdx])
                cur_pts.append(pts[m.trainIdx])
        if len(seed_pts) < 6:
            return None, empty, empty, len(seed_pts)
        seed_arr = np.array(seed_pts, dtype=np.float32)
        cur_arr = np.array(cur_pts, dtype=np.float32)
        M, inliers = cv2.estimateAffinePartial2D(
            seed_arr,
            cur_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        n_in = int(inliers.sum()) if inliers is not None else len(seed_arr)
        if M is None:
            return None, seed_arr, cur_arr, n_in
        return M, seed_arr, cur_arr, n_in

    def _box_from_affine(
        self,
        M: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h, w = shape
        if self._seed_mask_u8 is not None:
            warped = cv2.warpAffine(
                self._seed_mask_u8,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            contours, _ = cv2.findContours(
                warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 200:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect).astype(np.int32)
                    x, y, bw, bh = cv2.boundingRect(box)
                    return box, (int(x), int(y), int(bw), int(bh))
        if self._seed_kp is None or len(self._seed_kp) < 3:
            return None, None
        ones = np.ones((len(self._seed_kp), 1), dtype=np.float32)
        warped = (M @ np.hstack([self._seed_kp, ones]).T).T
        hull = cv2.convexHull(warped.astype(np.float32))
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(box)
        return box, (int(x), int(y), int(bw), int(bh))

    def track_orb(self, rgb_bgr: np.ndarray) -> OrbTrackResult:
        """ORB-only track: affine warp of seed mask -> rotated bounding box."""
        h, w = rgb_bgr.shape[:2]
        M, seed_arr, cur_arr, n_in = self._orb_match(rgb_bgr)
        if M is None:
            return OrbTrackResult(
                ok=False,
                matches=len(cur_arr),
                inliers=n_in,
                match_pts=cur_arr if len(cur_arr) else None,
            )
        box_pts, bbox = self._box_from_affine(M, (h, w))
        return OrbTrackResult(
            ok=box_pts is not None,
            matches=len(cur_arr),
            inliers=n_in,
            affine=M,
            match_pts=cur_arr,
            box_pts=box_pts,
            bbox_xywh=bbox,
        )

    def seed_box_pts(self) -> np.ndarray | None:
        """Rotated box from seed mask (for pre-track frames)."""
        if self._seed_mask_u8 is None:
            return None
        contours, _ = cv2.findContours(
            self._seed_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        return cv2.boxPoints(rect).astype(np.int32)

    def _orb_mask(
        self,
        rgb_bgr: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        M, _seed, _cur, _n = self._orb_match(rgb_bgr)
        if M is None:
            return None
        h, w = shape
        if self._seed_mask_u8 is not None:
            warped = cv2.warpAffine(
                self._seed_mask_u8,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            mask = warped > 0
            if int(mask.sum()) > 200:
                return mask
        if self._seed_kp is None:
            return None
        ones = np.ones((len(self._seed_kp), 1), dtype=np.float32)
        warped_pts = (M @ np.hstack([self._seed_kp, ones]).T).T
        return _hull_mask(warped_pts, shape, margin_px=16)

    def include_mask(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        tilt_deg: float,
        rot_deg: float,
    ) -> np.ndarray:
        h, w = depth_mm.shape[:2]
        kin_hull, z_near, z_far = self._kinematic_hull((h, w), tilt_deg=tilt_deg, rot_deg=rot_deg)
        orb_mask = self._orb_mask(rgb_bgr, (h, w))
        if orb_mask is not None:
            hull = orb_mask | kin_hull
        else:
            hull = kin_hull

        valid = (depth_mm > 0) & (depth_mm >= z_near) & (depth_mm <= z_far)
        return hull & valid

    def mask_depth(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        tilt_deg: float,
        rot_deg: float,
    ) -> np.ndarray:
        include = self.include_mask(rgb_bgr, depth_mm, tilt_deg=tilt_deg, rot_deg=rot_deg)
        out = np.zeros_like(depth_mm)
        out[include] = depth_mm[include]
        return out


def save_object_track_model(
    path: Path, model: ObjectTrackModel, *, seed_mask_u8: np.ndarray | None = None
) -> None:
    if seed_mask_u8 is not None:
        cv2.imwrite(str(path.parent / model.seed_mask_png), seed_mask_u8)
    path.write_text(
        json.dumps(model.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_object_track_model(path: Path) -> ObjectTrackModel | None:
    if not path.is_file():
        return None
    model = ObjectTrackModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return model


def load_object_tracker(session_root: Path) -> ObjectTracker | None:
    model = load_object_track_model(session_root / "object_track.json")
    if model is None:
        return None
    tracker = ObjectTracker(model)
    tracker.load_seed_mask(session_root)
    return tracker


def render_orb_tracking(
    rgb_bgr: np.ndarray,
    track: OrbTrackResult,
    *,
    label: str = "",
    draw_points: bool = True,
) -> np.ndarray:
    """Draw ORB track box and match points on clean RGB."""
    out = rgb_bgr.copy()
    color_box = (0, 220, 0)
    color_pts = (0, 200, 255)

    if track.box_pts is not None:
        cv2.polylines(
            out,
            [track.box_pts.reshape(-1, 1, 2)],
            isClosed=True,
            color=color_box,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    elif track.bbox_xywh is not None:
        x, y, bw, bh = track.bbox_xywh
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color_box, 2, cv2.LINE_AA)

    if draw_points and track.match_pts is not None and len(track.match_pts):
        for pt in track.match_pts:
            cv2.circle(
                out,
                (int(pt[0]), int(pt[1])),
                3,
                color_pts,
                -1,
                lineType=cv2.LINE_AA,
            )

    hud = label
    if not hud:
        status = "ok" if track.ok else "lost"
        hud = f"ORB {status}  matches {track.matches}  inliers {track.inliers}"
    cv2.putText(
        out,
        hud,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        hud,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_track_overlay(
    rgb_bgr: np.ndarray,
    include: np.ndarray,
    *,
    label: str = "",
) -> np.ndarray:
    overlay = rgb_bgr.copy()
    overlay[include] = (
        overlay[include].astype(np.float32) * 0.35
        + np.array([40, 220, 80], dtype=np.float32) * 0.65
    ).astype(np.uint8)
    overlay[~include] = (
        overlay[~include].astype(np.float32) * 0.5
        + np.array([40, 40, 200], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    if label:
        cv2.putText(
            overlay,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay
