"""Click-based scan region mask (include + exclude RGB clicks) for web turntable scans."""



from __future__ import annotations



import threading

from dataclasses import dataclass

from typing import Any, Literal, Sequence



import cv2

import numpy as np



from scanner.object_mask import object_mask_from_click

from scanner.mask_haiku import HaikuShapeAssist, haiku_available, haiku_unavailable_reason
from scanner.scan_bounds import (
    BoundsTracker,
    adaptive_depth_band_mm,
    build_adaptive_shape_mask,
    draw_bounds_overlay,
    sample_depth_at_click,
)



MAX_MASK_CLICKS = 16

ClickMode = Literal["include", "exclude"]





@dataclass

class _ClickPoint:

    u: int

    v: int

    depth_mm: float | None = None

    mode: ClickMode = "include"





def draw_click_markers(

    bgr: np.ndarray,

    include_clicks: Sequence[tuple[int, int]],

    *,

    exclude_clicks: Sequence[tuple[int, int]] | None = None,

    colors: Sequence[tuple[int, int, int]] | None = None,

    bounds_tracker: BoundsTracker | None = None,

    intrinsics: Sequence[Sequence[float]] | None = None,

) -> np.ndarray:

    """Burn crosshair markers, assist contour, and 3D cube overlay into a BGR frame copy."""

    if bgr is None or getattr(bgr, "size", 0) == 0:

        return bgr

    try:

        out = bgr.copy()

        h, w = out.shape[:2]

    except Exception:

        return bgr

    include_palette = [

        (0, 255, 120),

        (0, 230, 140),

        (40, 255, 160),

        (80, 255, 180),

        (120, 255, 200),

        (160, 255, 220),

    ]

    exclude_color = (0, 80, 255)

    for i, pt in enumerate(include_clicks):

        if not isinstance(pt, (tuple, list)) or len(pt) < 2:

            continue

        try:

            u, v = int(pt[0]), int(pt[1])

        except (TypeError, ValueError):

            continue

        palette = include_palette if colors is None else colors

        color = palette[i % len(palette)] if palette else include_palette[0]

        u = int(np.clip(u, 0, w - 1))

        v = int(np.clip(v, 0, h - 1))

        r = 14

        cv2.circle(out, (u, v), r, color, 2, cv2.LINE_AA)

        cv2.line(out, (u - r - 6, v), (u + r + 6, v), color, 2, cv2.LINE_AA)

        cv2.line(out, (u, v - r - 6), (u, v + r + 6), color, 2, cv2.LINE_AA)

        cv2.putText(

            out,

            str(i + 1),

            (u + 14, v - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            1,

            cv2.LINE_AA,

        )

    for pt in exclude_clicks or ():

        if not isinstance(pt, (tuple, list)) or len(pt) < 2:

            continue

        try:

            u, v = int(pt[0]), int(pt[1])

        except (TypeError, ValueError):

            continue

        u = int(np.clip(u, 0, w - 1))

        v = int(np.clip(v, 0, h - 1))

        r = 12

        cv2.circle(out, (u, v), r, exclude_color, 2, cv2.LINE_AA)

        cv2.line(out, (u - r - 4, v), (u + r + 4, v), exclude_color, 2, cv2.LINE_AA)

        cv2.line(out, (u, v - r - 4), (u, v + r + 4), exclude_color, 2, cv2.LINE_AA)

        cv2.putText(

            out,

            "X",

            (u + 12, v + 5),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.4,

            exclude_color,

            1,

            cv2.LINE_AA,

        )

    if bounds_tracker is not None and intrinsics is not None:

        try:

            out = draw_bounds_overlay(

                out,

                bounds_tracker.bounds,

                intrinsics,

                assist_mask=bounds_tracker.assist_mask(),

                expand_flash=bounds_tracker.expand_flash_active,

                phase=bounds_tracker.phase,

                image_rect_uv=bounds_tracker.image_rect_uv,

                contour_uv=bounds_tracker.contour_points,

                obb_corners=bounds_tracker.obb_corners,

                ellipse_params=bounds_tracker.ellipse_params,

            )

        except Exception:

            pass

    return out





class ScanMaskState:

    """RGB include clicks (1–16) seed/refine mask; exclude clicks subtract depth blobs."""



    def __init__(self) -> None:

        self._lock = threading.Lock()

        self._clicks: list[_ClickPoint] = []

        self._tracker = BoundsTracker()

        self._intrinsics: Sequence[Sequence[float]] | None = None

        self._haiku = HaikuShapeAssist()

        self._use_haiku = False

        self._last_rgb: np.ndarray | None = None

        self._pending_depth_reprocess = False



    @property

    def needs_camera_depth(self) -> bool:

        with self._lock:

            return self._pending_depth_reprocess



    @property

    def tracker(self) -> BoundsTracker:

        return self._tracker



    @property

    def click_count(self) -> int:

        with self._lock:

            return sum(1 for c in self._clicks if c.mode == "include")



    @property

    def exclude_count(self) -> int:

        with self._lock:

            return sum(1 for c in self._clicks if c.mode == "exclude")



    @property

    def ready(self) -> bool:

        with self._lock:

            return any(c.mode == "include" for c in self._clicks)



    @property

    def focus_mm(self) -> float | None:

        with self._lock:

            return self._focus_mm_unlocked()



    def include_uvs(self) -> list[tuple[int, int]]:

        with self._lock:

            return [(c.u, c.v) for c in self._clicks if c.mode == "include"]



    def exclude_uvs(self) -> list[tuple[int, int]]:

        with self._lock:

            return [(c.u, c.v) for c in self._clicks if c.mode == "exclude"]



    def click_uvs(self) -> list[tuple[int, int]]:

        """Primary include clicks (backward compatible)."""

        return self.include_uvs()



    def clear(self) -> None:

        with self._lock:

            self._clicks.clear()

        self._tracker.clear()

        self._haiku.clear_cache()

        self._use_haiku = False

        self._last_rgb = None



    def set_intrinsics(self, intrinsics: Sequence[Sequence[float]] | None) -> None:

        with self._lock:

            self._intrinsics = intrinsics



    @staticmethod

    def _parse_points(

        raw: Sequence[Sequence[int | float]],

        mode: ClickMode,

        depth_mm: np.ndarray | None,

    ) -> list[_ClickPoint]:

        out: list[_ClickPoint] = []

        for pt in raw:

            if len(pt) < 2:

                raise ValueError("Each point must be [u, v]")

            u, v = int(round(float(pt[0]))), int(round(float(pt[1])))

            d = sample_depth_at_click(depth_mm, u, v) if depth_mm is not None else None

            out.append(_ClickPoint(u=u, v=v, depth_mm=d, mode=mode))

        return out



    def set_clicks(

        self,

        points: Sequence[Sequence[int | float]] | None = None,

        depth_mm: np.ndarray | None = None,

        intrinsics: Sequence[Sequence[float]] | None = None,

        rgb_bgr: np.ndarray | None = None,

        *,

        include: Sequence[Sequence[int | float]] | None = None,

        exclude: Sequence[Sequence[int | float]] | None = None,

        mode: ClickMode | None = None,

        use_haiku: bool | None = None,

    ) -> None:

        """Set include/exclude clicks. Legacy ``points`` + optional ``mode`` still supported."""

        if include is None and points is not None:

            include = points

        include_pts = list(include or [])

        exclude_pts = list(exclude or [])



        if mode == "exclude" and points is not None and include is None:

            exclude_pts = list(points)

            include_pts = [

                (c.u, c.v) for c in self._clicks if c.mode == "include"

            ]



        if not include_pts and not exclude_pts:

            raise ValueError("Provide at least one include or exclude click")

        if len(include_pts) > MAX_MASK_CLICKS:

            raise ValueError(f"At most {MAX_MASK_CLICKS} include clicks")

        if len(exclude_pts) > MAX_MASK_CLICKS:

            raise ValueError(f"At most {MAX_MASK_CLICKS} exclude clicks")

        if not include_pts and exclude_pts:

            raise ValueError("Need at least one include click before exclude")



        include_new = self._parse_points(include_pts, "include", depth_mm)

        exclude_new = self._parse_points(exclude_pts, "exclude", depth_mm)

        uvs = [(c.u, c.v) for c in include_new]



        with self._lock:

            prev_include = [(c.u, c.v) for c in self._clicks if c.mode == "include"]

            prev_count = len(prev_include)

            self._clicks = include_new + exclude_new

            if intrinsics is not None:

                self._intrinsics = intrinsics

            if rgb_bgr is not None:

                self._last_rgb = rgb_bgr

            if use_haiku is not None:

                self._use_haiku = bool(use_haiku)

            elif not prev_include and include_new and haiku_available():

                self._use_haiku = True

            use_haiku_flag = self._use_haiku



        if depth_mm is None or intrinsics is None:

            self._pending_depth_reprocess = True

            return

        self._pending_depth_reprocess = False



        if len(uvs) == 1:

            self._tracker.initialize_assist_detect(uvs[0], rgb_bgr, depth_mm, intrinsics)

        elif len(uvs) > prev_count:

            if prev_count == 0:

                self._tracker.initialize_assist_detect(uvs[0], rgb_bgr, depth_mm, intrinsics)

                extra = uvs[1:]

            else:

                extra = uvs[prev_count:]

            for uv in extra:

                self._tracker.handle_additional_click(uv, depth_mm, intrinsics, rgb_bgr)

        elif uvs != prev_include:

            self._tracker.initialize_assist_detect(uvs[0], rgb_bgr, depth_mm, intrinsics)

            for uv in uvs[1:]:

                self._tracker.handle_additional_click(uv, depth_mm, intrinsics, rgb_bgr)



        exclude_uvs = [(c.u, c.v) for c in exclude_new]

        self._tracker.set_exclude_clicks(exclude_uvs, depth_mm, intrinsics)

        self._refresh_adaptive_shape(rgb_bgr, uvs, exclude_uvs, depth_mm, intrinsics)

        if use_haiku_flag and haiku_available() and rgb_bgr is not None:

            self._maybe_request_haiku(rgb_bgr, uvs, exclude_uvs, depth_mm, intrinsics)



    def update_frame(

        self,

        rgb_bgr: np.ndarray,

        depth_mm: np.ndarray,

        intrinsics: Sequence[Sequence[float]],

    ) -> None:

        if not self.ready:

            return

        with self._lock:

            self._intrinsics = intrinsics

        self._tracker.update(rgb_bgr, depth_mm, intrinsics)

        if self.exclude_uvs():

            self._tracker.set_exclude_clicks(self.exclude_uvs(), depth_mm)



    def build_mask(self, depth_mm: np.ndarray) -> np.ndarray | None:

        with self._lock:

            include_clicks = [c for c in self._clicks if c.mode == "include"]

            intrinsics = self._intrinsics



        if not include_clicks:

            return None



        tracked = self._tracker.build_mask(depth_mm)

        if tracked is not None:

            return tracked



        seed = include_clicks[0]

        focus = seed.depth_mm

        if focus is None:

            focus = sample_depth_at_click(depth_mm, seed.u, seed.v)

        if focus is None:

            return None



        mask = object_mask_from_click(depth_mm, seed.u, seed.v, center_mm=focus)

        exclude_uvs = self.exclude_uvs()

        if exclude_uvs and np.any(mask):

            for u, v in exclude_uvs:

                d = sample_depth_at_click(depth_mm, u, v)

                if d is None:

                    continue

                ex = object_mask_from_click(

                    depth_mm,

                    u,

                    v,

                    center_mm=d,

                    band_mm=adaptive_depth_band_mm(d),

                )

                mask = mask & ~ex

        return mask if np.any(mask) else None



    def patch_settings(self, updates: dict[str, Any]) -> dict[str, Any]:

        return self._tracker.patch_settings(updates)



    def set_use_haiku(self, enabled: bool) -> None:

        with self._lock:

            self._use_haiku = bool(enabled)



    def request_haiku_assist(

        self,

        depth_mm: np.ndarray | None,

        intrinsics: Sequence[Sequence[float]] | None,

        rgb_bgr: np.ndarray | None = None,

    ) -> bool:

        """Re-run Haiku vision on demand. Returns True if request queued."""

        with self._lock:

            include = [(c.u, c.v) for c in self._clicks if c.mode == "include"]

            exclude = [(c.u, c.v) for c in self._clicks if c.mode == "exclude"]

            rgb = rgb_bgr or self._last_rgb

            intr = intrinsics or self._intrinsics

        if not include or rgb is None or depth_mm is None or intr is None:

            return False

        if not haiku_available():

            return False

        self._use_haiku = True

        return self._maybe_request_haiku(rgb, include, exclude, depth_mm, intr, force=True)



    def _refresh_adaptive_shape(

        self,

        rgb_bgr: np.ndarray | None,

        include_uvs: list[tuple[int, int]],

        exclude_uvs: list[tuple[int, int]],

        depth_mm: np.ndarray,

        intrinsics: Sequence[Sequence[float]],

    ) -> None:

        if not include_uvs:

            return

        settings = self._tracker.settings_dict()

        from scanner.scan_bounds import ScanBoundsSettings

        s = ScanBoundsSettings(

            margin_mm=float(settings.get("margin_mm", 15)),

            expand_margin_mm=float(settings.get("expand_margin_mm", 20)),

            edge_canny_low=40,

            edge_canny_high=120,

        )

        ex_mask = self._tracker._combined_exclude_mask(depth_mm) if exclude_uvs else None

        mask, contour, shape_src = build_adaptive_shape_mask(

            rgb_bgr,

            depth_mm,

            include_uvs,

            intrinsics,

            s,

            exclude_mask=ex_mask if exclude_uvs else None,

        )

        if np.any(mask):

            self._tracker.apply_shape_mask(mask, contour, shape_src, depth_mm, intrinsics)



    def _maybe_request_haiku(

        self,

        rgb_bgr: np.ndarray,

        include_uvs: list[tuple[int, int]],

        exclude_uvs: list[tuple[int, int]],

        depth_mm: np.ndarray,

        intrinsics: Sequence[Sequence[float]],

        *,

        force: bool = False,

    ) -> bool:

        cached = self._haiku.cached_mask(include_uvs, exclude_uvs)

        if cached is not None and not force:

            contour = self._haiku.cached_contour(include_uvs, exclude_uvs)

            self._tracker.apply_shape_mask(

                cached, contour, "haiku", depth_mm, intrinsics

            )

            self._haiku.mark_used(len(contour))

            return False

        self._tracker.set_haiku_pending(True)

        depth_ref = depth_mm

        intr_ref = intrinsics



        def _on_result(mask: np.ndarray | None, contour: list[tuple[int, int]]) -> None:

            if mask is not None and np.any(mask):

                self._tracker.apply_shape_mask(

                    mask, contour, "haiku", depth_ref, intr_ref

                )

                self._haiku.mark_used(len(contour))

            else:

                self._tracker.set_haiku_pending(False)

                self._haiku.mark_fallback(self._haiku.status_msg)

                self._refresh_adaptive_shape(

                    rgb_bgr, include_uvs, exclude_uvs, depth_ref, intr_ref

                )



        return self._haiku.request_shape(

            rgb_bgr,

            include_uvs,

            exclude_uvs,

            on_result=_on_result,

        )



    def _focus_mm_unlocked(self) -> float | None:

        include = [c for c in self._clicks if c.mode == "include"]

        if not include:

            return None

        depths = [c.depth_mm for c in include if c.depth_mm]

        if depths:

            return float(np.median(depths))

        cube = self._tracker.bounds

        if cube is not None:

            return float(cube.center_m[2] * 1000.0)

        return None



    def _mode_unlocked(self) -> str:

        include = [c for c in self._clicks if c.mode == "include"]

        exclude = [c for c in self._clicks if c.mode == "exclude"]

        if not include:

            return "none"

        if exclude and self._tracker.status_dict().get("last_click_mode") == "exclude":

            return "exclude"

        if len(include) == 1:

            return "assist_detect"

        last = self._tracker.status_dict().get("last_click_mode", "expand_to_include")

        if last in ("refine_inside", "expand_to_include", "no_depth", "exclude"):

            return last

        return "expand_to_include"



    def status_dict(self) -> dict[str, Any]:

        with self._lock:

            clicks = [

                {

                    "u": c.u,

                    "v": c.v,

                    "depth_mm": c.depth_mm,

                    "mode": c.mode,

                }

                for c in self._clicks

            ]

            include_count = sum(1 for c in self._clicks if c.mode == "include")

            exclude_count = sum(1 for c in self._clicks if c.mode == "exclude")

            mode = self._mode_unlocked()

            focus = self._focus_mm_unlocked()

            base = {

                "click_count": include_count,

                "exclude_count": exclude_count,

                "clicks": clicks,

                "include_clicks": [

                    {"u": c.u, "v": c.v, "depth_mm": c.depth_mm}

                    for c in self._clicks

                    if c.mode == "include"

                ],

                "exclude_clicks": [

                    {"u": c.u, "v": c.v, "depth_mm": c.depth_mm}

                    for c in self._clicks

                    if c.mode == "exclude"

                ],

                "mode": mode,

                "ready": include_count >= 1,

                "focus_mm": focus,

                "bounds": None,

                "max_clicks": MAX_MASK_CLICKS,

                "help": (

                    f"Include clicks 1–{MAX_MASK_CLICKS} (green): seed object; yellow contour adapts to "

                    "shape (depth + edges). Optional AI shape (Haiku) when API key is set. "

                    "Exclude clicks (red X) subtract regions and tighten the cube. Clear resets all."

                ),

                "use_haiku": self._use_haiku,

                "haiku_available": haiku_available(),

                "haiku_unavailable_reason": haiku_unavailable_reason(),

                "haiku_busy": self._haiku.busy,

                "haiku_status": self._haiku.status_msg,

                "needs_camera_depth": self._pending_depth_reprocess,

            }

        base.update(self._tracker.status_dict())

        base.update(self._haiku.status_dict())

        return base


