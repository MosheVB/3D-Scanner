"""Optional Claude Haiku vision assist for scan-region shape (isolated from stream thread)."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence

import cv2
import numpy as np

from scanner.env_secrets import ensure_anthropic_api_key

ensure_anthropic_api_key()

try:
    import anthropic as _anthropic

    _HAVE_ANTHROPIC = True
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    _HAVE_ANTHROPIC = False

HAIKU_MODEL = "claude-haiku-4-5"
_IMG_W = 480
_IMG_H = 360
_JPEG_Q = 72
_MIN_INTERVAL_S = 1.5

ShapeSource = Literal["haiku", "depth", "clicks"]
HaikuStatus = Literal["idle", "pending", "success", "error"]

_SYSTEM = (
    "You segment the object on a turntable for a 3D scanner.\n"
    "Reply with ONLY a JSON object (no markdown):\n"
    '{"polygon_uv":[[u,v],...]} OR {"bbox_uv":[x0,y0,x1,y1]}\n\n'
    "Coordinates are image pixels in the SMALL preview image: u=x column, v=y row, "
    "origin top-left.\n"
    "Include the full object on the turntable; exclude table and background.\n"
    "Green numbered markers are INCLUDE clicks (object). Red X markers are EXCLUDE "
    "(table/background to remove). Respect exclude regions as negative.\n"
    "Prefer a tight polygon around the visible object silhouette."
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _haiku_log(message: str) -> None:
    try:
        print(f"[Haiku] {message}", flush=True)
    except Exception:
        pass


def haiku_available() -> bool:
    """True when anthropic is installed and ANTHROPIC_API_KEY is set."""
    ensure_anthropic_api_key()
    return _HAVE_ANTHROPIC and bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def haiku_unavailable_reason() -> str | None:
    """Human-readable reason Haiku is off, or None when available."""
    ensure_anthropic_api_key()
    if not _HAVE_ANTHROPIC:
        return "pip install anthropic"
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "set ANTHROPIC_API_KEY (User env) and restart web server"
    return None


def _clicks_key(
    include_uvs: Sequence[tuple[int, int]],
    exclude_uvs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(include_uvs) + [(-1, uv[0], uv[1]) for uv in sorted(exclude_uvs)])


def parse_haiku_shape_json(text: str) -> dict[str, Any] | None:
    """Parse Haiku response into polygon_uv or bbox_uv."""
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if m is None:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if "polygon_uv" in data:
        poly = data["polygon_uv"]
        if isinstance(poly, list) and len(poly) >= 3:
            return {"polygon_uv": poly}
    if "bbox_uv" in data:
        bbox = data["bbox_uv"]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return {"bbox_uv": [float(b) for b in bbox]}
    return None


def polygon_uv_to_full(
    polygon_uv: Sequence[Sequence[float | int]],
    *,
    scale_u: float,
    scale_v: float,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Scale preview-space polygon_uv to full-frame (u, v) for overlay."""
    out: list[tuple[int, int]] = []
    for pt in polygon_uv:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        u = int(round(float(pt[0]) / scale_u))
        v = int(round(float(pt[1]) / scale_v))
        out.append((int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))))
    return out


def rasterize_shape(
    shape: dict[str, Any],
    h: int,
    w: int,
    *,
    scale_u: float = 1.0,
    scale_v: float = 1.0,
) -> np.ndarray:
    """Rasterize polygon_uv or bbox_uv to a boolean mask (coords may be in scaled space)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if "polygon_uv" in shape:
        pts: list[list[float]] = []
        for pt in shape["polygon_uv"]:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            u = float(pt[0]) / scale_u
            v = float(pt[1]) / scale_v
            pts.append([u, v])
        if len(pts) < 3:
            return mask.astype(bool)
        arr = np.array(pts, dtype=np.float32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [arr.astype(np.int32)], 1)
    elif "bbox_uv" in shape:
        x0, y0, x1, y1 = shape["bbox_uv"]
        u0 = int(np.clip(min(x0, x1) / scale_u, 0, w - 1))
        v0 = int(np.clip(min(y0, y1) / scale_v, 0, h - 1))
        u1 = int(np.clip(max(x0, x1) / scale_u, 0, w - 1))
        v1 = int(np.clip(max(y0, y1) / scale_v, 0, h - 1))
        mask[v0 : v1 + 1, u0 : u1 + 1] = 1
    return mask.astype(bool)


def contour_points_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Largest external contour as list of (u, v) points."""
    if mask is None or not np.any(mask):
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return [(int(p[0][0]), int(p[0][1])) for p in largest]


class HaikuShapeAssist:
    """Async Haiku vision on click submit; cache until clicks change."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: Any = None
        self._api_in_flight = False
        self._last_call = 0.0
        self._status: HaikuStatus = "idle"
        self._message = "idle"
        self._last_called_at: str | None = None
        self._last_completed_at: str | None = None
        self._last_used_at: str | None = None
        self._cached_key: tuple[tuple[int, int], ...] | None = None
        self._cached_mask: np.ndarray | None = None
        self._cached_contour: list[tuple[int, int]] = []
        self._on_result: Callable[[np.ndarray | None, list[tuple[int, int]]], None] | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._api_in_flight

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._api_in_flight

    @property
    def status_msg(self) -> str:
        with self._lock:
            return self._message

    def mark_used(self, vertex_count: int) -> None:
        now = _iso_now()
        with self._lock:
            self._last_used_at = now
            self._status = "success"
            self._message = f"Using Haiku shape (polygon, {vertex_count} pts)"
        _haiku_log(self._message)

    def mark_fallback(self, reason: str) -> None:
        with self._lock:
            self._status = "error"
            self._message = f"Using depth fallback — {reason}"
        _haiku_log(self._message)

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "haiku_last_called_at": self._last_called_at,
                "haiku_last_completed_at": self._last_completed_at,
                "haiku_last_used_at": self._last_used_at,
                "haiku_status": self._status,
                "haiku_message": self._message,
                "haiku_pending": self._api_in_flight,
                "haiku_busy": self._api_in_flight,
            }

    def cached_mask(
        self,
        include_uvs: Sequence[tuple[int, int]],
        exclude_uvs: Sequence[tuple[int, int]],
    ) -> np.ndarray | None:
        key = _clicks_key(include_uvs, exclude_uvs)
        with self._lock:
            if self._cached_key == key and self._cached_mask is not None:
                return self._cached_mask.copy()
        return None

    def cached_contour(
        self,
        include_uvs: Sequence[tuple[int, int]],
        exclude_uvs: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        key = _clicks_key(include_uvs, exclude_uvs)
        with self._lock:
            if self._cached_key == key:
                return list(self._cached_contour)
        return []

    def clear_cache(self) -> None:
        with self._lock:
            self._cached_key = None
            self._cached_mask = None
            self._cached_contour = []
            self._status = "idle"
            self._message = "idle"
            self._last_called_at = None
            self._last_completed_at = None
            self._last_used_at = None

    def request_shape(
        self,
        rgb_bgr: np.ndarray,
        include_uvs: Sequence[tuple[int, int]],
        exclude_uvs: Sequence[tuple[int, int]],
        *,
        on_result: Callable[[np.ndarray | None, list[tuple[int, int]]], None] | None = None,
    ) -> bool:
        """Queue async Haiku call. Returns True if a new request started."""
        if not haiku_available():
            return False
        key = _clicks_key(include_uvs, exclude_uvs)
        with self._lock:
            if self._cached_key == key and self._cached_mask is not None:
                if on_result is not None:
                    on_result(self._cached_mask.copy(), list(self._cached_contour))
                return False
            if self._api_in_flight:
                return False
            now = time.time()
            if now - self._last_call < _MIN_INTERVAL_S:
                return False
            self._api_in_flight = True
            self._last_call = now
            called_at = _iso_now()
            self._last_called_at = called_at
            self._status = "pending"
            self._message = f"Calling Haiku {called_at[11:19]}…"
            self._on_result = on_result
        _haiku_log(self._message)

        h, w = rgb_bgr.shape[:2]
        scale_u = _IMG_W / max(w, 1)
        scale_v = _IMG_H / max(h, 1)
        small = cv2.resize(rgb_bgr, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
        for i, (u, v) in enumerate(include_uvs):
            cu, cv_ = int(u * scale_u), int(v * scale_v)
            cv2.circle(small, (cu, cv_), 6, (0, 255, 120), 2)
            cv2.putText(
                small, str(i + 1), (cu + 8, cv_ - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1,
            )
        for u, v in exclude_uvs:
            cu, cv_ = int(u * scale_u), int(v * scale_v)
            cv2.circle(small, (cu, cv_), 5, (0, 80, 255), 2)
            cv2.putText(
                small, "X", (cu + 6, cv_ + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1,
            )

        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_Q])
        if not ok:
            with self._lock:
                self._api_in_flight = False
                self._status = "error"
                self._message = "Haiku failed: image encode"
            _haiku_log(self._message)
            return False

        img_b64 = base64.b64encode(buf.tobytes()).decode()
        inc_str = ", ".join(f"({u},{v})" for u, v in include_uvs)
        exc_str = ", ".join(f"({u},{v})" for u, v in exclude_uvs) or "none"
        user_text = (
            f"Preview image is {_IMG_W}×{_IMG_H} px (full frame is {w}×{h}). "
            f"Include clicks (green): {inc_str}. "
            f"Exclude clicks (red X): {exc_str}. "
            "Return polygon_uv or bbox_uv in preview pixel coordinates."
        )

        threading.Thread(
            target=self._api_worker,
            args=(img_b64, user_text, key, h, w, scale_u, scale_v),
            daemon=True,
            name="haiku-shape",
        ).start()
        return True

    def _api_worker(
        self,
        img_b64: str,
        user_text: str,
        key: tuple[tuple[int, int], ...],
        h: int,
        w: int,
        scale_u: float,
        scale_v: float,
    ) -> None:
        mask: np.ndarray | None = None
        contour: list[tuple[int, int]] = []
        err_msg = ""
        try:
            if self._client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
                self._client = _anthropic.Anthropic(api_key=api_key)
            resp = self._client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=512,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
            text = resp.content[0].text.strip() if resp.content else ""
            shape = parse_haiku_shape_json(text)
            if shape is not None:
                raw = rasterize_shape(shape, h, w, scale_u=scale_u, scale_v=scale_v)
                if np.any(raw):
                    mask = raw
                    if "polygon_uv" in shape:
                        contour = polygon_uv_to_full(
                            shape["polygon_uv"],
                            scale_u=scale_u,
                            scale_v=scale_v,
                            h=h,
                            w=w,
                        )
                    if len(contour) < 3:
                        contour = contour_points_from_mask(mask)
                else:
                    err_msg = "empty mask"
            else:
                err_msg = "parse error"
        except Exception as exc:
            err_msg = str(exc)

        completed_at = _iso_now()
        callback: Callable[[np.ndarray | None, list[tuple[int, int]]], None] | None
        with self._lock:
            self._api_in_flight = False
            self._last_completed_at = completed_at
            if mask is not None and np.any(mask):
                self._cached_key = key
                self._cached_mask = mask
                self._cached_contour = contour
                self._status = "success"
                n = len(contour)
                self._message = f"Haiku OK ({n} vertices) at {completed_at[11:19]}"
            else:
                self._status = "error"
                self._message = f"Haiku failed: {err_msg or 'unknown'}"
            callback = self._on_result
            self._on_result = None
        _haiku_log(self._message)

        if callback is not None:
            try:
                callback(mask, contour)
            except Exception:
                pass
