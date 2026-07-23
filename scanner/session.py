"""Scan session storage on disk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Sequence

import cv2
import numpy as np

from scanner.depth_vis import colorize_depth_mm, depth_mm_linear_gray8

ScanMode = Literal["turntable", "handheld"]


@dataclass
class FrameRecord:
    index: int
    rgb: str
    depth: str
    tilt_deg: float | None = None
    rotation_deg: float | None = None


@dataclass
class ScanSession:
    version: int = 1
    mode: ScanMode = "turntable"
    name: str = ""
    created: str = ""
    rgb_width: int = 0
    rgb_height: int = 0
    intrinsics: List[List[float]] = field(default_factory=list)
    frames: List[FrameRecord] = field(default_factory=list)

    @property
    def root(self) -> Path:
        raise NotImplementedError("root is set after load/create")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mode": self.mode,
            "name": self.name,
            "created": self.created,
            "rgb_width": self.rgb_width,
            "rgb_height": self.rgb_height,
            "intrinsics": self.intrinsics,
            "frames": [asdict(f) for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict, root: Path) -> ScanSession:
        session = cls(
            version=data.get("version", 1),
            mode=data.get("mode", "turntable"),
            name=data.get("name", root.name),
            created=data.get("created", ""),
            rgb_width=data.get("rgb_width", 0),
            rgb_height=data.get("rgb_height", 0),
            intrinsics=data.get("intrinsics", []),
            frames=[
                FrameRecord(
                    index=int(f.get("index", 0)),
                    rgb=str(f["rgb"]),
                    depth=str(f["depth"]),
                    tilt_deg=f.get("tilt_deg"),
                    rotation_deg=f.get("rotation_deg"),
                )
                for f in data.get("frames", [])
            ],
        )
        session._root = root  # type: ignore[attr-defined]
        return session


def _attach_root(session: ScanSession, root: Path) -> ScanSession:
    session._root = root  # type: ignore[attr-defined]
    return session


def session_root(session: ScanSession) -> Path:
    return getattr(session, "_root")


def create_session(
    base_dir: Path,
    name: str,
    mode: ScanMode,
    rgb_size: tuple[int, int],
    intrinsics: Sequence[Sequence[float]],
) -> ScanSession:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = base_dir / f"{stamp}_{name}" if name else base_dir / stamp
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "frames").mkdir()

    session = ScanSession(
        mode=mode,
        name=name or folder.name,
        created=datetime.now(timezone.utc).isoformat(),
        rgb_width=rgb_size[0],
        rgb_height=rgb_size[1],
        intrinsics=[list(map(float, row)) for row in intrinsics],
    )
    _attach_root(session, folder)
    save_manifest(session)
    return session


def save_manifest(session: ScanSession) -> None:
    root = session_root(session)
    path = root / "manifest.json"
    path.write_text(
        json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_session(path: Path) -> ScanSession:
    root = path if path.is_dir() else path.parent
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"No manifest.json in {root}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return _attach_root(ScanSession.from_dict(data, root), root)


def add_frame(
    session: ScanSession,
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    *,
    tilt_deg: float | None = None,
    rotation_deg: float | None = None,
) -> FrameRecord:
    idx = len(session.frames)
    frames_dir = session_root(session) / "frames"
    rgb_path = frames_dir / f"{idx:04d}_rgb.png"
    depth_path = frames_dir / f"{idx:04d}_depth.png"
    depth_vis_path = frames_dir / f"{idx:04d}_depth_vis.jpg"
    depth_gray_path = frames_dir / f"{idx:04d}_depth_gray.png"
    cv2.imwrite(str(rgb_path), rgb_bgr)
    cv2.imwrite(str(depth_path), depth_mm)
    cv2.imwrite(str(depth_vis_path), colorize_depth_mm(depth_mm, use_d405_range=True))
    cv2.imwrite(str(depth_gray_path), depth_mm_linear_gray8(depth_mm))

    record = FrameRecord(
        index=idx,
        rgb=str(rgb_path.relative_to(session_root(session))),
        depth=str(depth_path.relative_to(session_root(session))),
        tilt_deg=tilt_deg,
        rotation_deg=rotation_deg,
    )
    session.frames.append(record)
    save_manifest(session)
    return record


def load_frame_rgb_depth(session: ScanSession, record: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
    root = session_root(session)
    rgb = cv2.imread(str(root / record.rgb), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / record.depth), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise IOError(f"Could not load frame {record.index}")
    return rgb, depth
