"""RGB-D camera abstraction (Intel RealSense D405)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RgbdCamera(Protocol):
    @property
    def resolution(self) -> tuple[int, int]: ...

    @property
    def intrinsics(self) -> list[list[float]]: ...

    @property
    def label(self) -> str: ...

    def host_metadata(self) -> dict: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_running(self) -> bool: ...

    def read_frame(self, *, block: bool = True) -> tuple[np.ndarray, np.ndarray] | None: ...


def open_camera(*, serial: str | None = None) -> RgbdCamera:
    """Return a started RealSense D405 camera instance."""
    from scanner.realsense_camera import RealSenseD405

    return RealSenseD405(serial=serial)
