"""Exclusive file lock so only one process opens the RealSense camera."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from scanner.config import PROJECT_ROOT

LOCK_PATH = PROJECT_ROOT / ".realsense_camera.lock"


@dataclass(frozen=True)
class LockInfo:
    pid: int
    holder: str
    started: float
    command: str


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock() -> LockInfo | None:
    if not LOCK_PATH.exists():
        return None
    try:
        raw: dict[str, Any] = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        return LockInfo(
            pid=int(raw["pid"]),
            holder=str(raw.get("holder", "unknown")),
            started=float(raw.get("started", 0.0)),
            command=str(raw.get("command", "")),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def _remove_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def current_lock() -> LockInfo | None:
    """Return active lock info, or None if the camera is free."""
    info = _read_lock()
    if info is None:
        return None
    if info.pid == os.getpid():
        return info
    if not _pid_alive(info.pid):
        _remove_lock()
        return None
    return info


def _busy_message(info: LockInfo) -> str:
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.started))
    cmd = info.command or f"PID {info.pid}"
    if info.holder == "web":
        stop_hint = (
            "Stop the web viewer with Ctrl+C in its terminal, then retry.\n"
            f"  Or: taskkill /PID {info.pid} /F   (Windows)\n"
            f"  Or: kill {info.pid}              (macOS/Linux)"
        )
    else:
        stop_hint = (
            f"Stop the other process ({cmd}), then retry.\n"
            f"  taskkill /PID {info.pid} /F   (Windows)  |  kill {info.pid}  (Unix)"
        )
    return (
        f"RealSense camera is in use by {info.holder!r} (PID {info.pid}, since {started}).\n"
        f"{stop_hint}"
    )


def acquire(holder: str, *, command: str = "") -> None:
    """Claim the camera lock for this process. Raises RuntimeError if busy."""
    my_pid = os.getpid()
    existing = current_lock()
    if existing is not None and existing.pid == my_pid:
        return

    payload = {
        "pid": my_pid,
        "holder": holder,
        "started": time.time(),
        "command": command,
    }
    for attempt in range(3):
        stale = current_lock()
        if stale is not None and stale.pid != my_pid:
            raise RuntimeError(_busy_message(stale))
        try:
            LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOCK_PATH, "x", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            return
        except FileExistsError:
            if attempt < 2:
                time.sleep(0.05)
                continue
            stale = current_lock()
            if stale is not None and stale.pid == my_pid:
                return
            if stale is not None:
                raise RuntimeError(_busy_message(stale)) from None
            raise RuntimeError(
                "RealSense camera lock file exists but could not be read. "
                f"Delete {LOCK_PATH} if no scanner process is running."
            ) from None


def release() -> None:
    """Release the lock if this process holds it."""
    info = _read_lock()
    if info is None:
        return
    if info.pid != os.getpid():
        return
    _remove_lock()
