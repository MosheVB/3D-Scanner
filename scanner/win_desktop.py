"""Windows desktop screenshot and mouse/keyboard control (ctypes, no extra deps)."""

from __future__ import annotations

import ctypes
import platform
import time
from ctypes import wintypes
from typing import Literal

import numpy as np

if platform.system() != "Windows":
    raise RuntimeError("win_desktop is Windows-only")

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_ABSOLUTE = 0x8000


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


def screen_size() -> tuple[int, int]:
    return _user32.GetSystemMetrics(SM_CXSCREEN), _user32.GetSystemMetrics(SM_CYSCREEN)


def screenshot_bgr() -> np.ndarray:
    """Full-screen BGR uint8 image (top-left origin)."""
    width, height = screen_size()
    hdc_screen = _user32.GetDC(0)
    hdc_mem = _gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = _gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    _gdi32.SelectObject(hdc_mem, hbmp)
    _gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, 0, 0, SRCCOPY)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    buf = (ctypes.c_ubyte * (width * height * 4))()
    _gdi32.GetDIBits(hdc_mem, hbmp, 0, height, buf, ctypes.byref(bmi), DIB_RGB_COLORS)

    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(hdc_mem)
    _user32.ReleaseDC(0, hdc_screen)

    bgra = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    return bgra[:, :, :3].copy()


def _to_absolute(x: int, y: int) -> tuple[int, int]:
    sw, sh = screen_size()
    ax = int(x * 65535 / max(sw - 1, 1))
    ay = int(y * 65535 / max(sh - 1, 1))
    return ax, ay


def mouse_move(x: int, y: int) -> None:
    ax, ay = _to_absolute(x, y)
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def mouse_click(
    x: int,
    y: int,
    *,
    button: Literal["left", "right"] = "left",
    double: bool = False,
) -> None:
    mouse_move(x, y)
    time.sleep(0.05)
    down = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
    up = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
    for _ in range(2 if double else 1):
        for flag in (down, up):
            inp = INPUT(type=INPUT_MOUSE)
            inp.union.mi = MOUSEINPUT(0, 0, 0, flag, 0, None)
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            time.sleep(0.03)


def mouse_scroll(clicks: int) -> None:
    """Positive = scroll up, negative = scroll down."""
    MOUSEEVENTF_WHEEL = 0x0800
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = MOUSEINPUT(0, 0, int(clicks * 120), MOUSEEVENTF_WHEEL, 0, None)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def available() -> bool:
    return platform.system() == "Windows"
