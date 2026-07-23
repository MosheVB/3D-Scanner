"""Windows desktop control MCP server (stdio transport).

Captures screenshots and drives mouse/keyboard for MCP clients (e.g. Cursor).
Set DRY_RUN=1 to log actions without performing them.

Safety: agents must stop at login, captcha, 2FA, payment, and destructive dialogs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

# UTF-8 stderr/stdout for Windows consoles (Cursor spawns this process).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
SCREENSHOT_DIR = Path(
    os.environ.get(
        "DESKTOP_CONTROL_SCREENSHOT_DIR",
        str(Path(tempfile.gettempdir()) / "desktop-control-mcp"),
    )
)

MouseButton = Literal["left", "right", "middle"]

mcp = FastMCP(
    "desktop-control",
    instructions=(
        "Windows desktop mouse, keyboard, and screenshot control. "
        "Coordinates use the virtual screen (multi-monitor aware). "
        "Stop and ask the user at login, captcha, 2FA, payment, or destructive prompts. "
        "Set DRY_RUN=1 to disable all input actions."
    ),
)


def _dry_run_result(action: str, **details: object) -> dict:
    return {"dry_run": True, "action": action, **details}


def _ensure_screenshot_dir() -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOT_DIR


def _import_pyautogui():
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = float(os.environ.get("DESKTOP_CONTROL_PAUSE", "0.05"))
    return pyautogui


@mcp.tool()
def get_screen_size() -> dict:
    """Return virtual screen width and height in pixels (multi-monitor span)."""
    pag = _import_pyautogui()
    width, height = pag.size()
    return {"width": width, "height": height}


@mcp.tool()
def screenshot(
    x: Optional[int] = None,
    y: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    monitor: int = 0,
    return_base64: bool = False,
) -> dict:
    """Capture the screen or a region; save PNG to temp and return its path.

    Args:
        x, y, width, height: Optional region in virtual-screen coordinates.
        monitor: Monitor index for full capture when no region is given (0 = all monitors combined).
        return_base64: If true, also include base64 PNG (large; prefer path only).
    """
    import base64

    import mss

    out_dir = _ensure_screenshot_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"screenshot_{stamp}.png"

    try:
        with mss.mss() as sct:
            if x is not None and y is not None and width is not None and height is not None:
                region = {"left": x, "top": y, "width": width, "height": height}
            else:
                monitors = sct.monitors
                if monitor < 0 or monitor >= len(monitors):
                    raise ValueError(
                        f"monitor index {monitor} out of range (0..{len(monitors) - 1})"
                    )
                region = monitors[monitor]

            img = sct.grab(region)
            mss.tools.to_png(img.rgb, img.size, output=str(path))
            result = {
                "path": str(path),
                "width": img.width,
                "height": img.height,
                "monitor": monitor,
            }
            if return_base64:
                result["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            return result
    except Exception as exc:
        raise RuntimeError(f"screenshot failed: {exc}") from exc


@mcp.tool()
def mouse_move(x: int, y: int, duration: float = 0.0) -> dict:
    """Move the mouse cursor to virtual-screen coordinates (x, y)."""
    if DRY_RUN:
        return _dry_run_result("mouse_move", x=x, y=y, duration=duration)
    pag = _import_pyautogui()
    try:
        pag.moveTo(x, y, duration=max(0.0, duration))
        return {"x": x, "y": y}
    except Exception as exc:
        raise RuntimeError(f"mouse_move failed: {exc}") from exc


@mcp.tool()
def mouse_click(
    x: int,
    y: int,
    button: MouseButton = "left",
    clicks: int = 1,
    interval: float = 0.0,
) -> dict:
    """Click at (x, y). Use clicks=2 for double-click."""
    if DRY_RUN:
        return _dry_run_result(
            "mouse_click", x=x, y=y, button=button, clicks=clicks
        )
    pag = _import_pyautogui()
    try:
        pag.click(
            x=x,
            y=y,
            button=button,
            clicks=max(1, clicks),
            interval=max(0.0, interval),
        )
        return {"x": x, "y": y, "button": button, "clicks": clicks}
    except Exception as exc:
        raise RuntimeError(f"mouse_click failed: {exc}") from exc


@mcp.tool()
def mouse_drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    button: MouseButton = "left",
    duration: float = 0.3,
) -> dict:
    """Drag from (x1, y1) to (x2, y2)."""
    if DRY_RUN:
        return _dry_run_result(
            "mouse_drag",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            button=button,
            duration=duration,
        )
    pag = _import_pyautogui()
    try:
        pag.moveTo(x1, y1)
        pag.dragTo(
            x2,
            y2,
            duration=max(0.0, duration),
            button=button,
        )
        return {"from": [x1, y1], "to": [x2, y2], "button": button}
    except Exception as exc:
        raise RuntimeError(f"mouse_drag failed: {exc}") from exc


@mcp.tool()
def scroll(
    clicks: int,
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> dict:
    """Scroll the mouse wheel at optional position. Positive clicks scroll up."""
    if DRY_RUN:
        return _dry_run_result("scroll", clicks=clicks, x=x, y=y)
    pag = _import_pyautogui()
    try:
        if x is not None and y is not None:
            pag.moveTo(x, y)
        pag.scroll(clicks)
        return {"clicks": clicks, "x": x, "y": y}
    except Exception as exc:
        raise RuntimeError(f"scroll failed: {exc}") from exc


def _type_text_impl(text: str) -> None:
    pag = _import_pyautogui()
    if text.isascii():
        pag.write(text, interval=0.02)
        return
    import pyperclip

    try:
        previous = pyperclip.paste()
    except Exception:
        previous = None
    pyperclip.copy(text)
    pag.hotkey("ctrl", "v")
    if previous is not None:
        try:
            pyperclip.copy(previous)
        except Exception:
            pass


@mcp.tool()
def type_text(text: str) -> dict:
    """Type text at the current focus. Non-ASCII uses clipboard paste on Windows."""
    if DRY_RUN:
        return _dry_run_result("type_text", text=text, length=len(text))
    try:
        _type_text_impl(text)
        return {"typed_length": len(text)}
    except Exception as exc:
        raise RuntimeError(f"type_text failed: {exc}") from exc


@mcp.tool()
def press_key(key: str) -> dict:
    """Press a key or combo, e.g. 'enter', 'escape', 'ctrl+c', 'alt+tab'."""
    if DRY_RUN:
        return _dry_run_result("press_key", key=key)
    pag = _import_pyautogui()
    parts = [part.strip().lower() for part in key.split("+") if part.strip()]
    if not parts:
        raise ValueError("key must not be empty")
    alias = {"esc": "escape", "control": "ctrl", "cmd": "win", "command": "win"}
    parts = [alias.get(part, part) for part in parts]
    try:
        if len(parts) == 1:
            pag.press(parts[0])
        else:
            pag.hotkey(*parts)
        return {"key": key, "parts": parts}
    except Exception as exc:
        raise RuntimeError(f"press_key failed for {key!r}: {exc}") from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")
