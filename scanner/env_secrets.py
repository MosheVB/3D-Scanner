"""Load optional secrets (e.g. ANTHROPIC_API_KEY) into os.environ at process start."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _read_windows_env(name: str, *, scope: str) -> str | None:
    if platform.system() != "Windows":
        return None
    try:
        import winreg

        root = winreg.HKEY_CURRENT_USER if scope == "user" else winreg.HKEY_LOCAL_MACHINE
        with winreg.OpenKey(root, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            text = str(value).strip()
            return text or None
    except OSError:
        return None


def _read_windows_user_env(name: str) -> str | None:
    return _read_windows_env(name, scope="user")


def _read_windows_machine_env(name: str) -> str | None:
    return _read_windows_env(name, scope="machine")


def _read_local_key_file() -> str | None:
    """Optional gitignored file: repo/.anthropic_api_key (single line, no quotes)."""
    root = Path(__file__).resolve().parent.parent
    path = root / ".anthropic_api_key"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def ensure_anthropic_api_key() -> bool:
    """Populate ANTHROPIC_API_KEY from env, Windows user registry, or local file."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True
    for source in (_read_windows_user_env("ANTHROPIC_API_KEY"), _read_local_key_file()):
        if source:
            os.environ["ANTHROPIC_API_KEY"] = source
            return True
    return False
