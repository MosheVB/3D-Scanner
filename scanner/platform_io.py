"""Windows-safe UTF-8 console and remote-session helpers."""

from __future__ import annotations

import os
import sys
from typing import TextIO

UTF8 = "utf-8"


def is_remote_session() -> bool:
    """True when running in RDP or similar remote desktop on Windows."""
    if os.name != "nt":
        return False
    session = os.environ.get("SESSIONNAME", "")
    if session.upper().startswith("RDP"):
        return True
    client = os.environ.get("CLIENTNAME", "")
    if not client:
        return False
    return client.upper() not in ("", "CONSOLE")


def _reconfigure_stream(stream: TextIO | None) -> None:
    if stream is None or not hasattr(stream, "reconfigure"):
        return
    try:
        stream.reconfigure(encoding=UTF8, errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def configure_stdio_utf8(*, log_remote: bool = True) -> None:
    """Windows/remote-safe UTF-8 for console and logs (°, ×, →, em dashes)."""
    if os.environ.get("PYTHONIOENCODING") is None:
        os.environ["PYTHONIOENCODING"] = UTF8
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)
    if log_remote and is_remote_session():
        session = os.environ.get("SESSIONNAME", "?")
        client = os.environ.get("CLIENTNAME", "?")
        print(
            f"Remote session detected ({session}, client={client}); UTF-8 console enabled.",
            file=sys.stderr,
        )
