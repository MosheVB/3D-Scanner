"""Quick import and screenshot smoke test (no MCP client required)."""

from __future__ import annotations

import sys


def main() -> int:
    print("Importing server module...")
    import server  # noqa: F401

    print("get_screen_size:", server.get_screen_size())
    print("screenshot (monitor 0 = all)...")
    result = server.screenshot(monitor=0)
    print("screenshot:", {k: v for k, v in result.items() if k != "base64"})
    path = result["path"]
    if not __import__("pathlib").Path(path).is_file():
        print("ERROR: screenshot file missing:", path, file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
