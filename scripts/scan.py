#!/usr/bin/env python3
"""Entry point: interactive 3D scan capture and mesh export."""

import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
