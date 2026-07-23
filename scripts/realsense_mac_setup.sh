#!/usr/bin/env bash
# macOS RealSense D405 setup aligned with Intel's SDK install guide.
# https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/3d-scanner/bin/python}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Set PYTHON to your 3d-scanner conda python (not found: $PYTHON)" >&2
  exit 1
fi

echo "=== RealSense macOS setup ==="
echo "Guide: https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/"
echo "Project: docs/realsense-macos.md"
echo

if command -v brew >/dev/null 2>&1; then
  if ! brew list librealsense &>/dev/null; then
    echo "Installing Homebrew librealsense (optional SDK tools)…"
    brew install librealsense
  else
    echo "Homebrew librealsense: already installed"
  fi
else
  echo "Homebrew not found — skip 'brew install librealsense' or install Homebrew first."
fi

echo
echo "Installing Python bindings (pyrealsense2-macosx)…"
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"

echo
echo "Running diagnostics (no sudo yet)…"
"$PYTHON" "$ROOT/scripts/realsense_mac_diag.py" || true

echo
echo "Next (macOS 12+ requires sudo for USB — per Intel docs):"
echo "  cd \"$ROOT\""
echo "  ./scripts/realsense_sudo.sh scripts/realsense_probe.py"
echo "  ./scripts/realsense_sudo.sh scripts/realsense_preview.py"
if [[ -x /opt/homebrew/bin/rs-enumerate-devices ]]; then
  echo "  sudo /opt/homebrew/bin/rs-enumerate-devices -s"
fi
