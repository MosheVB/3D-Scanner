#!/usr/bin/env bash
# One-time setup for Linux: Miniconda + 3d-scanner env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="3d-scanner"

if ! command -v conda >/dev/null 2>&1; then
  if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    echo "Installing Miniconda to $CONDA_ROOT ..."
    tmp="$(mktemp)"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$tmp"
    bash "$tmp" -b -p "$CONDA_ROOT"
    rm -f "$tmp"
  fi
  export PATH="$CONDA_ROOT/bin:$PATH"
fi

conda config --set channel_priority strict
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.11 pip -c conda-forge --override-channels -y
fi

conda run -n "$ENV_NAME" conda install open3d -c conda-forge --override-channels -y
conda run -n "$ENV_NAME" pip install -r "$REPO_ROOT/requirements.txt"

# --- Hardware prerequisites the pip wheels do not cover -------------------
# pyrealsense2 bundles librealsense, so the SDK itself needs no system install,
# but the D405 is only reachable as a plain user with udev rules present.
UDEV_RULES="/etc/udev/rules.d/99-realsense-libusb.rules"
if [[ ! -f "$UDEV_RULES" ]]; then
  echo ""
  echo "RealSense udev rules are not installed ($UDEV_RULES)."
  echo "Without them the D405 is root-only and capture fails with a permission error."
  echo "Install them with:"
  echo "  sudo curl -fsSL -o $UDEV_RULES \\"
  echo "    https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules"
  echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
fi

# BLE turntable control needs a running BlueZ stack for bleak to talk to.
if ! systemctl is-active --quiet bluetooth 2>/dev/null; then
  echo ""
  echo "bluetooth.service is not active — the BLE turntable will not connect."
  echo "  sudo apt install bluez && sudo systemctl enable --now bluetooth"
fi

# USB autosuspend on the camera shows up as random mid-scan disconnects.
echo ""
echo "Tip: if the camera drops mid-scan, disable USB autosuspend for it"
echo "     (kernel arg usbcore.autosuspend=-1, or a per-device udev rule)."

echo ""
echo "Done. Activate with:"
echo "  export PATH=\"$CONDA_ROOT/bin:\$PATH\""
echo "  conda activate $ENV_NAME"
echo "  cd \"$REPO_ROOT\""
echo "  python -m scanner --help"
