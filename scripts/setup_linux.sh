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

echo ""
echo "Done. Activate with:"
echo "  export PATH=\"$CONDA_ROOT/bin:\$PATH\""
echo "  conda activate $ENV_NAME"
echo "  cd \"$REPO_ROOT\""
echo "  python -m scanner --help"
