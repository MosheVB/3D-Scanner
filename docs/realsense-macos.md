# RealSense D405 on macOS

Intel’s [macOS installation guide](https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/) explains why the D405 often fails on Mac without extra steps. This project wraps those constraints in scripts and diagnostics.

## What the official SDK says (and what it means here)

| Official note | Practical impact |
|---------------|------------------|
| **macOS 12+ (Monterey) and newer:** librealsense tools that use libusb need **`sudo`** for USB access | Run probe, preview, and any librealsense tool as root. Use `./scripts/realsense_sudo.sh …` so the same conda Python and env vars are preserved (`sudo -E`). |
| **RealSense Viewer is not supported** on macOS in current releases | Do not rely on Viewer to test the camera. Use `scripts/realsense_probe.py` and `scripts/realsense_preview.py` instead. |
| **IMU / motion sensors disabled** on macOS | No impact for D405-only RGB-D scanning. |
| Build from source needs **`FORCE_RSUSB_BACKEND=ON`** and Homebrew **libusb** | Prefer **`brew install librealsense`** for CLI checks; Python uses **`pyrealsense2-macosx`** (community wheels, not Intel’s Linux/Windows PyPI package). |
| macOS support is **incomplete** | If USB shows the camera but librealsense sees **0 devices**, use Linux/Windows or **`--camera oak`** for production scans. |

## One-time setup (Apple Silicon, conda env `3d-scanner`)

```bash
cd "/path/to/3D-Scanner"
conda activate 3d-scanner
pip install -r requirements.txt   # installs pyrealsense2-macosx on Darwin

# Optional: Homebrew SDK tools (rs-enumerate-devices, etc.)
brew install librealsense
```

Official source build (only if you need a custom librealsense; most users should use brew + pip above):

```bash
brew install cmake libusb pkg-config openssl
git clone https://github.com/realsenseai/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=true -DFORCE_RSUSB_BACKEND=ON
make -j2
# Examples must run with sudo on macOS 12+:
sudo ./examples/rs-enumerate-devices
```

If `make` fails with `library not found for -lusb-1.0`:

```bash
/bin/launchctl setenv LIBRARY_PATH /opt/homebrew/lib
```

If CMake cannot find OpenSSL:

```bash
export OPENSSL_ROOT_DIR="$(brew --prefix openssl)"
```

## Verify the camera (run in Terminal.app)

1. Quit FaceTime, Zoom, Teams, Photo Booth, and any “camera” app.
2. Plug the D405 into a **direct USB-C / USB 3** port (no unpowered hub).
3. System Settings → Privacy & Security → **Camera** → allow **Terminal** (and Cursor if you run from the IDE).

```bash
conda activate 3d-scanner
python scripts/realsense_mac_diag.py

# macOS 12+ — required for librealsense USB (per Intel docs):
./scripts/realsense_sudo.sh scripts/realsense_probe.py
./scripts/realsense_sudo.sh scripts/realsense_preview.py

# Optional cross-check with Homebrew SDK (also needs sudo):
sudo /opt/homebrew/bin/rs-enumerate-devices -s
```

**Success:** probe lists the D405 with serial and USB type; preview shows RGB | depth until you press `q`.

**Failure patterns:**

| Symptom | Likely cause | What to try |
|---------|--------------|-------------|
| USB visible, **0** pyrealsense devices | macOS **UVCAssistant** holds the device | Run with **`./scripts/realsense_sudo.sh`** (not plain `python`). Use Terminal.app, not only the IDE. |
| `failed to set power state` | Another app or a stale handle | Quit camera apps, unplug 5s, replug, run probe with sudo within ~10s of replug. |
| Works in Terminal with sudo, fails in IDE | IDE sandbox / no Camera privacy | Grant Camera to Cursor or always use Terminal + `realsense_sudo.sh`. |
| Still 0 devices after sudo | Platform limitation | `brew install librealsense` + `sudo rs-enumerate-devices -s`; if still empty, use Linux or OAK-D Lite. |

## Using RealSense in this repo

- **Module:** `scanner/realsense_camera.py` (`RealSenseD405`, subprocess enumeration on macOS to avoid segfaults).
- **Preview / probe:** `scripts/realsense_preview.py`, `scripts/realsense_probe.py`.
- **Default CLI capture** (`python -m scanner capture`) still targets **OAK-D Lite** (`depthai`). RealSense is integrated for preview/diagnostics and `scanner/rgbd_camera.open_camera(backend="realsense")` for future or custom workflows.

## References

- [macOS installation for RealSense SDK](https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/) (Intel / RealSense)
- [pyrealsense2-macosx](https://github.com/cansik/pyrealsense2-macosx) (Python wheels for macOS)
- [Homebrew librealsense](https://formulae.brew.sh/formula/librealsense)
