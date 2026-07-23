# 3D-Scanner

A portable hybrid 3D scanner built from off-the-shelf hardware: an **Intel RealSense D405** close-range RGB-D camera and a **Revopoint dual-axis BLE turntable**, driven by a Python pipeline that goes from live capture to a clean, watertight, Fusion 360-ready mesh.

Two capture modes:

- **Turntable mode** — the platter and tilt axis are BLE-controlled; frames are captured at known (tilt, rotation) poses and fused with pose-guided TSDF integration in the turntable frame.
- **Handheld mode** — free capture with RGB-D odometry / ICP registration for larger subjects.

## Highlights

- **Reverse-engineered BLE turntable protocol** ([scanner/turntable.py](scanner/turntable.py)) — the Revopoint dual-axis turntable speaks an undocumented serial-over-BLE protocol (service `0xFFE0`). Tilt, rotation, speed, and homing commands were recovered by protocol analysis and wrapped in an async Python controller.
- **Live bounded voxel preview** ([scanner/voxel_model.py](scanner/voxel_model.py)) — click the object in the live view and a ~2 MB voxel model integrates only masked object depth in real time; RAM stays flat during long scans instead of accumulating an ever-growing point cloud.
- **Web control panel** ([scanner/web.py](scanner/web.py)) — a dependency-free `http.server` app that owns the camera, streams live RGB/depth, and exposes a REST API so capture can be driven remotely (or by an agent).
- **LLM-assisted segmentation** ([scanner/mask_haiku.py](scanner/mask_haiku.py)) — optional Claude Haiku vision pass that proposes the scan-region shape, isolated from the streaming thread so vision latency never stalls capture.
- **Cross-platform capture** — Windows is the primary host; macOS support (experimental) is documented in [docs/realsense-macos.md](docs/realsense-macos.md).
- **MCP desktop-control server** ([mcp/desktop-control/server.py](mcp/desktop-control/server.py)) — a small Model Context Protocol server exposing screenshot + mouse/keyboard control, used to let an AI agent operate the scanner UI end-to-end.

## Pipeline

```
capture (RealSense D405, BLE turntable poses)
  → depth preprocessing & masking        scanner/depth_preprocess.py, scanner/object_mask.py
  → per-frame point clouds               scanner/pointcloud.py
  → registration / pose-guided fusion    (core fusion + axis-calibration modules withheld — see below)
  → segmentation & meshing               scanner/segment.py, scanner/mesh.py
  → measurement vs. ground truth         scanner/measure.py
```

Typical session output: `manifest.json`, raw `frames/`, `scan_object.ply`, and a Fusion-friendly `scan_mesh.obj`. Depth is stored in mm; reconstruction runs in meters.

## About this public mirror

This is a curated mirror of a private working repository. A few core modules are withheld:

- the turntable **axis/hub solver** and pose-guided **TSDF fusion & calibration suite** (`tt_*`, `turntable_axis_solver`, `turntable_fusion`, `depth_fusion`)
- the **autonomous tuning agents** (Claude-driven depth autotune and operator loop)

Because of that, this mirror is meant for **code review, not as a runnable release** — some CLI subcommands import withheld modules. Happy to walk through the withheld code in a conversation or screen share.

## Setup

Python 3.11 via conda-forge (Open3D) plus pip requirements:

```bash
conda create -n 3d-scanner python=3.11 pip -c conda-forge --override-channels -y
conda activate 3d-scanner
conda install open3d -c conda-forge --override-channels -y
pip install -r requirements.txt
```

Windows one-shot bootstrap: run [install-windows.cmd](install-windows.cmd) (fetches and runs `scripts/bootstrap_windows.ps1`).

Useful entry points:

```bash
python -m scanner capture --mode turntable --name myobject   # live capture UI
python -m scanner web                                        # web control panel
python scripts/ble_scan.py                                   # find your turntable's BLE address
python scripts/turntable_test.py                             # verify BLE commands, no camera needed
```

## Hardware

| Part | Role |
|---|---|
| Intel RealSense D405 | close-range (70–740 mm) RGB-D capture |
| Revopoint dual-axis BLE turntable | motorized rotation + tilt at known angles |
| Windows or macOS host | capture + processing |

Concept mockups for a fully portable build (enclosure, rotary dock) are in [docs/mockups/](docs/mockups/).
