"""Shared capture and reconstruction defaults for Intel RealSense D405."""

from pathlib import Path

# RGB + aligned depth — default for live/web preview
RGB_WIDTH = 640
RGB_HEIGHT = 480
CAPTURE_FPS = 15.0

# Turntable scans: match calibration (640x480 @ 15 fps = best D405 depth SNR)
TURNTABLE_CAPTURE_WIDTH = 640
TURNTABLE_CAPTURE_HEIGHT = 480
TURNTABLE_CAPTURE_FPS = 15

# D405 stereo module: RGB + depth share exposure/gain (range ~1-165000 us, gain 16-248)
TURNTABLE_GAIN = 128.0
TURNTABLE_GAIN_BRIGHT = 48.0  # when scene mean already high (room lights + LEDs)
TURNTABLE_AE_WARMUP_FRAMES = 45
TURNTABLE_RGB_TARGET = 130.0  # lock exposure toward this gray mean
TURNTABLE_RGB_MEAN_MIN = 105.0  # boost if below
TURNTABLE_RGB_MEAN_MAX = 150.0  # cut if above
TURNTABLE_HIGHLIGHT_FRAC_MAX = 0.025  # fraction of pixels >= 245 (clipped whites)
# D405 passive stereo: High Accuracy preset (aboutDepth.txt); fallback short_range in SDK
TURNTABLE_VISUAL_PRESET = "medium_density"
# Turntable: object moves between poses — disable temporal (carries state across rotation)
TURNTABLE_HOLE_FILL = False
TURNTABLE_TEMPORAL_FILTER = False

# RGB rotation mask preview (stable exposure, no AE drift)
MASK_RGB_WIDTH = 848
MASK_RGB_HEIGHT = 480
MASK_RGB_FPS = 15
MASK_RGB_RECORD_S = 12.0

# Point cloud / fusion
MAX_DEPTH_MM = 2500
DEPTH_STRIDE = 2
# Live object voxel model (capture right panel — bounded RAM)
LIVE_DEPTH_STRIDE = 4
LIVE_DEPTH_BAND_MM = 90
LIVE_VOXEL_M = 0.006
LIVE_ICP_MAX_CORRESPONDENCE_M = 0.035
LIVE_ICP_MAX_ITER = 25
LIVE_ICP_KEYFRAME_INTERVAL = 3
VOXEL_MODEL_EXTENT_M = 0.22
VOXEL_MODEL_VOXEL_M = 0.005
VOXEL_MODEL_MAX_AXIS = 56
VOXEL_MODEL_MIN_WEIGHT = 2
VOXEL_SIZE_M = 0.004
# Offline turntable fusion / TSDF integration resolution (meters)
FUSION_VOXEL_M = 0.0015
ICP_MAX_CORRESPONDENCE_M = 0.025
# RGB-D odometry (Open3D pipelines.odometry)
ODOMETRY_DEPTH_MIN_M = 0.05
ODOMETRY_DEPTH_MAX_M = 1.2
ODOMETRY_DEPTH_DIFF_MAX = 0.07
ODOMETRY_DEPTH_STRIDE = 2
LIVE_ODOMETRY_DEPTH_STRIDE = 4
POISSON_DEPTH = 9

# Segmentation (table / background removal)
PLANE_DISTANCE_M = 0.008
CLUSTER_EPS_M = 0.015
CLUSTER_MIN_POINTS = 200
# Compact cluster (reject table-sized slabs; matches alt_scanner/tsdf.py)
COMPACT_CLUSTER_EPS_M = 0.010
COMPACT_CLUSTER_MAX_EXTENT_MM = 120.0

# Mask preview (quick 360 before turntable scan)
MASK_PREVIEW_STEPS = 24
MASK_PREVIEW_ROTATE_STEP_DEG = 15.0
MASK_PREVIEW_ROTATE_WAIT_S = 1.25
MASK_PREVIEW_TILT_DEG = 0.0

# D405 distance guidance (mm) — working range 70–740 mm, sweet spot ~100–350 mm
DISTANCE_OPTIMAL_MIN_MM = 150
DISTANCE_OPTIMAL_MAX_MM = 280
DISTANCE_TOO_CLOSE_MM = 70       # D405 minimum working distance
DISTANCE_TOO_FAR_MM = 400
D405_MIN_DEPTH_MM = 70           # hardware minimum; below this depth is zero
CLICK_PATCH_RADIUS = 3
CLICK_SEARCH_MAX_RADIUS = 60
# Ignore clicks in top strip of RGB pane (HUD text is drawn there)
HUD_CLICK_EXCLUDE_TOP_PX = 120
# Left column: RGB on top, adaptive depth strip below (total = RGB_HEIGHT)
LEFT_RGB_PANEL_H = 340
LEFT_DEPTH_STRIP_H = RGB_HEIGHT - LEFT_RGB_PANEL_H

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCANS_DIR = PROJECT_ROOT / "scans"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"

# ---------------------------------------------------------------------------
# Revopoint dual-axis turntable BLE
# Service 0xFFE0 / Characteristic 0xFFE1 (serial-over-BLE, HM-10 style)
# ---------------------------------------------------------------------------
TURNTABLE_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
TURNTABLE_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# Confirmed BLE address of this specific turntable unit
TURNTABLE_BLE_ADDRESS = "XX:XX:XX:XX:XX:XX"  # find yours with scripts/ble_scan.py

# Tilt axis — absolute positions visited per scan pass (-30 to +30 degrees)
TURNTABLE_TILT_LEVELS: list[float] = [-30.0, -15.0, 0.0, 15.0, 30.0]

# Rotation axis — incremental step between captured frames
TURNTABLE_ROTATE_STEP_DEG: float = 15.0   # 24 steps = 360°

# Motor speeds (protocol: larger value = SLOWER; 6.62 is tilt max speed)
TURNTABLE_TILT_SPEED: float = 8.0    # moderate tilt speed
TURNTABLE_ROTATE_SPEED: float = 36.0  # near-max rotation speed (fast step scans)
TURNTABLE_ROTATE_SPEED_FAST: float = 35.64  # protocol fastest rotation
TURNTABLE_VIDEO_ROTATE_SPEED: float = 280.0  # slow continuous spin (larger = slower)
TURNTABLE_VIDEO_CAPTURE_FPS: int = 30  # max aligned RGB-D at 640x480; 0 = auto-probe

# Settle delays after each command (seconds)
TURNTABLE_ROTATE_WAIT_S: float = 2.0  # wait after each rotation step
TURNTABLE_TILT_WAIT_S: float = 4.0    # wait after tilt change (longer travel)
TURNTABLE_HOME_WAIT_S: float = 12.0   # wait after homing full rotation axis
