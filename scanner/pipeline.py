"""DepthAI pipeline: synchronized RGB + depth aligned to RGB (OAK-D Lite / RVC2)."""

from __future__ import annotations

from datetime import timedelta
from typing import Tuple

import depthai as dai

from scanner.config import CAPTURE_FPS, MONO_HEIGHT, MONO_WIDTH, RGB_HEIGHT, RGB_WIDTH


def build_rgbd_pipeline(
    rgb_size: Tuple[int, int] = (RGB_WIDTH, RGB_HEIGHT),
    mono_size: Tuple[int, int] = (MONO_WIDTH, MONO_HEIGHT),
    fps: float = CAPTURE_FPS,
) -> tuple[dai.Pipeline, dai.MessageQueue, dai.CalibrationHandler]:
    """Build pipeline and return (pipeline, sync queue, device calibration)."""
    pipeline = dai.Pipeline()
    device = pipeline.getDefaultDevice()
    platform = device.getPlatform()
    calib = device.readCalibration()

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)

    # Extended disparity ~halves MinZ (~35 cm → ~20 cm at 640×480) for close-up scans
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(True)
    sync.setSyncThreshold(timedelta(seconds=1 / (2 * fps)))

    rgb_out = cam_rgb.requestOutput(
        size=rgb_size,
        fps=fps,
        enableUndistortion=True,
        type=dai.ImgFrame.Type.BGR888p,
    )
    left_out = left.requestOutput(size=mono_size, fps=fps)
    right_out = right.requestOutput(size=mono_size, fps=fps)

    rgb_out.link(sync.inputs["rgb"])
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)
        align.outputAligned.link(sync.inputs["depth_aligned"])
    else:
        stereo.depth.link(sync.inputs["depth_aligned"])
        rgb_out.link(stereo.inputAlignTo)

    queue = sync.out.createOutputQueue()
    return pipeline, queue, calib


def read_synced_frame(
    queue: dai.MessageQueue, block: bool = True
) -> tuple[dai.ImgFrame, dai.ImgFrame] | None:
    """Return (rgb, depth_aligned) or None if non-blocking and empty."""
    group = queue.get() if block else queue.tryGet()
    if group is None:
        return None
    if not isinstance(group, dai.MessageGroup):
        return None
    rgb = group["rgb"]
    depth = group["depth_aligned"]
    return rgb, depth
