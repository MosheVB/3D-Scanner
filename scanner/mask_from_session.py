"""Build motion mask from saved session frames (no camera / turntable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanner.platform_io import configure_stdio_utf8
from scanner.session import load_frame_rgb_depth, load_session, session_root
from scanner.session_mask import save_session_mask, build_mask_from_preview


def build_mask_from_session(
    session_path: Path,
    *,
    max_frames: int = 24,
    stride: int | None = None,
) -> int:
    session = load_session(session_path)
    root = session_root(session)
    frames = session.frames
    if len(frames) < 3:
        print("Need at least 3 frames in session.", file=sys.stderr)
        return 1

    if stride is None:
        stride = max(1, len(frames) // max_frames)
    picked = frames[::stride][:max_frames]

    rgb_list = []
    depth_list = []
    for rec in picked:
        rgb, depth = load_frame_rgb_depth(session, rec)
        rgb_list.append(rgb)
        depth_list.append(depth)

    print(f"Building motion mask from {len(picked)} frames (stride {stride}) …")
    result = build_mask_from_preview(rgb_list, depth_list)
    spec = save_session_mask(root, result, rgb_list)
    lo, hi = spec.depth_band_mm
    print(
        f"Saved mask → {root / 'mask_include.png'}  "
        f"({spec.extra.get('include_pixels', 0):,} px, depth {lo:.0f}–{hi:.0f} mm)"
    )
    print(f"Preview: {root / 'mask_motion_combo.jpg'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="Build motion-union mask from an existing scan session",
    )
    parser.add_argument("session", type=Path, help="Session folder")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Max preview frames to sample (default 24)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frame stride (default: auto from max-frames)",
    )
    args = parser.parse_args(argv)
    return build_mask_from_session(
        args.session,
        max_frames=args.max_frames,
        stride=args.stride,
    )


if __name__ == "__main__":
    sys.exit(main())
