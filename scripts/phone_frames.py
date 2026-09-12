#!/usr/bin/env python3
"""Select evenly-spaced, sharpest frames from a phone rotation capture.

Turns one continuous video of a full turntable rotation (or a burst of stills)
into a set of frames spread evenly around the object, keeping only the sharpest
frame in each rotation-degree bin — the input a photogrammetry run wants.

The video is streamed twice rather than loaded: pass one scores every frame for
sharpness, pass two re-reads and writes only the winners. 4K phone footage never
has to fit in RAM.

Usage
-----
    python scripts/phone_frames.py --video IMG_1234.MOV --out captures/mug_phone
    python scripts/phone_frames.py --video spin.mov --out out/ \\
        --spin-start-s 2.5 --spin-end-s 34.0 --pose-step-deg 5
    python scripts/phone_frames.py --images burst/ --out out/

Angles are estimated by interpolating linearly over the spin, so trim the
lead-in and lead-out with --spin-start-s/--spin-end-s and keep the turntable at
a constant speed. The labels drive binning and later pose-guided work;
photogrammetry solves its own poses and only needs the even coverage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scanner.video_frame_select import frame_sharpness, select_best_indices_per_degree

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".dng"}


def _score_frame(frame, score_width: int) -> float:
    """Sharpness at a fixed width so scores stay comparable across frames."""
    h, w = frame.shape[:2]
    if score_width > 0 and w > score_width:
        scale = score_width / float(w)
        frame = cv2.resize(
            frame, (score_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
        )
    return frame_sharpness(frame)


def _scan_video(path: Path, score_width: int) -> tuple[list[float], list[float], float]:
    """Pass one: score every frame, returning (qualities, times_s, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0:
        fps = 30.0
        print(f"  Warning: no frame rate in {path.name}, assuming {fps:g} fps")

    qualities: list[float] = []
    pos_s: list[float] = []
    try:
        while True:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = cap.read()
            if not ok:
                break
            qualities.append(_score_frame(frame, score_width))
            pos_s.append(float(ts_ms) / 1000.0)
    finally:
        cap.release()

    # Container timestamps are unreliable on some phone exports; fall back to
    # frame index over the nominal rate when they never advance.
    if not pos_s or max(pos_s) <= 0.0:
        pos_s = [i / fps for i in range(len(qualities))]
    return qualities, pos_s, fps


def _write_video_picks(path: Path, picks, out_dir: Path, fmt: str, jpeg_quality: int) -> list[dict]:
    """Pass two: re-read the video and write out only the selected frames."""
    wanted = {p.source_index: p for p in picks}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not reopen video: {path}")

    written: list[dict] = []
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if fmt == "jpg" else []
    idx = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            pick = wanted.pop(idx, None)
            if pick is not None:
                name = f"pose_{pick.bin_index:03d}_rot{pick.rotation_deg:06.2f}.{fmt}"
                cv2.imwrite(str(out_dir / name), frame, params)
                written.append(
                    {
                        "file": name,
                        "rotation_deg": round(pick.rotation_deg, 3),
                        "bin_index": pick.bin_index,
                        "quality": round(pick.quality, 3),
                        "source_index": pick.source_index,
                    }
                )
            idx += 1
    finally:
        cap.release()

    if wanted:
        print(f"  Warning: {len(wanted)} selected frames were not readable on the second pass")
    return written


def _scan_images(paths: list[Path], score_width: int) -> list[float]:
    qualities: list[float] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  Warning: could not read {p.name}, scoring it 0")
            qualities.append(0.0)
            continue
        qualities.append(_score_frame(img, score_width))
    return qualities


def _write_image_picks(paths: list[Path], picks, out_dir: Path) -> list[dict]:
    """Copy the originals so EXIF (focal length, model) survives for COLMAP."""
    written: list[dict] = []
    for pick in picks:
        src = paths[pick.source_index]
        name = f"pose_{pick.bin_index:03d}_rot{pick.rotation_deg:06.2f}{src.suffix.lower()}"
        shutil.copy2(src, out_dir / name)
        written.append(
            {
                "file": name,
                "rotation_deg": round(pick.rotation_deg, 3),
                "bin_index": pick.bin_index,
                "quality": round(pick.quality, 3),
                "source_index": pick.source_index,
                "source_file": src.name,
            }
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="One continuous rotation recording.")
    src.add_argument("--images", type=Path, help="Directory of stills, in capture order.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for selected frames.")
    ap.add_argument("--rotation-deg", type=float, default=360.0, help="Rotation covered (default 360).")
    ap.add_argument(
        "--pose-step-deg",
        type=float,
        default=5.0,
        help="Degrees per output frame (default 5 -> 72 frames over a full turn).",
    )
    ap.add_argument("--spin-start-s", type=float, default=None, help="Trim lead-in before rotation starts.")
    ap.add_argument("--spin-end-s", type=float, default=None, help="Trim lead-out after rotation ends.")
    ap.add_argument(
        "--score-width",
        type=int,
        default=960,
        help="Downscale width for sharpness scoring only; 0 disables (default 960).",
    )
    ap.add_argument("--format", choices=("png", "jpg"), default="png", help="Video frame output format.")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --format jpg.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        if not args.video.is_file():
            raise SystemExit(f"No such video: {args.video}")
        print(f"Scanning {args.video.name} ...")
        qualities, times, fps = _scan_video(args.video, args.score_width)
        if not qualities:
            raise SystemExit("No frames decoded — is this a readable video?")
        print(f"  {len(qualities)} frames at {fps:g} fps ({times[-1]:.1f}s)")
        source_desc = args.video.name
        image_paths: list[Path] = []
    else:
        if not args.images.is_dir():
            raise SystemExit(f"No such directory: {args.images}")
        image_paths = sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise SystemExit(f"No images found in {args.images}")
        print(f"Scanning {len(image_paths)} images in {args.images} ...")
        qualities = _scan_images(image_paths, args.score_width)
        times = [float(i) for i in range(len(qualities))]
        source_desc = str(args.images)

    spin_t0 = args.spin_start_s
    spin_duration = None
    if args.spin_end_s is not None:
        spin_duration = args.spin_end_s - (spin_t0 if spin_t0 is not None else times[0])
        if spin_duration <= 0:
            raise SystemExit("--spin-end-s must be after --spin-start-s")

    picks = select_best_indices_per_degree(
        qualities,
        times,
        spin_t0=spin_t0,
        spin_duration=spin_duration,
        rotation_deg_total=args.rotation_deg,
        pose_step_deg=args.pose_step_deg,
    )

    n_bins = max(1, int(round(args.rotation_deg / args.pose_step_deg)))
    print(f"  {len(picks)}/{n_bins} degree bins filled (step {args.pose_step_deg:g} deg)")
    if len(picks) < n_bins:
        print("  Warning: gaps in angular coverage — spin slower, or raise --pose-step-deg")

    if args.video is not None:
        written = _write_video_picks(
            args.video, picks, args.out, args.format, args.jpeg_quality
        )
    else:
        written = _write_image_picks(image_paths, picks, args.out)

    manifest = {
        "source": source_desc,
        "rotation_deg": args.rotation_deg,
        "pose_step_deg": args.pose_step_deg,
        "spin_start_s": args.spin_start_s,
        "spin_end_s": args.spin_end_s,
        "frames": written,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(written)} frames + manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
