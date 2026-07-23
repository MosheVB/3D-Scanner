"""RGB-D odometry diagnostics: motion estimates and feed visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scanner.config import ODOMETRY_DEPTH_STRIDE
from scanner.depth_vis import colorize_depth_mm
from scanner.rgbd_odometry import (
    compute_rgbd_odometry_step,
    odometry_feed_arrays,
    pinhole_intrinsic,
    pose_step_metrics,
    rgbd_from_arrays,
)
from scanner.session import ScanSession, load_frame_rgb_depth


MIN_TRUST_TRANSLATION_MM = 0.5
MIN_TRUST_ROTATION_DEG = 0.5
MIN_FEATURE_MATCHES = 20


@dataclass
class OdometryStepReport:
    index: int
    success: bool
    translation_mm: float
    rotation_deg: float
    feature_matches: int


def orb_feature_matches(
    rgb_a_bgr: np.ndarray,
    rgb_b_bgr: np.ndarray,
    *,
    max_features: int = 500,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
    """ORB keypoints and cross-checked matches on odometry-resolution RGB."""
    gray_a = cv2.cvtColor(rgb_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(rgb_b_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(des_a) == 0 or len(des_b) == 0:
        return kp_a, kp_b, []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    matches = sorted(matches, key=lambda m: m.distance)
    return kp_a, kp_b, matches


def analyze_odometry_steps(
    session: ScanSession,
    *,
    stride: int = ODOMETRY_DEPTH_STRIDE,
) -> list[OdometryStepReport]:
    """Run consecutive-frame RGB-D odometry and collect per-step metrics."""
    if len(session.frames) < 2:
        return []

    rgb0, depth0 = load_frame_rgb_depth(session, session.frames[0])
    h, w = depth0.shape[:2]
    intrinsic = pinhole_intrinsic(session.intrinsics, w, h, stride=stride)
    prev_rgbd = rgbd_from_arrays(rgb0, depth0, intrinsic, stride=stride)
    prev_rgb, _ = odometry_feed_arrays(rgb0, depth0, stride=stride)

    reports: list[OdometryStepReport] = []
    for i in range(1, len(session.frames)):
        rgb, depth = load_frame_rgb_depth(session, session.frames[i])
        rgbd = rgbd_from_arrays(rgb, depth, intrinsic, stride=stride)
        feed_rgb, _ = odometry_feed_arrays(rgb, depth, stride=stride)

        success, step, _ = compute_rgbd_odometry_step(rgbd, prev_rgbd, intrinsic)
        trans_mm, rot_deg = pose_step_metrics(step)
        _, _, matches = orb_feature_matches(prev_rgb, feed_rgb)
        reports.append(
            OdometryStepReport(
                index=i,
                success=success,
                translation_mm=trans_mm,
                rotation_deg=rot_deg,
                feature_matches=len(matches),
            )
        )
        prev_rgbd = rgbd
        prev_rgb = feed_rgb
    return reports


def print_motion_report(reports: list[OdometryStepReport]) -> bool:
    """Print per-step motion and median summary. Returns True if poses are trustworthy."""
    if not reports:
        print("Odometry motion: need at least 2 frames.")
        return False

    print("Odometry motion (consecutive frames):")
    for r in reports:
        status = "ok" if r.success else "FAIL"
        print(
            f"  step {r.index - 1:04d}->{r.index:04d}: "
            f"t={r.translation_mm:.3f} mm  rot={r.rotation_deg:.3f} deg  "
            f"matches={r.feature_matches}  [{status}]"
        )
        if r.feature_matches < MIN_FEATURE_MATCHES:
            print(
                f"    WARNING: only {r.feature_matches} feature matches "
                f"(<{MIN_FEATURE_MATCHES}); texture may be too weak for odometry."
            )

    trans = np.array([r.translation_mm for r in reports], dtype=np.float64)
    rots = np.array([r.rotation_deg for r in reports], dtype=np.float64)
    med_t = float(np.median(trans))
    med_r = float(np.median(rots))
    print()
    print(
        f"  median translation: {med_t:.3f} mm  "
        f"(min {trans.min():.3f}, max {trans.max():.3f})"
    )
    print(
        f"  median rotation:    {med_r:.3f} deg  "
        f"(min {rots.min():.3f}, max {rots.max():.3f})"
    )

    trusted = med_t >= MIN_TRUST_TRANSLATION_MM and med_r >= MIN_TRUST_ROTATION_DEG
    if not trusted:
        reasons: list[str] = []
        if med_t < MIN_TRUST_TRANSLATION_MM:
            reasons.append(f"median translation {med_t:.3f} mm < {MIN_TRUST_TRANSLATION_MM} mm")
        if med_r < MIN_TRUST_ROTATION_DEG:
            reasons.append(f"median rotation {med_r:.3f} deg < {MIN_TRUST_ROTATION_DEG} deg")
        print()
        print("  WARNING: odometry is failing to detect motion - poses should NOT be trusted.")
        print(f"    ({'; '.join(reasons)})")
    else:
        print()
        print("  Motion above trust thresholds - odometry is detecting movement.")
    return trusted


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_feature_matches(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    kp_a: list[cv2.KeyPoint],
    kp_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    *,
    max_draw: int = 80,
) -> np.ndarray:
    h = max(rgb_a.shape[0], rgb_b.shape[0])
    wa, wb = rgb_a.shape[1], rgb_b.shape[1]
    canvas = np.zeros((h, wa + wb, 3), dtype=np.uint8)
    canvas[: rgb_a.shape[0], :wa] = rgb_a
    canvas[: rgb_b.shape[0], wa : wa + wb] = rgb_b
    draw = matches
    if len(draw) > max_draw:
        step = max(1, len(draw) // max_draw)
        draw = draw[::step][:max_draw]
    for m in draw:
        pa = tuple(int(round(v)) for v in kp_a[m.queryIdx].pt)
        pb = (
            int(round(kp_b[m.trainIdx].pt[0])) + wa,
            int(round(kp_b[m.trainIdx].pt[1])),
        )
        cv2.line(canvas, pa, pb, (60, 220, 60), 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 2, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pb, 2, (0, 200, 255), -1, cv2.LINE_AA)
    return canvas


def visualize_odometry_step(
    session: ScanSession,
    step_index: int,
    out_path: Path,
    *,
    stride: int = ODOMETRY_DEPTH_STRIDE,
) -> OdometryStepReport:
    """Save RGB/depth feed panels and feature-match overlay for one odometry step."""
    if step_index < 1 or step_index >= len(session.frames):
        raise ValueError(f"step_index must be 1..{len(session.frames) - 1}")

    rec_prev = session.frames[step_index - 1]
    rec_curr = session.frames[step_index]
    rgb_prev, depth_prev = load_frame_rgb_depth(session, rec_prev)
    rgb_curr, depth_curr = load_frame_rgb_depth(session, rec_curr)

    feed_rgb_prev, feed_depth_prev = odometry_feed_arrays(
        rgb_prev, depth_prev, stride=stride
    )
    feed_rgb_curr, feed_depth_curr = odometry_feed_arrays(
        rgb_curr, depth_curr, stride=stride
    )
    dep_vis_prev = colorize_depth_mm(feed_depth_prev, use_d405_range=True)
    dep_vis_curr = colorize_depth_mm(feed_depth_curr, use_d405_range=True)

    h, w = depth_prev.shape[:2]
    intrinsic = pinhole_intrinsic(session.intrinsics, w, h, stride=stride)
    prev_rgbd = rgbd_from_arrays(rgb_prev, depth_prev, intrinsic, stride=stride)
    curr_rgbd = rgbd_from_arrays(rgb_curr, depth_curr, intrinsic, stride=stride)
    success, step, _ = compute_rgbd_odometry_step(curr_rgbd, prev_rgbd, intrinsic)
    trans_mm, rot_deg = pose_step_metrics(step)
    kp_a, kp_b, matches = orb_feature_matches(feed_rgb_prev, feed_rgb_curr)

    feed_row = np.hstack(
        [
            _label_panel(feed_rgb_prev, "prev RGB"),
            _label_panel(dep_vis_prev, "prev depth"),
            _label_panel(feed_rgb_curr, "curr RGB"),
            _label_panel(dep_vis_curr, "curr depth"),
        ]
    )
    match_panel = _draw_feature_matches(
        feed_rgb_prev, feed_rgb_curr, kp_a, kp_b, matches
    )
    match_panel = _label_panel(
        match_panel,
        f"ORB matches={len(matches)}  t={trans_mm:.2f}mm rot={rot_deg:.2f}deg",
    )
    if match_panel.shape[1] != feed_row.shape[1]:
        scale = feed_row.shape[1] / match_panel.shape[1]
        match_panel = cv2.resize(
            match_panel,
            (feed_row.shape[1], int(match_panel.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    combo = np.vstack([feed_row, match_panel])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combo)

    report = OdometryStepReport(
        index=step_index,
        success=success,
        translation_mm=trans_mm,
        rotation_deg=rot_deg,
        feature_matches=len(matches),
    )
    if report.feature_matches < MIN_FEATURE_MATCHES:
        print(
            f"  WARNING step {step_index}: only {report.feature_matches} feature matches "
            f"(<{MIN_FEATURE_MATCHES})"
        )
    return report


def diagnose_session(
    session: ScanSession,
    out_dir: Path,
    *,
    stride: int = ODOMETRY_DEPTH_STRIDE,
    visualize_steps: int = 3,
) -> bool:
    """Run motion report and save visualizations. Returns whether poses are trusted."""
    reports = analyze_odometry_steps(session, stride=stride)
    trusted = print_motion_report(reports)
    if not reports:
        return False

    n_vis = min(visualize_steps, len(reports))
    if n_vis <= 0:
        return trusted

    print()
    print(f"Saving odometry feed visuals ({n_vis} steps) -> {out_dir}")
    idxs = [
        int(round(i * (len(reports) - 1) / max(1, n_vis - 1)))
        for i in range(n_vis)
    ]
    for pos in idxs:
        step_index = reports[pos].index
        out_path = out_dir / f"step_{step_index:04d}_feed_matches.jpg"
        visualize_odometry_step(session, step_index, out_path, stride=stride)
        print(f"  {out_path.name}")
    return trusted
