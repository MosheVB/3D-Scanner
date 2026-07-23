"""Register and merge multiple depth frames into one point cloud."""

from __future__ import annotations

from typing import List

import numpy as np
import open3d as o3d

from scanner.config import ICP_MAX_CORRESPONDENCE_M, VOXEL_SIZE_M


def _prep(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel)
    if len(down.points) == 0:
        return down
    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
    )
    return down


def register_frame_to_reference(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = VOXEL_SIZE_M,
    max_corr: float = ICP_MAX_CORRESPONDENCE_M,
    max_iteration: int = 50,
) -> np.ndarray:
    """Point-to-plane ICP; returns 4x4 transform mapping source → target frame."""
    src = _prep(source, voxel_size)
    tgt = _prep(target, voxel_size)
    if len(src.points) == 0 or len(tgt.points) == 0:
        return np.eye(4)

    init = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    return result.transformation


def fuse_point_clouds_concat(
    clouds: List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Merge clouds without registration (same pose / quick validation only)."""
    if not clouds:
        return o3d.geometry.PointCloud()
    merged = clouds[0]
    for cloud in clouds[1:]:
        merged += cloud
    merged = merged.voxel_down_sample(VOXEL_SIZE_M)
    if len(merged.points) > 0:
        merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE_M * 4, max_nn=30
            )
        )
    return merged


def fuse_point_clouds(
    clouds: List[o3d.geometry.PointCloud],
    *,
    use_icp: bool = True,
) -> o3d.geometry.PointCloud:
    """Chain ICP registration and merge clouds into the first frame's coordinates."""
    if not use_icp:
        return fuse_point_clouds_concat(clouds)
    if not clouds:
        return o3d.geometry.PointCloud()
    if len(clouds) == 1:
        return clouds[0]

    merged = clouds[0]
    for i in range(1, len(clouds)):
        transform = register_frame_to_reference(clouds[i], merged)
        aligned = o3d.geometry.PointCloud(clouds[i])
        aligned.transform(transform)
        merged += aligned
        merged = merged.voxel_down_sample(VOXEL_SIZE_M)

    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE_M * 4, max_nn=30
        )
    )
    return merged
