"""Isolate scan subject: remove dominant plane and background clusters."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from scanner.config import (
    CLUSTER_EPS_M,
    CLUSTER_MIN_POINTS,
    COMPACT_CLUSTER_EPS_M,
    COMPACT_CLUSTER_MAX_EXTENT_MM,
    PLANE_DISTANCE_M,
)


def remove_dominant_plane(
    cloud: o3d.geometry.PointCloud,
    *,
    distance: float = PLANE_DISTANCE_M,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) < 100:
        return cloud
    plane_model, inliers = cloud.segment_plane(distance, ransac_n, num_iterations)
    del plane_model
    return cloud.select_by_index(inliers, invert=True)


def select_compact_object_cluster(
    cloud: o3d.geometry.PointCloud,
    *,
    eps: float = COMPACT_CLUSTER_EPS_M,
    min_points: int = CLUSTER_MIN_POINTS,
    max_extent_mm: float = COMPACT_CLUSTER_MAX_EXTENT_MM,
) -> o3d.geometry.PointCloud:
    """Largest DBSCAN cluster under *max_extent_mm* (drops turntable/table slabs)."""
    if len(cloud.points) < min_points:
        return cloud

    labels = np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=50, print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        return cloud

    pts = np.asarray(cloud.points)
    best_label: int | None = None
    best_count = 0
    for label in range(labels.max() + 1):
        count = int((labels == label).sum())
        if count < min_points:
            continue
        cluster_pts = pts[labels == label]
        max_ext_mm = float(
            np.max(cluster_pts.max(axis=0) - cluster_pts.min(axis=0)) * 1000.0
        )
        if max_ext_mm > max_extent_mm:
            continue
        if count > best_count:
            best_count = count
            best_label = label

    if best_label is None:
        return cloud
    return cloud.select_by_index(np.where(labels == best_label)[0])


def keep_largest_cluster(
    cloud: o3d.geometry.PointCloud,
    *,
    eps: float = CLUSTER_EPS_M,
    min_points: int = CLUSTER_MIN_POINTS,
) -> o3d.geometry.PointCloud:
    if len(cloud.points) < min_points:
        return cloud

    labels = np.asarray(
        cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        return cloud

    counts = [(label, int((labels == label).sum())) for label in range(labels.max() + 1)]
    counts.sort(key=lambda item: item[1], reverse=True)
    best_label = counts[0][0]
    indices = np.where(labels == best_label)[0]
    return cloud.select_by_index(indices)


def isolate_object(
    cloud: o3d.geometry.PointCloud,
    *,
    compact: bool = False,
) -> o3d.geometry.PointCloud:
    """Table/plane removal then object cluster (compact or largest)."""
    filtered = remove_dominant_plane(cloud)
    if compact:
        filtered = select_compact_object_cluster(filtered)
    else:
        filtered = keep_largest_cluster(filtered)
    filtered, _ = filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return filtered
