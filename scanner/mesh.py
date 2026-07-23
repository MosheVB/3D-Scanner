"""Mesh reconstruction and Fusion-friendly export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from scanner.config import POISSON_DEPTH


def _clip_mesh_to_cloud(
    mesh: o3d.geometry.TriangleMesh,
    cloud: o3d.geometry.PointCloud,
    max_dist_m: float,
) -> o3d.geometry.TriangleMesh:
    """Drop mesh vertices farther than *max_dist_m* from any input point.

    Poisson closes open regions (e.g. an unseen reflective top face) by inventing
    a billowing surface. Removing vertices with no nearby data keeps only the
    genuinely observed surface.
    """
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        return mesh
    kdt = o3d.geometry.KDTreeFlann(cloud)
    far = np.zeros(len(verts), dtype=bool)
    for i in range(len(verts)):
        _, _, d2 = kdt.search_knn_vector_3d(verts[i], 1)
        far[i] = bool(d2) and (float(d2[0]) ** 0.5 > max_dist_m)
    mesh.remove_vertices_by_mask(far)
    return mesh


def pointcloud_to_mesh(
    cloud: o3d.geometry.PointCloud,
    *,
    depth: int = POISSON_DEPTH,
    density_quantile: float = 0.05,
    keep_largest: bool = True,
    smooth_iters: int = 0,
    clip_to_cloud_max_dist: float | None = None,
) -> o3d.geometry.TriangleMesh:
    if len(cloud.points) < 500:
        raise ValueError(f"Too few points for meshing ({len(cloud.points)}). Capture more frames.")

    working = cloud
    if not working.has_normals():
        working.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
    working.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        working, depth=depth
    )
    densities = np.asarray(densities)
    if densities.size:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    if clip_to_cloud_max_dist is not None:
        mesh = _clip_mesh_to_cloud(mesh, working, clip_to_cloud_max_dist)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if keep_largest and len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if counts.size:
            keep = int(np.argmax(counts))
            mesh.remove_triangles_by_mask(labels != keep)
            mesh.remove_unreferenced_vertices()

    if smooth_iters > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)

    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh, print_progress=False)
    elif suffix in (".obj", ".ply"):
        o3d.io.write_triangle_mesh(str(path), mesh, print_progress=False)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix} (use .obj, .stl, or .ply)")
