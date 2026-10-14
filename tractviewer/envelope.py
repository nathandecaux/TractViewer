"""Envelopes: convex hull of any dataset, voxelized envelope of a tractogram.

Used by `style='envelope'` (see tractviewer.scene).
"""
from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pyvista as pv

DataInput = Union[str, Path, pv.DataSet]


def _load_any(obj: DataInput) -> pv.DataSet:
    if isinstance(obj, pv.DataSet):
        return obj
    return pv.read(str(obj))


def minimal_envelope(
    obj: DataInput,
    method: str = "convex",
    alpha: Optional[float] = None,
    clean: bool = True
) -> pv.PolyData:
    """
    Compute the smallest envelope around the dataset (convex hull).

    Parameters:
      obj: path or PyVista DataSet.
      method: 'convex' (default), reserved for future extensions.
      alpha: if given (>0), try an alpha shape (possibly less convex envelope);
             otherwise strict convex hull.
      clean: remove duplicated points / degenerate geometry first.

    Returns:
      pv.PolyData of the closed envelope surface.

    Notes:
      - Uses scipy.spatial.ConvexHull when available (direct mesh).
      - Otherwise falls back to Delaunay 3D and extracts the outer surface.
      - Alpha shape: delaunay_3d(alpha=alpha) (approximately concave).
    """
    ds = _load_any(obj)
    try:
        geom = ds.extract_surface()
    except Exception:
        geom = pv.wrap(ds)  # raw fallback
    if clean:
        try:
            geom = geom.clean()
        except Exception:
            pass

    pts = np.asarray(geom.points)
    if pts.shape[0] < 4:
        # Not enough points for a 3D hull: return as is
        poly = pv.PolyData(pts)
        poly.field_data["envelope_mode"] = np.array(["degenerate"], dtype=object)
        return poly

    # scipy first (more minimal mesh)
    if method == "convex" and alpha is None:
        try:
            from scipy.spatial import ConvexHull  # type: ignore
            hull = ConvexHull(pts)
            faces = hull.simplices  # (n_tri, 3)
            # VTK face array: [3, i0, i1, i2] repeated
            n_tri = faces.shape[0]
            face_array = np.hstack(
                [np.full((n_tri, 1), 3, dtype=np.int64), faces.astype(np.int64)]
            ).ravel()
            poly = pv.PolyData(pts, face_array)
            poly.field_data["envelope_mode"] = np.array(["scipy_convex"], dtype=object)
            return poly
        except Exception:
            pass  # Delaunay fallback

    # Fallback / alpha shape via Delaunay 3D
    try:
        cloud = pv.PolyData(pts)
        if alpha is not None and alpha > 0:
            vol = cloud.delaunay_3d(alpha=alpha)
        else:
            vol = cloud.delaunay_3d()
        hull_surf = vol.extract_surface()
        if clean:
            hull_surf = hull_surf.clean()
        hull_surf.field_data["envelope_mode"] = np.array(
            ["delaunay_alpha" if alpha else "delaunay_convex"], dtype=object
        )
        if alpha:
            hull_surf.field_data["alpha"] = np.array([alpha], dtype=float)
        return hull_surf
    except Exception as e:
        # Last resort: axis-aligned bounding box
        mins = pts.min(0)
        maxs = pts.max(0)
        bounds_points = np.array([
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ])
        # Box faces (12 triangles)
        quads = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        tri_list = []
        for a, b, c, d in quads:
            tri_list.append((a, b, c))
            tri_list.append((a, c, d))
        tris = np.array(tri_list)
        faces = np.hstack([np.full((tris.shape[0], 1), 3), tris]).ravel()
        poly = pv.PolyData(bounds_points, faces)
        poly.field_data["envelope_mode"] = np.array(["aabb_fallback"], dtype=object)
        poly.field_data["error"] = np.array([str(e)], dtype=object)
        return poly


def _cell_to_point_data_array(ds: pv.PolyData, array_name: str) -> np.ndarray:
    """Propagate a cell_data array to points with PyVista, with a manual fallback."""
    try:
        tmp = ds.cell_data_to_point_data()
        if array_name in tmp.point_data.keys():
            return np.asarray(tmp.point_data[array_name])
    except Exception:
        pass
    # Manual fallback: walk the cells
    cell_values = np.asarray(ds.cell_data[array_name])
    point_values = np.zeros(ds.n_points, dtype=cell_values.dtype)
    offset = 0
    for i in range(ds.n_cells):
        cell = ds.get_cell(i)
        n = cell.n_points
        if offset + n <= len(point_values):
            point_values[offset:offset + n] = cell_values[i]
        offset += n
    return point_values


def _voxelize_and_contour(
    pts: np.ndarray,
    voxel_size: float,
    smooth_iterations: int,
) -> Optional[pv.PolyData]:
    """Voxelize a point cloud and run marching cubes.

    Returns None if no surface could be computed.
    """
    if pts.shape[0] < 2:
        return None

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = maxs - mins
    max_span = float(span.max())
    if max_span < 1e-6:
        return None

    dims = np.ceil(span / voxel_size).astype(int) + 3  # padding around
    # Memory guard: cap the grid at 600^3
    if np.any(dims > 600):
        voxel_size = max_span / 200.0
        dims = np.ceil(span / voxel_size).astype(int) + 3

    origin = mins - voxel_size  # 1 voxel of padding

    # Voxel index of each point
    idx = np.floor((pts - origin) / voxel_size).astype(int)
    idx = np.clip(idx, 0, dims - 1)

    # Binary grid
    grid_data = np.zeros(dims, dtype=np.float32)
    grid_data[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    # Morphological dilation to fill the gaps between streamlines
    try:
        from scipy.ndimage import binary_dilation
        grid_data = binary_dilation(grid_data.astype(bool), iterations=2).astype(np.float32)
    except ImportError:
        pass

    image = pv.ImageData()
    image.origin = tuple(origin.tolist())
    image.spacing = (voxel_size, voxel_size, voxel_size)
    image.dimensions = tuple(dims.tolist())
    image.point_data["mask"] = grid_data.ravel(order="F")

    try:
        surf = image.contour([0.5], scalars="mask", method="marching_cubes")
    except Exception:
        try:
            surf = image.contour([0.5], scalars="mask")
        except Exception:
            return None

    if surf is None or surf.n_points == 0:
        return None

    if smooth_iterations > 0:
        try:
            surf = surf.smooth(n_iter=smooth_iterations)
        except Exception:
            pass

    return surf


def tractogram_envelope(
    ds: pv.PolyData,
    display_array: Optional[str] = None,
    voxel_size: float = 1.0,
    smooth_iterations: int = 50,
) -> pv.PolyData:
    """
    Compute the envelope of a tractogram by voxelization + marching cubes.

    If display_array is given (point_data or cell_data), one envelope is computed
    per unique value of the array and that value is assigned as a scalar on the
    resulting surface (allows coloring by label).

    Parameters:
      ds: tractogram PolyData.
      display_array: name of the label array for per-label envelopes.
      voxel_size: voxel size in mm (point units). Default: 1.0 mm.
      smooth_iterations: Laplacian smoothing after marching cubes (0 = disabled).

    Returns:
      pv.PolyData of the envelope surface.
    """
    pts = np.asarray(ds.points)
    if pts.shape[0] < 4:
        result = pv.PolyData(pts)
        result.field_data["envelope_mode"] = np.array(["degenerate"], dtype=object)
        return result

    # Per-point values if display_array is requested
    values = None
    if display_array is not None:
        if display_array in ds.point_data.keys():
            values = np.asarray(ds.point_data[display_array])
        elif display_array in ds.cell_data.keys():
            values = _cell_to_point_data_array(ds, display_array)

    if values is not None:
        unique_labels = np.unique(values)
        surfaces = []
        for label in unique_labels:
            mask = values == label
            label_pts = pts[mask]
            if label_pts.shape[0] < 4:
                continue
            surf = _voxelize_and_contour(label_pts, voxel_size, smooth_iterations)
            if surf is not None and surf.n_points > 0:
                surf.point_data[display_array] = np.full(
                    surf.n_points, label, dtype=values.dtype
                )
                surfaces.append(surf)

        if surfaces:
            if len(surfaces) == 1:
                combined = surfaces[0]
            else:
                try:
                    combined = pv.merge(surfaces)
                    # Make sure the array survived the merge
                    if display_array not in combined.array_names:
                        raise ValueError("array lost after merge")
                except Exception:
                    # Fallback: manual concatenation
                    all_pts = np.vstack([s.points for s in surfaces])
                    all_vals = np.concatenate([
                        np.asarray(s.point_data[display_array]) for s in surfaces
                    ])
                    # Rebuild faces with point offsets
                    all_faces_list = []
                    pt_offset = 0
                    for s in surfaces:
                        if s.n_faces > 0:
                            faces = s.faces.reshape(-1, 4)  # [3, i0, i1, i2]
                            faces_shifted = faces.copy()
                            faces_shifted[:, 1:] += pt_offset
                            all_faces_list.append(faces_shifted.ravel())
                        pt_offset += s.n_points
                    if all_faces_list:
                        combined = pv.PolyData(all_pts, np.concatenate(all_faces_list))
                    else:
                        combined = pv.PolyData(all_pts)
                    combined.point_data[display_array] = all_vals
            combined.field_data["envelope_mode"] = np.array(
                ["tractogram_per_label"], dtype=object
            )
            return combined
        # Fallback: global envelope if no label produced a surface

    surf = _voxelize_and_contour(pts, voxel_size, smooth_iterations)
    if surf is None or surf.n_points == 0:
        return minimal_envelope(ds)
    surf.field_data["envelope_mode"] = np.array(
        ["tractogram_marching_cubes"], dtype=object
    )
    return surf


__all__ = ["minimal_envelope", "tractogram_envelope"]
