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

def enveloppe_minimale(
    obj: DataInput,
    method: str = "convex",
    alpha: Optional[float] = None,
    clean: bool = True
) -> pv.PolyData:
    """
    Calcule la plus petite enveloppe entourant le dataset (convex hull).

    Paramètres:
      obj: chemin ou DataSet PyVista.
      method: 'convex' (par défaut) – réservé pour extensions futures.
      alpha: si fourni (>0), tente une alpha-shape (enveloppe potentiellement moins convexe);
             sinon enveloppe convexe stricte.
      clean: supprime points dupliqués / geo dégénérée avant calcul.

    Retour:
      pv.PolyData représentant la surface fermée de l’enveloppe.

    Notes:
      - Si scipy.spatial.ConvexHull est disponible, on l'utilise pour un maillage direct.
      - Sinon on fallback sur Delaunay 3D puis extraction de la surface externe.
      - Alpha-shape: on utilise delaunay_3d(alpha=alpha) (approx concave si support).
    """
    ds = _load_any(obj)
    try:
        geom = ds.extract_surface()
    except Exception:
        geom = pv.wrap(ds)  # fallback brut
    if clean:
        try:
            geom = geom.clean()
        except Exception:
            pass

    pts = np.asarray(geom.points)
    if pts.shape[0] < 4:
        # Pas assez de points pour un hull 3D – renvoyer tel quel
        poly = pv.PolyData(pts)
        poly.field_data["enveloppe_mode"] = np.array(["degenerate"], dtype=object)
        return poly

    # Tentative scipy (maillage plus minimal)
    if method == "convex" and alpha is None:
        try:
            from scipy.spatial import ConvexHull  # type: ignore
            hull = ConvexHull(pts)
            faces = hull.simplices  # (n_tri, 3)
            # Construire tableau VTK: [3, i0, i1, i2] répété
            n_tri = faces.shape[0]
            face_array = np.hstack(
                [np.full((n_tri, 1), 3, dtype=np.int64), faces.astype(np.int64)]
            ).ravel()
            poly = pv.PolyData(pts, face_array)
            poly.field_data["enveloppe_mode"] = np.array(["scipy_convex"], dtype=object)
            return poly
        except Exception:
            pass  # fallback Delaunay

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
        hull_surf.field_data["enveloppe_mode"] = np.array(
            ["delaunay_alpha" if alpha else "delaunay_convex"], dtype=object
        )
        if alpha:
            hull_surf.field_data["alpha"] = np.array([alpha], dtype=float)
        return hull_surf
    except Exception as e:
        # Dernier recours: bounding box (AABB)
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
        # Faces (12 triangles) de la box
        quads = [
            (0,1,2,3),
            (4,5,6,7),
            (0,1,5,4),
            (1,2,6,5),
            (2,3,7,6),
            (3,0,4,7),
        ]
        # Convertir en triangles
        tri_list = []
        for a,b,c,d in quads:
            tri_list.append((a,b,c))
            tri_list.append((a,c,d))
        tris = np.array(tri_list)
        faces = np.hstack([np.full((tris.shape[0],1),3), tris]).ravel()
        poly = pv.PolyData(bounds_points, faces)
        poly.field_data["enveloppe_mode"] = np.array(["aabb_fallback"], dtype=object)
        poly.field_data["error"] = np.array([str(e)], dtype=object)
        return poly

def _cell_to_point_data_array(ds: pv.PolyData, array_name: str) -> np.ndarray:
    """Propage un array cell_data vers les points via PyVista, avec fallback manuel."""
    try:
        tmp = ds.cell_data_to_point_data()
        if array_name in tmp.point_data.keys():
            return np.asarray(tmp.point_data[array_name])
    except Exception:
        pass
    # Fallback manuel: parcourir les cellules
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
    """Voxelise un nuage de points et applique marching cubes.

    Retourne None si la surface n'a pu être calculée.
    """
    if pts.shape[0] < 2:
        return None

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = maxs - mins
    max_span = float(span.max())
    if max_span < 1e-6:
        return None

    dims = np.ceil(span / voxel_size).astype(int) + 3  # +padding autour
    # Sécurité mémoire: limiter la grille à 600^3
    if np.any(dims > 600):
        voxel_size = max_span / 200.0
        dims = np.ceil(span / voxel_size).astype(int) + 3

    origin = mins - voxel_size  # 1 voxel de padding

    # Indices voxel pour chaque point
    idx = np.floor((pts - origin) / voxel_size).astype(int)
    idx = np.clip(idx, 0, dims - 1)

    # Grille binaire
    grid_data = np.zeros(dims, dtype=np.float32)
    grid_data[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    # Dilatation morphologique pour combler les trous entre streamlines
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

    # Marching cubes
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


def enveloppe_tractogram(
    ds: pv.PolyData,
    display_array: Optional[str] = None,
    voxel_size: float = 1.0,
    smooth_iterations: int = 50,
) -> pv.PolyData:
    """
    Calcule l'enveloppe d'un tractogramme par voxelisation + marching cubes.

    Si display_array est fourni (point_data ou cell_data), calcule une enveloppe
    par valeur unique de l'array et assigne la valeur comme scalaire sur la surface
    résultante (permet une coloration par label).

    Paramètres:
      ds: PolyData tractogramme.
      display_array: nom de l'array de labels pour calcul per-label.
      voxel_size: taille d'un voxel en mm (unités des points). Défaut: 1.0 mm.
      smooth_iterations: lissage Laplacien après marching cubes (0 = désactivé).

    Retour:
      pv.PolyData de la surface enveloppe.
    """
    pts = np.asarray(ds.points)
    if pts.shape[0] < 4:
        result = pv.PolyData(pts)
        result.field_data["enveloppe_mode"] = np.array(["degenerate"], dtype=object)
        return result

    # Récupérer les valeurs par point si display_array demandé
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
                    # Vérifier que l'array est préservé
                    if display_array not in combined.array_names:
                        raise ValueError("array perdu après merge")
                except Exception:
                    # Fallback: concaténation manuelle
                    all_pts = np.vstack([s.points for s in surfaces])
                    all_vals = np.concatenate([
                        np.asarray(s.point_data[display_array]) for s in surfaces
                    ])
                    # Reconstruire les faces avec offset
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
            combined.field_data["enveloppe_mode"] = np.array(
                ["tractogram_per_label"], dtype=object
            )
            return combined
        # Fallback: enveloppe globale si aucun label n'a produit de surface

    surf = _voxelize_and_contour(pts, voxel_size, smooth_iterations)
    if surf is None or surf.n_points == 0:
        return enveloppe_minimale(ds)
    surf.field_data["enveloppe_mode"] = np.array(
        ["tractogram_marching_cubes"], dtype=object
    )
    return surf


__all__ = ["enveloppe_minimale", "enveloppe_tractogram"]
