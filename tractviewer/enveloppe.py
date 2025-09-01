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

__all__ = ["enveloppe_minimale"]
