"""Glyph payload of a diffusion grid for the vtk.js page.

The page builds the glyphs of each 2D view itself (no server round trip when the slice
moves): it receives, for every voxel of the model on a grid aligned with the world axes
(the grid of the slices, nearest-neighbour resampling of oblique images):

  - ellipsoids (DTI, MCM): one entry per tensor; `matrix` (n, 9) maps the unit sphere onto
    the ellipsoid (largest semi-axis = 1), `direction` (n, 3) = principal direction (unit),
    `color` (n, 3) uint8 = |principal direction|, `amp` = FA (DTI) / compartment weight (MCM),
    `compartment` (n,) = index of the tensor compartment. The page uses `direction` to project
    the model on a tractogram (compartment best aligned with the streamlines, per voxel).
  - sh (ODF): `coeffs` (n, n_coeffs) and, per sphere resolution, the basis matrix evaluated at
    one vertex of each antipodal pair (`basis` (n_half, n_coeffs), `half` (n_vertices,) = row of
    each vertex: even SH series are antipodally symmetric, half the work); `amp` = l=0
    coefficient / its 99th percentile.

`spheres` lists the sphere resolutions, coarse to fine: the page picks one from the size of
the glyphs on screen (level of detail) and only builds the glyphs in view, so the cost follows
the pixels shown rather than the voxels of the slice.

All vectors are in the display frame of the viewer (LPS or RAS).
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from tractviewer.diffusion import sh as sh_mod
from tractviewer.diffusion.models import model_array, sym_matrix

FLIP_LPS = np.diag([-1.0, -1.0, 1.0])
# Sphere subdivisions per resolution, coarse to fine (42, 162, 642 vertices).
SPHERES = {"ellipsoid": (1, 2), "sh": (1, 2, 3)}
AMP_LABELS = {"dti": "FA", "odf": "AFD (l=0, normalized)", "mcm": "Compartment weight"}


def world_index_map(shape: Tuple[int, int, int], affine: np.ndarray):
    """Nearest voxel of each node of a grid aligned with the world axes (same grid as
    vtkjs._align_to_world) -> (index map (dims) int64, -1 outside; origin; spacing)."""
    m = affine[:3, :3]
    nonzero = np.abs(m) > 1e-6 * np.abs(m).max()
    src = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    if (nonzero.sum(axis=0) == 1).all() and (nonzero.sum(axis=1) == 1).all():
        perm = [int(np.argmax(np.abs(m[i]))) for i in range(3)]
        out = np.transpose(src, perm)
        steps = np.array([m[i, perm[i]] for i in range(3)])
        origin = affine[:3, 3].copy()
        for i in range(3):
            if steps[i] < 0:
                out = np.flip(out, axis=i)
                origin[i] += steps[i] * (out.shape[i] - 1)
        return np.ascontiguousarray(out), origin, np.abs(steps)

    step = float(np.linalg.norm(m, axis=0).min())
    corners = np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1)
                        for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
    world = (corners @ affine.T)[:, :3]
    origin, top = world.min(axis=0), world.max(axis=0)
    dims = np.floor((top - origin) / step).astype(int) + 1
    inv = np.linalg.inv(affine)
    xs, ys = (origin[a] + step * np.arange(dims[a]) for a in (0, 1))
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    out = np.full(tuple(dims), -1, dtype=np.int64)
    for k in range(dims[2]):
        pts = np.stack([gx, gy, np.full_like(gx, origin[2] + step * k), np.ones_like(gx)], axis=-1)
        idx = np.rint(pts @ inv.T)[..., :3].astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=-1)
        out[..., k][ok] = src[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out, origin, np.full(3, step)


def glyph_payload(ds, coord_system: str = "LPS") -> Dict:
    """Glyph data of a diffusion grid (see module doc); arrays as numpy (serialized by vtkjs)."""
    kind = str(ds.field_data["diffusion_model"][0])
    affine = np.asarray(ds.field_data["nifti_affine"], dtype=np.float64).reshape(4, 4)
    flip = FLIP_LPS if coord_system.upper() == "LPS" else np.eye(3)
    affine = np.diag([*np.diag(flip), 1.0]) @ affine
    shape = tuple(int(d) for d in ds.dimensions)
    index, origin, spacing = world_index_map(shape, affine)
    inside = index >= 0
    ijk_all = np.argwhere(inside)            # world grid nodes (C order, like `index`)
    src_all = index[inside]                  # flat C index of their voxel

    def voxel_values(name):
        values = model_array(ds, name)
        return values.reshape((-1,) + values.shape[3:])[src_all]

    out = {
        "model": kind, "ampLabel": AMP_LABELS[kind],
        "origin": [float(v) for v in origin], "spacing": [float(v) for v in spacing],
        "dims": [int(d) for d in index.shape],
    }
    if kind in ("dti", "mcm"):
        if kind == "dti":
            groups = [("tensor", None)]
        else:
            n = int(np.asarray(ds.field_data["mcm_compartments"]).ravel()[0])
            groups = [(f"tensor_{k}", f"weight_{k}") for k in range(n)]
        ijk, matrix, direction, color, amp, comp = [], [], [], [], [], []
        for k, (tname, wname) in enumerate(groups):
            t = voxel_values(tname)
            keep = np.abs(t).sum(axis=1) > 0
            if wname is not None:
                keep &= voxel_values(wname) > 1e-3
            t = t[keep]
            m, d, c, fa = _ellipsoids(t, flip)
            a = fa if wname is None else voxel_values(wname)[keep].astype(np.float32)
            ok = np.isfinite(m).all(axis=(1, 2))
            ijk.append(ijk_all[keep][ok])
            matrix.append(m[ok].reshape(-1, 9))
            direction.append(d[ok])
            color.append(c[ok])
            amp.append(a[ok])
            comp.append(np.full(int(ok.sum()), k, dtype=np.uint8))
        out.update(
            kind="ellipsoid", ijk=np.concatenate(ijk).astype(np.uint16),
            matrix=np.concatenate(matrix).astype(np.float32),
            direction=np.concatenate(direction).astype(np.float32), color=np.concatenate(color).astype(np.uint8),
            amp=np.concatenate(amp).astype(np.float32), compartment=np.concatenate(comp),
        )
    else:
        sh = model_array(ds, "sh")
        sh = sh.reshape(-1, sh.shape[-1])
        keep = sh[src_all, 0] > 0  # filter before gathering all the coefficients (memory)
        coeffs, ijk = sh[src_all[keep]].astype(np.float32), ijk_all[keep]
        scale = float(np.percentile(coeffs[:, 0], 99)) if len(coeffs) else 1.0
        out.update(
            kind="sh", ijk=ijk.astype(np.uint16), coeffs=coeffs, nCoeffs=int(coeffs.shape[1]),
            amp=np.clip(coeffs[:, 0] / (scale or 1.0), 0.0, 1.0).astype(np.float32),
        )
    out["spheres"] = _spheres(out["kind"], flip, out.get("nCoeffs"))
    out["count"] = int(len(out["amp"]))
    return out


def _ellipsoids(t6: np.ndarray, flip: np.ndarray):
    """World RAS tensors (n, 6) -> (unit sphere -> ellipsoid matrices in the display frame,
    principal direction, its RGB, FA)."""
    d = flip @ sym_matrix(t6.astype(np.float64), "mrtrix") @ flip
    w, v = np.linalg.eigh(d)
    w = np.clip(w, 0.0, None)
    top = w[:, 2:3]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(top > 0, np.maximum(w / top, 0.05), 1.0)
        mean = w.mean(1, keepdims=True)
        fa = np.sqrt(1.5) * np.sqrt(((w - mean) ** 2).sum(1)) / np.sqrt((w ** 2).sum(1))
    matrix = np.einsum("nij,nj,nkj->nik", v, rel, v)
    color = np.rint(np.abs(v[:, :, 2]) * 255)
    return matrix, v[:, :, 2], color, np.nan_to_num(fa).astype(np.float32)


def _spheres(kind: str, flip: np.ndarray, n_coeffs=None):
    out = []
    for sub in SPHERES[kind]:
        verts, faces = sh_mod.icosphere(sub)
        sphere = {"vertices": verts.astype(np.float32), "faces": faces.astype(np.uint32).ravel()}
        if kind == "sh":
            rows, half = antipodal_half(verts)
            # Vertices are in the display frame; the SH coefficients in the RAS frame.
            basis = sh_mod.sh_basis(sh_mod.lmax_from_ncoeffs(n_coeffs), verts[rows] @ flip.T)
            sphere["basis"] = basis.astype(np.float32).ravel()
            sphere["half"] = half.astype(np.uint32)
        out.append(sphere)
    return out


def antipodal_half(verts: np.ndarray):
    """Vertices of a centrally symmetric sphere -> (one vertex index per antipodal pair (m,),
    row in it of every vertex (n,))."""
    key = lambda v: tuple(np.rint(v * 1e6).astype(np.int64))  # noqa: E731
    index = {key(v): i for i, v in enumerate(verts)}
    rows, half = [], np.empty(len(verts), dtype=np.int64)
    for i, v in enumerate(verts):
        j = index[key(-v)]
        if j >= i:  # first of the pair: new row
            half[i] = len(rows)
            rows.append(i)
        else:
            half[i] = half[j]
    return np.array(rows), half
