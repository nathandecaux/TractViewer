"""Real spherical harmonics (MRtrix3 basis) and icospheres, numpy only.

MRtrix3 basis (dipy: ``tournier07``, legacy=False), even orders l = 0, 2, ..., lmax,
m = -l..l, coefficient index l(l+1)/2 + m:

  m < 0 : sqrt(2) * N_l^|m| P_l^|m|(cos theta) sin(|m| phi)
  m = 0 :           N_l^0   P_l^0(cos theta)
  m > 0 : sqrt(2) * N_l^m   P_l^m(cos theta)   cos(m phi)

with the Condon-Shortley phase in P_l^m and N_l^m = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!).
"""
from __future__ import annotations
import math
from typing import Tuple

import numpy as np

SH_SIZES = {(l + 1) * (l + 2) // 2: l for l in range(0, 21, 2)}  # n coefficients -> lmax


def lmax_from_ncoeffs(n: int) -> int:
    if n not in SH_SIZES:
        raise ValueError(f"{n} coefficients is not an even-order SH series (6, 15, 28, 45, 66...).")
    return SH_SIZES[n]


def _legendre(lmax: int, x: np.ndarray) -> dict:
    """Associated Legendre P_l^m(x), m >= 0, with the Condon-Shortley phase -> {(l, m): array}."""
    p = {(0, 0): np.ones_like(x)}
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    for m in range(1, lmax + 1):  # P_m^m = -(2m-1) s P_{m-1}^{m-1}
        p[(m, m)] = -(2 * m - 1) * s * p[(m - 1, m - 1)]
    for m in range(0, lmax):
        p[(m + 1, m)] = (2 * m + 1) * x * p[(m, m)]
        for l in range(m + 2, lmax + 1):
            p[(l, m)] = ((2 * l - 1) * x * p[(l - 1, m)] - (l + m - 1) * p[(l - 2, m)]) / (l - m)
    return p


def sh_basis(lmax: int, directions: np.ndarray) -> np.ndarray:
    """(n_directions, n_coeffs) matrix of the MRtrix3 basis at unit `directions` (x, y, z)."""
    d = np.asarray(directions, dtype=np.float64)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0])
    p = _legendre(lmax, np.cos(theta))
    cols = []
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - am) / math.factorial(l + am))
            base = norm * p[(l, am)]
            if m < 0:
                cols.append(math.sqrt(2.0) * base * np.sin(am * phi))
            elif m == 0:
                cols.append(base)
            else:
                cols.append(math.sqrt(2.0) * base * np.cos(am * phi))
    return np.stack(cols, axis=1)


def icosphere(subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit icosphere -> (vertices (n, 3) float64, triangles (m, 3) int64), outward orientation."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                v = verts[a] + verts[b]
                verts.append(v / np.linalg.norm(v))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces, dtype=np.int64)
