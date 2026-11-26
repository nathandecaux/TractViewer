"""Diffusion models -> pv.ImageData on the voxel grid.

The returned grid looks like a raw NIfTI volume (point_data['intensity'] = scalar map,
field_data 'nifti_affine' = voxel -> RAS), so slices, volume rendering and the
tractogram reprojection work unchanged. field_data['diffusion_model'] says which
model arrays it carries (point_data, x fastest, *world RAS frame*):

  - 'dti' : 'tensor' (n, 6) xx, yy, zz, xy, xz, yz; intensity = FA
  - 'odf' : 'sh' (n, n_coeffs), MRtrix3 basis; intensity = l=0 coefficient
  - 'mcm' : 'tensor_<k>' (n, 6) + 'weight_<k>' (n,) per anisotropic compartment,
            'weight_<Type>' for the isotropic ones; intensity = anisotropic fraction
            (all of them can be shown: see scalar_maps)

Conventions (checked against the FOD peaks of a same subject):
  - MRtrix3 (dwi2tensor, dwi2fod): tensor xx, yy, zz, xy, xz, yz / SH in the world RAS frame.
  - Anima (DTI = 5D NIfTI, MCM compartments): tensor xx, xy, yy, xz, yz, zz (ITK lower
    triangle) in the *voxel index* frame: D_world = R D R^T, R = rotation of the affine.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyvista as pv

from tractviewer.diffusion import sh as sh_mod

MODELS = ("dti", "odf", "mcm")
ANISOTROPIC_TYPES = ("Tensor", "Stick", "Zeppelin")


# ------------------------------
# Tensor maths
# ------------------------------
def sym_matrix(t: np.ndarray, order: str) -> np.ndarray:
    """(..., 6) -> (..., 3, 3). order 'anima' = xx, xy, yy, xz, yz, zz; 'mrtrix' = xx, yy, zz, xy, xz, yz."""
    if order == "anima":
        xx, xy, yy, xz, yz, zz = np.moveaxis(t, -1, 0)
    else:
        xx, yy, zz, xy, xz, yz = np.moveaxis(t, -1, 0)
    return np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1),
                     np.stack([xz, yz, zz], -1)], -2)


def sym_vector(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6) xx, yy, zz, xy, xz, yz."""
    return np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]], -1)


def rotation_of(affine: np.ndarray) -> np.ndarray:
    """Rotation (with axis flips) of a voxel -> world affine: columns normalized."""
    m = np.asarray(affine, dtype=np.float64)[:3, :3]
    return m / np.linalg.norm(m, axis=0)


def to_world(t6: np.ndarray, order: str, rotation: Optional[np.ndarray]) -> np.ndarray:
    """Tensors (..., 6) in their file convention -> (..., 6) world RAS, mrtrix order."""
    m = sym_matrix(np.asarray(t6, dtype=np.float64), order)
    if rotation is not None:
        m = rotation @ m @ rotation.T
    return sym_vector(m).astype(np.float32)


def fractional_anisotropy(t6: np.ndarray) -> np.ndarray:
    """FA of (..., 6) tensors (mrtrix order); 0 where the tensor is null."""
    w = np.linalg.eigvalsh(sym_matrix(np.asarray(t6, dtype=np.float64), "mrtrix"))
    mean = w.mean(-1, keepdims=True)
    num = np.sqrt(((w - mean) ** 2).sum(-1))
    den = np.sqrt((w ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.sqrt(1.5) * num / den
    return np.clip(np.nan_to_num(fa), 0.0, 1.0).astype(np.float32)


# ------------------------------
# Detection / loading
# ------------------------------
def detect_nifti_model(img, model: Optional[str] = None) -> Optional[str]:
    """'dti' | 'odf' | None (plain volume) for a nibabel image; `model` forces the answer."""
    if model is not None:
        model = str(model).lower()
        if model in ("scalar", "none", "volume"):
            return None
        if model not in ("dti", "odf"):
            raise ValueError(f"Unknown diffusion model '{model}' (dti, odf, scalar).")
        return model
    shape = img.shape
    if len(shape) == 5 and shape[3] == 1 and shape[4] == 6:
        return "dti"  # ITK/Anima symmetric matrix image
    if len(shape) != 4:
        return None
    n = shape[3]
    if n == 6:
        # MRtrix tensor: diffusivities (~1e-3 mm²/s), unlike a 6-volume DWI.
        sample = np.asarray(img.dataobj[..., :3], dtype=np.float32)
        return "dti" if np.nanmax(np.abs(sample)) < 0.1 else None
    if n in sh_mod.SH_SIZES:
        # SH series: signed coefficients beyond l=0 (a DWI is non-negative).
        sample = np.asarray(img.dataobj[..., 1:min(n, 6)], dtype=np.float32)
        return "odf" if (sample < 0).mean() > 0.01 else None
    return None


def load_diffusion_nifti(path: Union[str, Path], model: Optional[str] = None) -> Optional[pv.ImageData]:
    """Diffusion grid of a NIfTI (DTI MRtrix/Anima, ODF MRtrix), or None if it is a plain volume."""
    import nibabel as nib

    img = nib.load(str(path))
    kind = detect_nifti_model(img, model)
    if kind is None:
        return None
    affine = np.asarray(img.affine, dtype=np.float64)
    data = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    if kind == "dti":
        if data.ndim == 5:  # Anima: (X, Y, Z, 1, 6), voxel frame
            tensor = to_world(data[:, :, :, 0, :], "anima", rotation_of(affine))
        elif data.ndim == 4 and data.shape[3] == 6:  # MRtrix: world frame
            tensor = to_world(data, "mrtrix", None)
        else:
            raise ValueError(f"{Path(path).name}: shape {data.shape} is not a tensor image.")
        grid = _grid(tensor.shape[:3], affine, path)
        grid.point_data["intensity"] = _flat(fractional_anisotropy(tensor))
        grid.point_data["tensor"] = _flat(tensor)
    else:
        if data.ndim != 4:
            raise ValueError(f"{Path(path).name}: shape {data.shape} is not an SH image.")
        sh_mod.lmax_from_ncoeffs(data.shape[3])  # validates the size
        grid = _grid(data.shape[:3], affine, path)
        grid.point_data["intensity"] = _flat(data[..., 0])
        grid.point_data["sh"] = _flat(data)
    grid.field_data["diffusion_model"] = np.array([kind], dtype=object)
    return grid


def load_mcm(path: Union[str, Path]) -> pv.ImageData:
    """Anima .mcm/.mcmx -> grid with the anisotropic compartments as world-frame tensors."""
    from tractviewer.diffusion.io import read_mcm

    mcm = read_mcm(path)
    rotation = rotation_of(mcm.affine)
    grid = _grid(mcm.shape, mcm.affine, path)
    aniso_weight = np.zeros(mcm.shape, dtype=np.float32)
    k = 0
    for comp in mcm.compartments:
        weight = np.nan_to_num(comp.weight.astype(np.float32))
        if comp.type in ANISOTROPIC_TYPES and comp.data.ndim == 4 and comp.data.shape[3] == 6:
            tensor = to_world(np.nan_to_num(comp.data), "anima", rotation)
            grid.point_data[f"tensor_{k}"] = _flat(tensor)
            grid.point_data[f"weight_{k}"] = _flat(weight)
            aniso_weight += weight
            k += 1
        elif comp.data.ndim == 3:  # isotropic (FreeWater, IRWater, StationaryWater...)
            name = f"weight_{comp.type}"
            if name in grid.point_data:
                grid.point_data[name] = grid.point_data[name] + _flat(weight)
            else:
                grid.point_data[name] = _flat(weight)
        else:
            warnings.warn(f"{Path(path).name}: compartment '{comp.type}' not displayed (unsupported).")
    if k == 0:
        raise ValueError(f"{Path(path).name}: no anisotropic (tensor) compartment.")
    grid.point_data["intensity"] = _flat(aniso_weight)
    grid.field_data["diffusion_model"] = np.array(["mcm"], dtype=object)
    grid.field_data["mcm_compartments"] = np.array([k], dtype=np.int32)
    return grid


def scalar_maps(ds: pv.ImageData) -> list:
    """(array name, label) of the scalar maps a diffusion grid can show, 'intensity' first.

    MCM: anisotropic fraction, weight of each isotropic compartment type (FreeWater, IRWater,
    StationaryWater...) and of each tensor compartment. Other models: the 'intensity' map only.
    """
    kind = str(ds.field_data["diffusion_model"][0])
    maps = [("intensity", {"dti": "FA", "odf": "l=0 coefficient", "mcm": "Anisotropic fraction"}[kind])]
    if kind == "mcm":
        names = list(ds.point_data.keys())
        maps += [(n, f"{n[len('weight_'):]} weight") for n in names
                 if n.startswith("weight_") and not n[len("weight_"):].isdigit()]
        n = int(np.asarray(ds.field_data["mcm_compartments"]).ravel()[0])
        maps += [(f"weight_{k}", f"Tensor {k + 1} weight") for k in range(n)]
    return maps


def _grid(shape, affine: np.ndarray, path) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = tuple(int(d) for d in shape[:3])
    grid.field_data["nifti_affine"] = np.asarray(affine, dtype=np.float32).ravel()
    grid.field_data["nifti_dims"] = np.array(shape[:3], dtype=np.int32)
    grid.field_data["source_file"] = np.array([str(path)], dtype=object)
    grid.field_data["coord_system"] = np.array(["RAS"], dtype=object)
    return grid


def _flat(data: np.ndarray) -> np.ndarray:
    """(X, Y, Z[, C]) -> (X*Y*Z[, C]) with x fastest (VTK point order)."""
    n = int(np.prod(data.shape[:3]))
    return data.reshape((n,) + data.shape[3:], order="F") if data.ndim > 3 else data.ravel(order="F")


def model_array(ds: pv.ImageData, name: str) -> np.ndarray:
    """point_data array back to (X, Y, Z[, C])."""
    values = np.asarray(ds.point_data[name])
    dims = tuple(ds.dimensions)
    return values.reshape(dims + values.shape[1:], order="F")
