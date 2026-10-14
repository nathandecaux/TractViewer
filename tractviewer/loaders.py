"""Loading files into PyVista datasets.

Each format has its own function; `load` picks one from the file extension.
Specialized formats:
  - .tck / .trk : tractography -> PolyData (lines)
  - .nii / .nii.gz : NIfTI -> isosurface (marching cubes) or raw volume
  - .png : image -> ImageData (RGB colors as scalars)
Everything else is delegated to `pyvista.read`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import os
import warnings
import numpy as np
import pyvista as pv

DataInput = Union[str, Path, pv.DataSet]


def load(obj: DataInput, marching_cubes: bool = True, smooth_iterations: int = 0,
         iso_value: Optional[float] = None, coord_system: Optional[str] = None) -> pv.DataSet:
    """Load an object (path or already built dataset) as a PyVista dataset.

    `iso_value` / `coord_system` only apply to NIfTI files (see `load_nifti`).
    """
    if isinstance(obj, pv.DataSet):
        return obj
    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    double_suffix = "".join(path.suffixes[-2:]).lower()  # for .nii.gz
    if suffix in (".tck", ".trk"):
        return load_tractogram(path)
    if suffix == ".nii" or double_suffix == ".nii.gz":
        return load_nifti(path, marching_cubes=marching_cubes, smooth_iterations=smooth_iterations,
                          iso_value=iso_value, coord_system=coord_system)
    if suffix == ".png":
        return load_png(path)
    return pv.read(path, progress_bar=True)


def load_tractogram(path: Path) -> pv.PolyData:
    """Read a .tck/.trk file into a PolyData where each streamline is a polyline.

    Added arrays:
      - cell_data['streamline_id'], cell_data['length_mm']
      - point_data['point_index'] (local index along the streamline)
    """
    try:
        import nibabel as nib
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "nibabel is required to load tractography files (.tck/.trk): 'pip install nibabel'"
        ) from e

    streamlines = nib.streamlines.load(str(path)).streamlines

    points = []
    lines = []
    lengths = []
    stream_ids = []
    point_indices = []
    offset = 0
    for sid, sl in enumerate(streamlines):
        npts = sl.shape[0]
        if npts < 2:
            continue
        points.append(sl.astype(np.float32))
        lines.append(npts)
        lines.extend(range(offset, offset + npts))  # polyline VTK: [npts, p0, p1, ...]
        lengths.append(float(np.linalg.norm(np.diff(sl, axis=0), axis=1).sum()))
        stream_ids.append(sid)
        point_indices.extend(range(npts))
        offset += npts

    if not points:
        raise ValueError(f"No valid streamline in {path}.")

    poly = pv.PolyData()
    poly.points = np.vstack(points)
    poly.lines = np.array(lines, dtype=np.int32)
    poly.cell_data["streamline_id"] = np.array(stream_ids, dtype=np.int32)
    poly.cell_data["length_mm"] = np.array(lengths, dtype=np.float32)
    poly.point_data["point_index"] = np.array(point_indices, dtype=np.int32)
    return poly


def load_nifti(path: Path, marching_cubes: bool = True, smooth_iterations: int = 0,
               iso_value: Optional[float] = None, coord_system: Optional[str] = None) -> pv.DataSet:
    """Load a NIfTI as an isosurface (default) or as a raw volume (marching_cubes=False).

    The voxel->world affine is applied to the points and stored in field_data
    ('nifti_affine', 'nifti_dims', 'source_file') to project other datasets
    later and to re-extract the volume (slices of the vtk.js export).
    `iso_value`: iso value (default: TRACTVIEWER_NIFTI_ISO, else mid-range).
    `coord_system`: 'RAS' or 'LPS' (default: TRACTVIEWER_NIFTI_COORD, else RAS).
    """
    try:
        import nibabel as nib
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "nibabel is required to load NIfTI files (.nii/.nii.gz): 'pip install nibabel'"
        ) from e

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data, nan=0.0)
    affine = img.affine
    dims = data.shape

    grid = pv.ImageData()
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.dimensions = tuple(d if d > 1 else 1 for d in dims)
    grid.point_data["intensity"] = data.ravel(order="F")

    if not marching_cubes:
        grid.field_data["nifti_affine"] = affine.astype(np.float32).ravel()
        grid.field_data["nifti_dims"] = np.array(dims, dtype=np.int32)
        grid.field_data["source_file"] = np.array([str(path)], dtype=object)
        print(f"NIfTI volume loaded (no marching cubes) from {path.name}")
        return grid

    iso_value = _iso_value(data, iso_value)

    if data.max() - data.min() <= 1e-6:
        warnings.warn("Nearly constant NIfTI volume: no surface can be extracted.")
        surf = grid.extract_points(grid.points)
    else:
        try:
            surf = grid.contour([iso_value], scalars="intensity", method="marching_cubes")
            print(f"Isosurface extracted (marching cubes) at {iso_value:.3f} in {path.name}")
        except Exception as e:
            warnings.warn(f"Contour failed: {e}; falling back to extract_surface (coarse).")
            surf = grid.extract_surface()
        if smooth_iterations > 0:
            surf = surf.smooth(n_iter=smooth_iterations)
            print(f"Smoothing applied ({smooth_iterations} iterations) to {path.name}")

    _apply_affine(surf, affine)
    surf.field_data["nifti_affine"] = affine.astype(np.float32).ravel()
    surf.field_data["nifti_dims"] = np.array(dims, dtype=np.int32)
    surf.field_data["source_file"] = np.array([str(path)], dtype=object)
    surf.field_data["nifti_iso_value"] = np.array([iso_value], dtype=np.float32)
    _apply_coord_system(surf, coord_system)
    return surf


def _iso_value(data: np.ndarray, iso_value: Optional[float] = None) -> float:
    """Explicit iso, else TRACTVIEWER_NIFTI_ISO, else the middle of the value range."""
    if iso_value is not None:
        return float(iso_value)
    iso_env = os.environ.get("TRACTVIEWER_NIFTI_ISO")
    if iso_env is not None:
        try:
            return float(iso_env)
        except ValueError:
            pass
    return float((data.min() + data.max()) / 2.0)


def _apply_affine(ds: pv.DataSet, affine: np.ndarray) -> None:
    """Apply a 4x4 affine (rotation + translation + anisotropy) to the points."""
    pts = ds.points
    pts_h = np.c_[pts, np.ones(len(pts))]
    ds.points = (pts_h @ affine.T)[:, :3]


def _apply_coord_system(surf: pv.DataSet, coord_system: Optional[str] = None) -> None:
    """Convert RAS+ -> LPS when requested (or TRACTVIEWER_NIFTI_COORD=LPS) and tag field_data."""
    target = (coord_system or os.environ.get("TRACTVIEWER_NIFTI_COORD", "RAS")).upper()
    if target == "LPS":
        surf.points[:, 0] *= -1.0  # R -> L
        surf.points[:, 1] *= -1.0  # A -> P
        surf.field_data["coord_system"] = np.array(["LPS"], dtype=object)
    else:
        surf.field_data["coord_system"] = np.array(["RAS"], dtype=object)


def stem(path: Union[str, Path]) -> str:
    """File name without its known extension(s): 'anat.nii.gz' -> 'anat'."""
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def load_png(path: Path) -> pv.ImageData:
    """Load a PNG as ImageData. RGB colors are encoded in 'rgb_encoded', plus
    separate arrays (red/green/blue/alpha/luminance) and 'is_transparent'."""
    img_array, has_alpha = _read_image_rgba(path)
    height, width = img_array.shape[:2]

    grid = pv.ImageData()
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.dimensions = (width, height, 1)

    # VTK expects (x, y) order: transpose, then flatten.
    rgba = np.transpose(img_array, (1, 0, 2)).reshape(-1, 4)
    rgb = rgba[:, :3]
    alpha = rgba[:, 3]

    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    rgb_encoded = (
        (rgb[:, 0].astype(np.uint32) << 16)
        + (rgb[:, 1].astype(np.uint32) << 8)
        + rgb[:, 2].astype(np.uint32)
    )

    grid.point_data["luminance"] = luminance.astype(np.float32)
    grid.point_data["rgb_encoded"] = rgb_encoded.astype(np.float32)
    grid.point_data["red"] = rgb[:, 0].astype(np.float32)
    grid.point_data["green"] = rgb[:, 1].astype(np.float32)
    grid.point_data["blue"] = rgb[:, 2].astype(np.float32)
    grid.point_data["alpha"] = alpha.astype(np.float32)
    grid.point_data["is_transparent"] = (alpha < 10).astype(np.uint8)  # transparency threshold

    grid.field_data["image_width"] = np.array([width], dtype=np.int32)
    grid.field_data["image_height"] = np.array([height], dtype=np.int32)
    grid.field_data["has_transparency"] = np.array([has_alpha], dtype=np.bool_)
    grid.field_data["source_file"] = np.array([str(path)], dtype=object)
    return grid


def _read_image_rgba(path: Path):
    """Return (HxWx4 uint8 array, has_alpha). Uses PIL, falls back to imageio."""
    try:
        from PIL import Image
        img = Image.open(str(path))
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        return np.array(img.convert("RGBA")), has_alpha
    except Exception:
        import imageio.v2 as imageio
        arr = imageio.imread(str(path))
        if arr.ndim == 2:  # grayscale -> RGBA
            arr = np.stack([arr] * 3 + [np.full_like(arr, 255)], axis=-1)
            return arr, False
        if arr.shape[2] == 3:  # RGB -> RGBA
            alpha = np.full(arr.shape[:2], 255, dtype=arr.dtype)
            return np.concatenate([arr, alpha[..., None]], axis=-1), False
        return arr, True
