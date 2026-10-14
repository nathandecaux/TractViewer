"""vtk.js export: serialized scene + HTML page, rendered in the browser (WebGL).

All the geometry and *all* arrays (point and cell) are sent to the browser:
switching array, colormap, range or opacity needs no server round trip. Each
NIfTI is also sent as a volume (slices).

  - build_scene(viewer) -> serializable dict (binary as base64)
  - serialize_dataset(viewer, ds, prm) -> (mesh layer | None, volume | None)
  - export_html(viewer, path) -> standalone HTML page (vtk.js loaded from a CDN)

The web server (file uploads) lives in tractviewer.web.server.

Colormaps are sampled from matplotlib on the Python side, so colors are
identical to the pyvista rendering.
"""
from __future__ import annotations
import base64
import contextlib
import json
import math
import sys
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv

from tractviewer import scene
from tractviewer.debug import warn_debug

VTKJS_URL = "https://cdn.jsdelivr.net/npm/vtk.js@37.1.1/vtk.js"
# Browser-side GIF encoder (ES module, ~9 kB), loaded only when a GIF is recorded.
GIFENC_URL = "https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js"

# Colormaps always offered by the page (on top of the datasets' own).
DEFAULT_CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet", "gray",
    "hot", "coolwarm", "RdBu_r", "Spectral", "tab10", "tab20",
]
CMAP_SAMPLES = 64
DEFAULT_CMAP = "viridis"
DEFAULT_MAX_VOXELS = 16_000_000


def build_scene(viewer, max_streamlines: Optional[int] = None,
                max_voxels: int = DEFAULT_MAX_VOXELS, server: bool = False) -> dict:
    """Serialize every dataset of the viewer (threshold/envelope applied) for vtk.js."""
    layers, volumes = [], []
    extra_cmaps: Dict[str, object] = {}
    for ds, prm in viewer._datasets:
        layer, volume = serialize_dataset(viewer, ds, prm, max_streamlines, max_voxels)
        if layer is not None:
            extra_cmaps.update(layer.pop("_cmap_def", {}))
            layers.append(layer)
        if volume is not None:
            volumes.append(volume)

    colormaps = sample_colormaps(extra_cmaps)
    for layer in layers:
        if layer.get("cmap") not in colormaps:
            layer["cmap"] = DEFAULT_CMAP

    return {
        "background": _to_hex(viewer.background, "#ffffff"),
        "coordSystem": getattr(viewer, "coord_system", "LPS"),
        "layers": layers,
        "volumes": volumes,
        "colormaps": colormaps,
        "server": server,
    }


def serialize_dataset(viewer, ds, prm, max_streamlines: Optional[int] = None,
                      max_voxels: int = DEFAULT_MAX_VOXELS) -> Tuple[Optional[dict], Optional[dict]]:
    """(mesh layer, volume) of a dataset; either may be None.

    The identifier is the dataset name (see `unique_name`).
    """
    name = str(prm.get("name"))
    layer = volume = None

    source = _nifti_source(ds)
    if source is not None:
        try:
            volume = nifti_volume(source, viewer.coord_system, max_voxels)
            volume.update(id=name, name=name, visible=prm.get("visible", True) is not False)
        except Exception:
            warn_debug(f"vtk.js export: volume of '{name}' skipped.", exc=sys.exc_info())

    if not (source is not None and isinstance(ds, pv.ImageData)):  # raw volume: no mesh
        try:
            mesh, display_array = _prepare_mesh(ds, prm, max_streamlines)
            layer = _serialize_layer(mesh, prm, display_array)
            layer["id"] = name
            cmap = prm.get("cmap")
            if isinstance(cmap, str):
                layer["cmap"] = cmap
                layer["_cmap_def"] = {cmap: cmap}
            elif isinstance(cmap, (list, tuple)) and cmap:
                cname = f"{name} (custom)"
                layer["cmap"] = cname
                layer["_cmap_def"] = {cname: list(cmap)}
        except Exception:
            warn_debug(f"vtk.js export: dataset '{name}' skipped.", exc=sys.exc_info())
    return layer, volume


def sample_colormaps(extra: Optional[Dict[str, object]] = None) -> Dict[str, list]:
    """Default colormaps + `extra` ({name: matplotlib name | list of colors})."""
    defs = {name: name for name in DEFAULT_CMAPS}
    defs.update(extra or {})
    out = {}
    for name, definition in defs.items():
        with contextlib.suppress(Exception):
            out[name] = _sample_cmap(definition)
    return out


def unique_name(viewer, base: str) -> str:
    names = set(viewer.names())
    if base not in names:
        return base
    i = 2
    while f"{base}_{i}" in names:
        i += 1
    return f"{base}_{i}"


def make_names_unique(viewer) -> None:
    """Rename duplicates ('a', 'a' -> 'a', 'a_2'): the name is the page-side identifier."""
    seen = set()
    for _, prm in viewer._datasets:
        name = str(prm.get("name"))
        if name in seen:
            i = 2
            while f"{name}_{i}" in seen:
                i += 1
            name = f"{name}_{i}"
        prm["name"] = name
        seen.add(name)


def render_html(scene_dict: dict, title: str = "TractViewer", vtkjs_url: str = VTKJS_URL) -> str:
    template = resources.files("tractviewer.web").joinpath("vtkjs_template.html").read_text("utf-8")
    # "</" escaped: the JSON is embedded in a <script> tag.
    scene_json = json.dumps(scene_dict, separators=(",", ":")).replace("</", "<\\/")
    return (template
            .replace("__TITLE__", _html_escape(title))
            .replace("__VTKJS_URL__", vtkjs_url)
            .replace("__GIFENC_URL__", GIFENC_URL)
            .replace("__SCENE_JSON__", scene_json))


def export_html(viewer, output_path: Union[str, Path], title: str = "TractViewer",
                max_streamlines: Optional[int] = None, max_voxels: int = DEFAULT_MAX_VOXELS,
                vtkjs_url: str = VTKJS_URL) -> str:
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_names_unique(viewer)
    scene_dict = build_scene(viewer, max_streamlines, max_voxels)
    output_path.write_text(render_html(scene_dict, title=title, vtkjs_url=vtkjs_url), encoding="utf-8")
    return str(output_path)


# ------------------------------
# Volumes NIfTI (coupes)
# ------------------------------
def _nifti_source(ds) -> Optional[str]:
    if "nifti_affine" not in ds.field_data or "source_file" not in ds.field_data:
        return None
    path = str(ds.field_data["source_file"][0])
    return path if Path(path).exists() else None


def nifti_volume(path: Union[str, Path], coord_system: str = "LPS",
                 max_voxels: int = DEFAULT_MAX_VOXELS) -> dict:
    """NIfTI volume on a grid aligned with the world axes (`coord_system` frame).

    Axis-aligned affine (common case, possibly permuted/flipped): exact voxel
    reordering. Oblique affine: nearest-neighbour resampling on an isotropic
    grid. Same coordinates as the isosurface built by loaders.load_nifti.
    """
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.nan_to_num(data, nan=0.0)
    affine = np.asarray(img.affine, dtype=float)
    if coord_system.upper() == "LPS":
        affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine

    out, origin, spacing = _align_to_world(data, affine)
    if out.size > max_voxels:
        f = math.ceil((out.size / max_voxels) ** (1 / 3))
        out = out[::f, ::f, ::f]
        spacing = spacing * f

    sample = out.ravel()[:: max(1, out.size // 1_000_000)]
    nonzero = sample[sample != 0]
    lo, hi = np.percentile(nonzero if nonzero.size else sample, [1, 99]).tolist()
    return {
        "dims": [int(d) for d in out.shape],
        "origin": [float(v) for v in origin],
        "spacing": [float(v) for v in spacing],
        "range": [float(out.min()), float(out.max())],
        "window": [float(lo), float(hi)],
        "data": _b64(out.ravel(order="F").astype(np.float32)),  # x fastest (VTK order)
    }


def _align_to_world(data: np.ndarray, affine: np.ndarray):
    """-> (volume indexed by world [x, y, z], origin, spacing)."""
    m = affine[:3, :3]
    nonzero = np.abs(m) > 1e-6 * np.abs(m).max()
    if (nonzero.sum(axis=0) == 1).all() and (nonzero.sum(axis=1) == 1).all():
        perm = [int(np.argmax(np.abs(m[i]))) for i in range(3)]  # voxel axis of each world axis
        out = np.transpose(data, perm)
        steps = np.array([m[i, perm[i]] for i in range(3)])
        origin = affine[:3, 3].copy()
        for i in range(3):
            if steps[i] < 0:
                out = np.flip(out, axis=i)
                origin[i] += steps[i] * (out.shape[i] - 1)
        return np.ascontiguousarray(out), origin, np.abs(steps)

    # Oblique: nearest neighbour on an isotropic grid covering the volume.
    step = float(np.linalg.norm(m, axis=0).min())
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world = (corners @ affine.T)[:, :3]
    origin, top = world.min(axis=0), world.max(axis=0)
    dims = np.floor((top - origin) / step).astype(int) + 1
    inv = np.linalg.inv(affine)
    xs, ys = (origin[a] + step * np.arange(dims[a]) for a in (0, 1))
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    out = np.zeros(tuple(dims), dtype=np.float32)
    for k in range(dims[2]):  # slice by slice: bounded memory
        pts = np.stack([gx, gy, np.full_like(gx, origin[2] + step * k), np.ones_like(gx)], axis=-1)
        idx = np.rint(pts @ inv.T)[..., :3].astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(data.shape)), axis=-1)
        out[..., k][ok] = data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out, origin, np.full(3, step)


# ------------------------------
# Mesh preparation
# ------------------------------
def _prepare_mesh(ds, prm, max_streamlines):
    """Apply threshold / envelope like scene.py, then convert to PolyData."""
    mesh = scene._apply_threshold(ds, prm)
    display_array = prm.get("display_array")
    if display_array and display_array not in mesh.array_names:
        display_array = None
    if prm.get("style") in ("envelope", "enveloppe"):
        mesh, display_array, _ = scene._apply_envelope(mesh, prm, display_array)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if max_streamlines and mesh.n_lines > max_streamlines and mesh.n_cells == mesh.n_lines:
        mesh = _subsample_lines(mesh, max_streamlines)
    if display_array is None and prm.get("color") is None:
        # Like pyvista: without array or color, color by the active scalars.
        display_array = mesh.active_scalars_name
    return mesh, display_array


def _subsample_lines(mesh: pv.PolyData, n: int) -> pv.PolyData:
    """Keep n randomly drawn streamlines (fixed seed: reproducible export)."""
    import vtk
    keep = np.sort(np.random.default_rng(0).choice(mesh.n_lines, size=n, replace=False))
    sub = mesh.extract_cells(keep)
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(sub)
    geom.Update()
    out = pv.wrap(geom.GetOutput())
    for store in (out.point_data, out.cell_data):
        for name in ("vtkOriginalPointIds", "vtkOriginalCellIds"):
            if name in store:
                del store[name]
    return out


# ------------------------------
# Serialization
# ------------------------------
def _serialize_layer(mesh: pv.PolyData, prm: dict, display_array: Optional[str]) -> dict:
    arrays = []
    for location, store in (("point", mesh.point_data), ("cell", mesh.cell_data)):
        for name in store.keys():
            arr = _serialize_array(name, location, np.asarray(store[name]))
            if arr is not None:
                arrays.append(arr)

    active = None
    if display_array:
        active = next((f"{a['location']}:{a['name']}" for a in arrays if a["name"] == display_array), None)
    clim = prm.get("clim")

    return {
        "name": str(prm.get("name")),
        "visible": prm.get("visible", True) is not False,
        "nPoints": int(mesh.n_points),
        "nCells": int(mesh.n_cells),
        "points": _b64(np.asarray(mesh.points, dtype=np.float32)),
        "verts": _b64_cells(mesh.verts),
        "lines": _b64_cells(mesh.lines),
        "polys": _b64_cells(mesh.faces),
        "strips": _b64_cells(mesh.strips),
        "arrays": arrays,
        "activeArray": active,
        "clim": [float(clim[0]), float(clim[1])] if clim is not None else None,
        "color": _to_hex(prm.get("color"), "#add8e6"),
        "opacity": _float_or(prm.get("opacity"), 1.0),
        "lineWidth": _float_or(prm.get("line_width"), 1.0),
        "pointSize": _float_or(prm.get("point_size"), 5.0),
        "representation": {"points": "points", "wireframe": "wireframe"}.get(prm.get("style"), "surface"),
        "ambient": _float_or(prm.get("ambient"), 0.0),
        "diffuse": _float_or(prm.get("diffuse"), 1.0),
        "specular": _float_or(prm.get("specular"), 0.0),
    }


def _serialize_array(name: str, location: str, values: np.ndarray) -> Optional[dict]:
    if values.dtype.kind not in "biuf" or values.size == 0:
        return None  # strings, objects...
    if values.ndim == 1:
        components = 1
        finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    else:
        values = values.reshape(len(values), -1)
        components = values.shape[1]
        finite = np.linalg.norm(values.astype(np.float64), axis=1)  # vtk.js colors by magnitude
        finite = finite[np.isfinite(finite)]
    rng = [float(finite.min()), float(finite.max())] if finite.size else [0.0, 1.0]
    return {
        "name": name,
        "location": location,
        "components": components,
        "range": rng,
        "data": _b64(values.astype(np.float32)),
    }


def _b64(values: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(values).tobytes()).decode("ascii")


def _b64_cells(cells) -> Optional[str]:
    cells = np.asarray(cells)
    return _b64(cells.astype(np.uint32)) if cells.size else None


def _sample_cmap(cmap) -> List[List[float]]:
    """matplotlib colormap -> list of [x, r, g, b] over [0, 1].

    Discrete colormaps (tab10, list of colors...) give sharp steps.
    """
    import matplotlib
    import matplotlib.colors as mcolors

    if isinstance(cmap, str):
        cm = matplotlib.colormaps[cmap]
    else:
        cm = mcolors.ListedColormap([mcolors.to_rgb(c) for c in cmap])
    if isinstance(cm, mcolors.ListedColormap) and cm.N <= 32:
        points = []
        for i in range(cm.N):
            r, g, b, _ = cm(i)
            lo, hi = i / cm.N, (i + 1) / cm.N
            points.append([lo, r, g, b])
            points.append([hi - 1e-6 if i < cm.N - 1 else hi, r, g, b])
        return [[round(v, 6) for v in p] for p in points]
    xs = np.linspace(0.0, 1.0, CMAP_SAMPLES)
    return [[round(float(x), 6), *(round(float(c), 6) for c in cm(x)[:3])] for x in xs]


def _to_hex(color, default: str) -> str:
    if color is None:
        return default
    try:
        import matplotlib.colors as mcolors
        return mcolors.to_hex(color)
    except Exception:
        try:
            return pv.Color(color).hex_rgb
        except Exception:
            return default


def _float_or(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default  # e.g. opacity="linear" (pyvista preset), unsupported by vtk.js


def _html_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
