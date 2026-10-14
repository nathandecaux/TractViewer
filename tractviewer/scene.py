"""Scene building: adds the datasets to the PyVista plotter.

`populate(plotter, datasets, font_color)` adds each (dataset, params) to the
plotter. Only one scalar bar is shown (the first one requested). Failures are
isolated per dataset: logged, then the next datasets are still added.
"""
from __future__ import annotations
import sys
import contextlib
import numpy as np
import pyvista as pv
from tractviewer.envelope import minimal_envelope, tractogram_envelope
from tractviewer.debug import warn_debug


def populate(plotter, datasets, font_color: str, show_ras_axes: bool = False,
             mesh_rotation: tuple = (0.0, 0.0, 0.0)) -> None:
    _clear_scalar_bars(plotter)
    scalar_bar_added = False
    for idx, (ds, prm) in enumerate(datasets):
        if prm.get("visible", True) is False:
            continue
        try:
            scalar_bar_added = _add_dataset(plotter, ds, prm, font_color, scalar_bar_added)
        except Exception:
            warn_debug(
                f"Failed to add dataset index={idx}, name={prm.get('name')}.",
                exc=sys.exc_info(),
                extra=f"display_array={prm.get('display_array')}, "
                      f"style={prm.get('style')}, arrays={list(ds.array_names)}",
            )
            continue
    if show_ras_axes:
        _add_ras_axes(plotter, font_color, *mesh_rotation)


def _add_dataset(plotter, ds, prm, font_color, scalar_bar_added) -> bool:
    """Add one dataset; return the new scalar_bar_added state."""
    mesh = _apply_threshold(ds, prm)
    display_array = _resolve_display_array(mesh, prm)
    mesh, display_array, style = _apply_envelope(mesh, prm, display_array)

    if display_array == "rgb_encoded":
        add_kwargs = _rgb_kwargs(mesh, prm, style)
    else:
        show_bar = bool(display_array) and prm.get("scalar_bar", True) and not scalar_bar_added
        add_kwargs = _scalar_kwargs(prm, display_array, style, show_bar)

    _apply_contour(plotter, mesh, prm, add_kwargs)
    _apply_point_style(prm, add_kwargs)
    if display_array:
        _apply_categorical_legend(mesh, prm, display_array, add_kwargs)
    if add_kwargs.get("show_scalar_bar"):
        add_kwargs["scalar_bar_args"] = _scalar_bar_args(prm, display_array, font_color)
    for opt in ("point_size", "line_width", "ambient", "specular", "diffuse"):
        if opt in prm:
            add_kwargs[opt] = prm[opt]

    plotter.add_mesh(mesh, **{k: v for k, v in add_kwargs.items() if v is not None})
    return scalar_bar_added or bool(add_kwargs.get("show_scalar_bar"))


def _apply_threshold(ds, prm):
    """Filter the mesh on an array when threshold=(array, (min, max)) is given."""
    if not prm.get("threshold"):
        return ds
    arr_name, (vmin, vmax) = prm["threshold"]
    if arr_name not in ds.array_names:
        raise ValueError(f"Array '{arr_name}' not found for threshold.")
    return ds.threshold(value=(vmin, vmax), scalars=arr_name, invert=False)


def _resolve_display_array(mesh, prm):
    """Validate display_array, mask transparent pixels, make it the active scalars."""
    display_array = prm.get("display_array")
    if display_array and display_array not in mesh.array_names:
        raise ValueError(f"Array '{display_array}' not found in dataset ({mesh.array_names}).")

    if display_array == "rgb_encoded" and "is_transparent" in mesh.array_names:
        mesh = mesh.threshold(value=0.5, scalars="is_transparent", invert=True)
    if display_array:
        with contextlib.suppress(Exception):
            mesh.set_active_scalars(display_array)

    cast_type = prm.get("type")
    if display_array and cast_type in ("int", "float"):
        dtype = np.int32 if cast_type == "int" else np.float32
        for store in (mesh.point_data, mesh.cell_data):
            if display_array in store.keys():
                store[display_array] = store[display_array].astype(dtype)
                break
    return display_array


def _apply_envelope(mesh, prm, display_array):
    """With style='envelope', replace the mesh by its envelope and force 'surface'."""
    if prm.get("style") not in ("envelope", "enveloppe"):
        return mesh, display_array, prm.get("style")

    try:
        is_tract = isinstance(mesh, pv.PolyData) and getattr(mesh, "n_lines", 0) > 0
        if is_tract:
            mesh = tractogram_envelope(
                mesh,
                display_array=display_array,
                voxel_size=float(prm.get("envelope_voxel_size", 1.0)),
                smooth_iterations=int(prm.get("envelope_smooth", 50)),
            )
            if display_array and not (
                display_array in mesh.point_data.keys() or display_array in mesh.cell_data.keys()
            ):
                display_array = None  # array missing from the resulting mesh
            elif display_array:
                with contextlib.suppress(Exception):
                    mesh.set_active_scalars(display_array)
        else:
            mesh = minimal_envelope(mesh)
    except Exception:
        warn_debug(f"Envelope failed (name={prm.get('name')}).", exc=sys.exc_info())
    return mesh, display_array, "surface"


def _rgb_kwargs(mesh, prm, style):
    """Decode 'rgb_encoded' into a uint8 RGB array and build the add_mesh kwargs."""
    rgb_values = mesh.point_data["rgb_encoded"].astype(np.uint32)
    colors = np.column_stack([
        (rgb_values >> 16) & 0xFF,
        (rgb_values >> 8) & 0xFF,
        rgb_values & 0xFF,
    ]).astype(np.uint8)
    mesh.point_data["RGB"] = colors
    return dict(
        scalars="RGB",
        rgb=True,
        opacity=prm.get("opacity", 1.0),
        show_edges=prm.get("show_edges", False),
        show_scalar_bar=False,
        name=prm.get("name"),
        style=style,
    )


def _scalar_kwargs(prm, display_array, style, show_bar):
    clim = prm.get("clim")
    if clim is None and 'unique_values' in prm:
        uv = prm['unique_values']
        clim = (float(min(uv)), float(max(uv)))
    return dict(
        scalars=display_array,
        cmap=prm.get("cmap"),
        clim=clim,
        opacity=prm.get("opacity", 1.0),
        show_edges=prm.get("show_edges", False),
        show_scalar_bar=show_bar,
        name=prm.get("name"),
        color=prm.get("color"),
        style=style,
    )


def _apply_contour(plotter, mesh, prm, add_kwargs):
    """Enable the contour: thicker underlay for lines, edges otherwise."""
    contour = prm.get("contour", None)
    if contour is None or contour is False:
        if "edge_color" in prm:
            add_kwargs["edge_color"] = prm["edge_color"]
        return

    is_line_dataset = (
        isinstance(mesh, pv.PolyData)
        and getattr(mesh, "n_lines", 0) > 0
        and getattr(mesh, "n_cells", 0) == getattr(mesh, "n_lines", 0)
    )
    contour_color = contour if isinstance(contour, str) else "black"

    if is_line_dataset:
        # show_edges has no effect on lines: draw a thicker line underneath.
        base_width = float(prm.get("line_width", 2.0))
        outline_width = max(1.0, base_width + float(prm.get("contour_width", 2.0)))
        outline_kwargs = dict(
            scalars=None,
            color=contour_color,
            opacity=prm.get("opacity", 1.0),
            show_scalar_bar=False,
            line_width=outline_width,
            render_lines_as_tubes=True,
            name=f"{prm.get('name')}_contour" if prm.get("name") else None,
        )
        plotter.add_mesh(mesh, **{k: v for k, v in outline_kwargs.items() if v is not None})
        add_kwargs["line_width"] = max(1.0, base_width)
        add_kwargs["render_lines_as_tubes"] = True
        add_kwargs.pop("show_edges", None)
        add_kwargs.pop("edge_color", None)
    else:
        add_kwargs["show_edges"] = True
        add_kwargs["edge_color"] = contour_color


def _apply_point_style(prm, add_kwargs):
    if add_kwargs.get("style") == "points" and "point_size" not in prm:
        add_kwargs["point_size"] = 5
    if "points_as_spheres" in prm:
        add_kwargs["render_points_as_spheres"] = prm["points_as_spheres"]


def _apply_categorical_legend(mesh, prm, display_array, add_kwargs):
    """Enable a categorical legend (automatic when there are few unique values).

    If `prm['unique_values']` is given, an *indexed* LUT is built (VTK
    IndexedLookup): the i-th value of the list gets the i-th color of the
    colormap. A value's color is therefore stable from one capture to the next,
    regardless of the values actually present in the mesh, their spacing and
    the chosen colormap.
    """
    ordered_unique_vals = None
    if 'unique_values' in prm:
        ordered_unique_vals = list(prm['unique_values'])
        unique_vals = np.asarray(ordered_unique_vals)
    else:
        values = None
        if display_array in mesh.point_data.keys():
            values = np.asarray(mesh.point_data[display_array]).ravel()
        elif display_array in mesh.cell_data.keys():
            values = np.asarray(mesh.cell_data[display_array]).ravel()

        unique_vals = np.array([])
        if values is not None and values.size > 0:
            if np.issubdtype(values.dtype, np.number):
                values = values[np.isfinite(values)]
            if values.size > 0:
                unique_vals = np.unique(values)

    setting = prm.get("categorical_legend", None)
    max_values = int(prm.get("categorical_max_values", 10))
    if isinstance(setting, bool):
        use_categorical = setting
    elif ordered_unique_vals is not None:
        # Explicit values: force the categorical legend.
        use_categorical = True
    else:
        use_categorical = 0 < unique_vals.size < max_values

    if not (use_categorical and unique_vals.size > 0):
        return

    if ordered_unique_vals is not None:
        # Indexed LUT: color fixed by position in unique_values.
        _cast_display_array(mesh, display_array, np.float64)
        add_kwargs["cmap"] = _build_indexed_lut(ordered_unique_vals, prm.get("cmap"))
        # The LUT handles colors, range and annotations: drop conflicting kwargs.
        add_kwargs["clim"] = None
        add_kwargs.pop("categories", None)
        add_kwargs.pop("annotations", None)
    else:
        add_kwargs["categories"] = True
        add_kwargs["annotations"] = {v: str(v) for v in unique_vals.tolist()}


def _cast_display_array(mesh, display_array, dtype):
    """Cast the active array to `dtype` (point_data first, then cell_data)."""
    for store in (mesh.point_data, mesh.cell_data):
        if display_array in store.keys():
            store[display_array] = np.asarray(store[display_array]).astype(dtype)
            return


def _build_indexed_lut(ordered_unique_vals, cmap_name):
    """Build an indexed categorical LUT for explicit values.

    The i-th value of `ordered_unique_vals` gets the color sampled at position
    i/(n-1) of the colormap. VTK IndexedLookup maps colors by value match (not
    interpolation), so a value missing from the mesh does not shift the colors
    of the others. Annotations are compared as doubles: scalars are cast to
    float64 beforehand.
    """
    import vtk

    n = len(ordered_unique_vals)
    ramp = pv.LookupTable(cmap=cmap_name or "viridis", n_values=n if n > 1 else 2)
    lut = pv.LookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
    for i, v in enumerate(ordered_unique_vals):
        lut.SetTableValue(i, *ramp.GetTableValue(i))
        label = str(int(v)) if float(v).is_integer() else str(v)
        lut.SetAnnotation(vtk.vtkVariant(float(v)), label)
    lut.SetIndexedLookup(True)
    lut.Build()
    return lut


def _scalar_bar_args(prm, display_array, font_color):
    legend = prm.get("legend")
    if isinstance(legend, str):
        title = legend
    else:
        title = display_array or ""
    defaults = {
        "title": title,
        "n_labels": 5,
        "fmt": "%.2f",
        "color": font_color,
    }
    defaults.update(prm.get("scalar_bar_args") or {})
    return defaults


def _add_ras_axes(plotter, font_color: str = "white",
                  rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
                  scale: float = 50.0) -> None:
    """Add an RAS axes overlay (always visible, follows the mesh rotation)."""
    import vtk
    from tractviewer import geometry

    # Data is in LPS (+X=Left, +Y=Posterior, +Z=Superior), so R and A point in negative directions.
    base_directions = {
        "R": (np.array([-1.0, 0.0, 0.0]), "red"),
        "A": (np.array([0.0, -1.0, 0.0]), "green"),
        "S": (np.array([0.0, 0.0, 1.0]), "blue"),
    }

    if rx or ry or rz:
        axes = {
            label: (geometry.rotate_points(d.reshape(1, 3), rx, ry, rz).ravel(), color)
            for label, (d, color) in base_directions.items()
        }
    else:
        axes = base_directions

    # Overlay renderer (layer 1) sharing the same camera.
    # plotter.clear() does not remove it: replace the one from a previous populate.
    ren_win = plotter.ren_win
    previous = getattr(plotter, "_tractviewer_ras_overlay", None)
    if previous is not None:
        ren_win.RemoveRenderer(previous)
    overlay = vtk.vtkRenderer()
    plotter._tractviewer_ras_overlay = overlay
    overlay.SetLayer(1)
    overlay.SetInteractive(0)
    overlay.SetViewport(0.0, 0.0, 1.0, 1.0)
    overlay.SetBackgroundAlpha(0.0)
    overlay.SetBackground(0.0, 0.0, 0.0)
    overlay.SetActiveCamera(plotter.renderer.GetActiveCamera())
    ren_win.SetNumberOfLayers(2)
    ren_win.AddRenderer(overlay)

    origin = np.zeros(3)
    font_rgb = pv.Color(font_color).float_rgb

    for label, (direction, color_name) in axes.items():
        tip = direction * scale
        rgb = pv.Color(color_name).float_rgb

        shaft = pv.Line(origin, tip * 0.85)
        shaft_mapper = vtk.vtkPolyDataMapper()
        shaft_mapper.SetInputData(shaft)
        shaft_actor = vtk.vtkActor()
        shaft_actor.SetMapper(shaft_mapper)
        shaft_actor.GetProperty().SetColor(*rgb)
        shaft_actor.GetProperty().SetLineWidth(3.0)
        overlay.AddActor(shaft_actor)

        cone = pv.Cone(center=tip * 0.925, direction=direction, height=scale * 0.15, radius=scale * 0.04)
        cone_mapper = vtk.vtkPolyDataMapper()
        cone_mapper.SetInputData(cone)
        cone_actor = vtk.vtkActor()
        cone_actor.SetMapper(cone_mapper)
        cone_actor.GetProperty().SetColor(*rgb)
        overlay.AddActor(cone_actor)

        text_source = vtk.vtkVectorText()
        text_source.SetText(label)
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())
        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetCamera(overlay.GetActiveCamera())
        text_actor.GetProperty().SetColor(*font_rgb)
        text_actor.SetScale(scale * 0.15)
        text_actor.SetPosition(*(tip * 1.15))
        overlay.AddActor(text_actor)


def _clear_scalar_bars(plotter):
    """Remove lingering scalar bars (backend dependent, after clear/rebuild)."""
    with contextlib.suppress(Exception):
        for title in list(getattr(plotter, "scalar_bars", {}).keys()):
            with contextlib.suppress(Exception):
                plotter.remove_scalar_bar(title=title)
