# Python API

```bash
pip install git+https://github.com/nathandecaux/TractViewer.git
```

```python
from tractviewer import TractViewer

vis = TractViewer(background="black", off_screen=True, window_size=(1280, 720))
vis.add("anat.nii.gz", name="anat", display_array="intensity", cmap="gray",
        clim=(200, 800), opacity=0.2, iso=0.5)  # marching_cubes=False: slices only
vis.add("bundle.tck", name="bundle", display_array="length_mm", cmap="viridis",
        line_width=2, threshold=("length_mm", (50, 200)))

vis.desktop()                                      # interface in a native window
vis.serve(open_browser=True)                       # interface served over HTTP
vis.export_html("bundle.html")                     # standalone web page

print(vis.list_arrays())
vis.edit("bundle", cmap="plasma")
vis.screenshot("capture.png")                      # mesh_rotation=(-90, 0, 0) by default
vis.record("rotation.mp4", n_frames=180, fps=30)   # .gif => GIF
vis.remove("anat")
```

Parameters can be passed as kwargs or as a dict (`vis.add(path, {"cmap": "gray"})`). Datasets can
also be pyvista objects instead of paths.

## Constructor

```python
TractViewer(background="white", off_screen=False, window_size=None, show_ras_axes=False,
            coord_system=None, iso_value=None)
```

## Methods

- `add(...)`, `edit(name, ...)`, `remove(name)`
- `desktop(title, max_streamlines, max_voxels, debug, check_updates)`: native window, blocking
- `serve(host, port, open_browser, ...)`: web server, blocking (see [web-server.md](web-server.md))
- `export_html(path)`: standalone page
- `show()`, `screenshot(...)`, `record(...)`: pyvista rendering
- `names()`, `get_params(name)`, `get_dataset(name)`, `list_arrays()`
- `TractViewer.from_paths(...)`

## `add(...)` parameters

- `display_array`, `cmap`, `clim`, `color`, `opacity`, `visible`, `name`
- `threshold=(array, (min, max))`
- `style`: `surface`, `wireframe`, `points`, `envelope`
- `points_as_spheres`, `point_size`, `line_width` (down to 0.1: below 1, 1-pixel lines made fainter in proportion)
- `ambient`, `diffuse`, `specular`
- `contour` (`True` or a color) and `contour_width`
- `categorical_legend`, `categorical_max_values`, `unique_values`
- `scalar_bar` (alias `show_scalar_bar`), `scalar_bar_args`, `legend`
- `envelope_voxel_size`, `envelope_smooth`
- NIfTI loading: `marching_cubes`, `smooth`, `iso`, `model` (`dti`, `odf` or `scalar`: forces
  the interpretation of a 4D/5D image, see [Diffusion models](interface.md#diffusion-models))
- Tractogram loading: `max_points` (`"auto"` by default, `0` = everything, see
  [Large tractograms](interface.md#large-tractograms))
- `coord_system`: `RAS` or `LPS`, frame of a mesh/tractogram (e.g. a `.vtk` written by
  ANIMA/ITK is in LPS); its points are converted to the frame of the viewer

The same keys are accepted on the command line: `file:key=val,...` (see
[per-file parameters](cli.md#per-file-parameters)).

## Deprecations

Still working, with a `TractViewerDeprecationWarning`:

| Old | New |
| --- | --- |
| `rm(name)` | `remove(name)` |
| `capture_screenshot(path, rotation_x=..)` | `screenshot(path, mesh_rotation=(x, y, z))` |
| `record_rotation(path, gif=.., rotation_x=..)` | `record(path, mesh_rotation=(x, y, z))` |
| `style="enveloppe"`, `enveloppe_voxel_size`, `enveloppe_smooth` | `style="envelope"`, `envelope_voxel_size`, `envelope_smooth` |
| `tractviewer.enveloppe` (`enveloppe_minimale`, `enveloppe_tractogram`) | `tractviewer.envelope` (`minimal_envelope`, `tractogram_envelope`) |
| `TractViewer._load*`, `_choose_font_color`, ... | `tractviewer.loaders`, `tractviewer.backend`, ... |
| `tractviewer.cli._parse_input_spec` | `tractviewer.params.parse_spec` |
| root module `TractViewer.py` | `from tractviewer import TractViewer` |
| `tractviewer FILES --screenshot x.png --rotate N --web ...` | `tractviewer screenshot / record / web ...` |

To silence these warnings:

```python
import warnings
from tractviewer import TractViewerDeprecationWarning
warnings.filterwarnings("ignore", category=TractViewerDeprecationWarning)
```
