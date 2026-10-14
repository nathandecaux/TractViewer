from __future__ import annotations
from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence, Tuple
import os
import sys
import gc
import contextlib
import warnings
import faulthandler
import numpy as np
import pyvista as pv

from tractviewer import loaders, recording, geometry, scene, params as params_mod
from tractviewer._deprecation import deprecated, warn_deprecated
from tractviewer._legacy import LegacyMethodsMixin
from tractviewer.loaders import DataInput
from tractviewer.backend import detect_vtk_backend, choose_font_color
from tractviewer.debug import debug_enabled, warn_debug, env_info

ParamDict = Dict[str, object]
Rotation = Tuple[float, float, float]

DEFAULT_COORD_SYSTEM = "LPS"
DEFAULT_ISO_VALUE = 0.2
DEFAULT_MESH_ROTATION: Rotation = (-90.0, 0.0, 0.0)


def default_coord_system() -> str:
    """TRACTVIEWER_NIFTI_COORD if set, else LPS."""
    return os.environ.get("TRACTVIEWER_NIFTI_COORD", DEFAULT_COORD_SYSTEM).upper()


def default_iso_value() -> float:
    """TRACTVIEWER_NIFTI_ISO if set (and valid), else 0.2."""
    try:
        return float(os.environ["TRACTVIEWER_NIFTI_ISO"])
    except (KeyError, ValueError):
        return DEFAULT_ISO_VALUE


class TractViewer(LegacyMethodsMixin):
    """Visualize VTK files/datasets with pyvista.

    Per-dataset parameters (key -> meaning), see also tractviewer.params:
      - display_array: str | None   scalar array to display
      - cmap: str | Sequence[str]   matplotlib colormap
      - clim: (float, float)        display range
      - threshold: (str, (min,max)) filter on an array
      - opacity: float | str | Sequence[float]
      - color: fixed color (when there is no display_array)
      - contour: bool | str         contour (True => black, str => color)
      - contour_width: float        extra contour thickness (lines)
      - categorical_legend: bool | None   force the categorical legend (None => auto)
      - categorical_max_values: int       automatic threshold for the categorical legend
      - unique_values: Sequence     explicit categorical values (fixed colors)
      - show_edges / scalar_bar: bool
      - point_size / line_width: sizes
      - ambient / specular / diffuse: material settings
      - style: 'surface' | 'wireframe' | 'points' | 'envelope'
      - points_as_spheres: bool (useful with style='points')
      - visible: bool
      - name: internal name (automatic otherwise)
      - legend : str   scalar bar title
    Loading parameters (add only):
      - marching_cubes: bool (NIfTI, default True)
      - smooth: int smoothing iterations (NIfTI, default 200)
      - iso: float iso value (NIfTI, default: the viewer's iso_value)

    Global parameters (constructor):
      - background: background color (default "white")
      - off_screen: render without a window (e.g. on a server)
      - window_size: (width, height)
      - show_ras_axes: show RAS axes
      - coord_system: 'LPS' or 'RAS' for NIfTI files (default: TRACTVIEWER_NIFTI_COORD, else LPS)
      - iso_value: default NIfTI iso value (default: TRACTVIEWER_NIFTI_ISO, else 0.2)
    """

    def __init__(self, background: str = "white", off_screen: bool = False,
                 window_size: Optional[Tuple[int, int]] = None,
                 show_ras_axes: bool = False,
                 coord_system: Optional[str] = None,
                 iso_value: Optional[float] = None):
        self._datasets: List[Tuple[pv.DataSet, ParamDict]] = []
        self.background = background
        self.off_screen = off_screen
        self.window_size = window_size
        self.show_ras_axes = show_ras_axes
        self.coord_system = (coord_system or default_coord_system()).upper()
        self.iso_value = default_iso_value() if iso_value is None else float(iso_value)
        self._plotter: Optional[pv.Plotter] = None
        self._font_color = choose_font_color(background)
        self._reference_affine: Optional[np.ndarray] = None  # 4x4 of the first NIfTI
        self._debug = debug_enabled()
        self._current_mesh_rotation: Rotation = (0.0, 0.0, 0.0)

        if self._debug:
            with contextlib.suppress(Exception):
                faulthandler.enable()
            os.environ.setdefault("QT_DEBUG_PLUGINS", "1")

        # Switch to headless without a DISPLAY
        if not self.off_screen and not os.environ.get("DISPLAY"):
            warnings.warn("No DISPLAY detected -> switching to off_screen mode.")
            self.off_screen = True

        # Switch to headless with a display-less VTK backend (OSMesa/EGL)
        self._vtk_backend_class, backend_headless = detect_vtk_backend()
        if not self.off_screen and backend_headless:
            warnings.warn(
                f"Backend VTK '{self._vtk_backend_class}' (OSMesa/EGL) -> switching to off_screen."
            )
            self.off_screen = True

    # ------------------------------
    # Adding / editing datasets
    # ------------------------------
    def add(self, data: DataInput, params: Optional[ParamDict] = None, **kwargs):
        params = params_mod.normalize({**(params or {}), **kwargs})
        marching_cubes = params.pop("marching_cubes", True)
        smooth_iterations = int(params.pop("smooth", 200))
        iso = params.pop("iso", None)
        ds = loaders.load(
            data, marching_cubes=marching_cubes, smooth_iterations=smooth_iterations,
            iso_value=self.iso_value if iso is None else iso, coord_system=self.coord_system,
        )
        if not params.get("name"):
            params["name"] = f"ds{len(self._datasets)}"

        # Remember the reference affine of the first NIfTI and reproject existing datasets.
        if self._reference_affine is None and "nifti_affine" in ds.field_data:
            flat = np.asarray(ds.field_data["nifti_affine"], dtype=float)
            if flat.size == 16:
                self._reference_affine = flat.reshape(4, 4)
                for existing_ds, _ in self._datasets:
                    self._maybe_project_tract(existing_ds)

        self._datasets.append((ds, params))
        self._maybe_project_tract(ds)
        self._invalidate()
        return self

    def edit(self, name: str, params: Optional[ParamDict] = None, **kwargs):
        """Update the parameters of an existing dataset (merges params + kwargs)."""
        updates = params_mod.normalize({**(params or {}), **kwargs})
        if not updates:
            return self
        matches = [prm for _, prm in self._datasets if prm.get("name") == name]
        if not matches:
            raise ValueError(f"No dataset named '{name}'.")
        for prm in matches:
            prm.update(updates)
        self._invalidate()
        return self

    def remove(self, name: str):
        """Remove a dataset by name and release its resources."""
        removed = [ds for ds, prm in self._datasets if prm.get("name") == name]
        if not removed:
            return self
        self._datasets = [(ds, prm) for ds, prm in self._datasets if prm.get("name") != name]
        self._invalidate()
        for ds in removed:
            self._release_dataset_resources(ds)
        gc.collect()
        return self

    @deprecated("TractViewer.remove", old="TractViewer.rm")
    def rm(self, name: str):
        return self.remove(name)

    @classmethod
    def from_paths(cls, paths: Sequence[DataInput],
                   params_list: Optional[Sequence[ParamDict]] = None, **kwargs):
        vis = cls(**kwargs)
        params_list = params_list or [{}] * len(paths)
        for p, prm in zip(paths, params_list):
            vis.add(p, prm)
        return vis

    # ------------------------------
    # Introspection
    # ------------------------------
    def names(self) -> List[str]:
        return [prm.get("name") for _, prm in self._datasets]

    def get_params(self, name: str) -> ParamDict:
        """Copy of the parameters of dataset `name`."""
        return dict(self._find(name)[1])

    def get_dataset(self, name: str) -> pv.DataSet:
        return self._find(name)[0]

    def list_arrays(self) -> Dict[str, List[str]]:
        return {prm.get("name"): list(ds.array_names) for ds, prm in self._datasets}

    def _find(self, name: str) -> Tuple[pv.DataSet, ParamDict]:
        for ds, prm in self._datasets:
            if prm.get("name") == name:
                return ds, prm
        raise ValueError(f"No dataset named '{name}'.")

    # ------------------------------
    # Tractogram projection
    # ------------------------------
    def _maybe_project_tract(self, ds: pv.DataSet):
        """Project the points of a tractogram through the reference affine.

        A tract is detected through cell_data['streamline_id']. If its points look
        like voxel indices, the affine is applied. The field_data['tract_transformed']
        flag prevents applying it twice.
        """
        if self._reference_affine is None or not isinstance(ds, pv.PolyData):
            return
        if "streamline_id" not in ds.cell_data or "tract_transformed" in ds.field_data:
            return
        try:
            pts = ds.points
            is_voxel = self._looks_like_voxel_coords(pts)
            if is_voxel:
                pts_h = np.c_[pts, np.ones(len(pts))]
                ds.points = (pts_h @ self._reference_affine.T)[:, :3]

            # Flip RAS->LPS if requested and not done yet.
            if self.coord_system == "LPS" and ds.field_data.get("coord_system", ["RAS"])[0] != "LPS":
                ds.points[:, 0] *= -1.0
                ds.points[:, 1] *= -1.0
                ds.field_data["coord_system"] = np.array(["LPS"], dtype=object)
            ds.field_data["tract_transformed"] = np.array([1 if is_voxel else 0], dtype=np.int8)
        except Exception as e:
            warnings.warn(f"Tract projection failed: {e}")

    def _looks_like_voxel_coords(self, pts: np.ndarray) -> bool:
        """Heuristic: points inside the volume bounds and mostly integers."""
        dims = next(
            (np.asarray(ref.field_data["nifti_dims"], dtype=int)
             for ref, _ in self._datasets if "nifti_dims" in ref.field_data),
            None,
        )
        if dims is None or dims.size < 3:
            return False
        if not (np.all(pts.min(axis=0) >= -1) and np.all(pts.max(axis=0) <= dims[:3] + 2)):
            return False
        frac = np.abs(pts - np.round(pts)).ravel()
        return frac.size > 0 and np.mean(frac < 0.05) > 0.8

    @staticmethod
    def _release_dataset_resources(ds: pv.DataSet):
        """Explicitly release the large PyVista/VTK buffers of a dataset."""
        with contextlib.suppress(Exception):
            ds.clear_data()
        with contextlib.suppress(Exception):
            ds.points = np.empty((0, 3), dtype=np.float32)
        for attr in ("lines", "faces", "polys", "strips"):
            with contextlib.suppress(Exception):
                setattr(ds, attr, np.empty((0,), dtype=np.int32))
        with contextlib.suppress(Exception):
            ds.deep_clean()

    # ------------------------------
    # Plotter lifecycle
    # ------------------------------
    def _invalidate(self):
        """The scene changed (datasets, parameters, rotation): the plotter must be rebuilt.

        Extension point for subclasses that need to keep their plotter alive.
        """
        self._close_plotter()

    def _close_plotter(self):
        if self._plotter is None:
            return
        with contextlib.suppress(Exception):
            self._plotter.clear()
        with contextlib.suppress(Exception):
            self._plotter.close()
        self._plotter = None

    def _ensure_plotter(self):
        """Build the plotter if missing or closed, and add the datasets to it."""
        if self._plotter is not None and not getattr(self._plotter, "_closed", False):
            return
        self._plotter = self._create_plotter()
        self._font_color = choose_font_color(self.background)
        self._populate()
        self._init_camera()

    def _create_plotter(self) -> pv.Plotter:
        plotter = pv.Plotter(off_screen=self.off_screen)
        plotter.set_background(self.background)
        if self.window_size:
            with contextlib.suppress(Exception):
                plotter.window_size = tuple(map(int, self.window_size))
        return plotter

    def _populate(self):
        try:
            scene.populate(self._plotter, self._datasets, self._font_color,
                           show_ras_axes=self.show_ras_axes,
                           mesh_rotation=self._current_mesh_rotation)
        except Exception:
            warn_debug("Error while populating the plotter.", exc=sys.exc_info(),
                       extra=self._env_info())
            raise

    def _init_camera(self):
        if self._datasets:
            self._plotter.camera_position = "xy"

    # ------------------------------
    # Rotations
    # ------------------------------
    def rotate_mesh(self, rotation_x: float = 0.0, rotation_y: float = 0.0, rotation_z: float = 0.0):
        """Rotate (degrees) the meshes of all datasets."""
        if not any([rotation_x, rotation_y, rotation_z]):
            return
        for ds, _ in self._datasets:
            if isinstance(ds, pv.ImageData):
                continue  # points implicites: rotation non applicable
            ds.points = geometry.rotate_points(ds.points.copy(), rotation_x, rotation_y, rotation_z)
        self._invalidate()

    def _backup_mesh_state(self) -> List[Optional[np.ndarray]]:
        return [None if isinstance(ds, pv.ImageData) else ds.points.copy()
                for ds, _ in self._datasets]

    def _restore_mesh_state(self, backup: List[Optional[np.ndarray]]):
        for (ds, _), points in zip(self._datasets, backup):
            if points is not None:
                ds.points = points

    def _apply_camera_rotation(self, azimuth_deg: float, elevation_deg: float):
        """Rotate the camera and render (required off-screen for captures to follow)."""
        geometry.rotate_camera(self._plotter.camera, azimuth_deg, elevation_deg)
        self._plotter.render()

    # ------------------------------
    # Interactive display
    # ------------------------------
    def show(self, **show_kwargs):
        if self._debug:
            warn_debug("Appel show()", extra=self._env_info())
        self._ensure_plotter()

        if self.off_screen:
            reason = f" (backend: {self._vtk_backend_class})" if self._vtk_backend_class else ""
            warnings.warn(
                "off_screen=True: no interactive window will open."
                f"{reason} Use screenshot(), record(), export_html() or serve(), "
                "or install a VTK build with display support."
            )
        else:
            self._plotter.render()
        return self._plotter.show(**show_kwargs, interactive=True)

    # ------------------------------
    # Screenshot
    # ------------------------------
    def screenshot(self, output_path: Union[str, Path, None] = None, transparent: bool = False,
                   mesh_rotation: Rotation = DEFAULT_MESH_ROTATION, **kwargs):
        """Capture an image (PNG) and return the RGB(A) array.

        mesh_rotation: (x, y, z) rotation in degrees applied to the meshes during the capture.
        Without output_path, nothing is written to disk.
        """
        with self._temporary_mesh_rotation(*mesh_rotation):
            self._ensure_plotter()
            filename = None
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                filename = str(output_path)
            kwargs.setdefault("return_img", True)
            return self._plotter.screenshot(
                filename=filename, transparent_background=transparent, **kwargs
            )

    @deprecated("TractViewer.screenshot(path, mesh_rotation=(x, y, z))",
                old="TractViewer.capture_screenshot")
    def capture_screenshot(self, output_path: Union[str, Path], transparent: bool = False,
                           rotation_x: float = -90.0, rotation_y: float = 0.0,
                           rotation_z: float = 0.0, **kwargs):
        return self.screenshot(output_path, transparent=transparent,
                               mesh_rotation=(rotation_x, rotation_y, rotation_z), **kwargs)

    # ------------------------------
    # Video / GIF recording
    # ------------------------------
    def record(
        self,
        output_path: Union[str, Path],
        n_frames: int = 180,
        step: Optional[float] = None,
        elevation: float = 0.0,
        fps: int = 30,
        mesh_rotation: Rotation = DEFAULT_MESH_ROTATION,
        window_size: Optional[Tuple[int, int]] = None,
        supersample: int = 1,
        anti_aliasing: bool = True,
        quality: int = 5,
        codec: str = "mpeg4",
        bitrate: Optional[str] = None,
        crf: Optional[int] = None,
        auto_codec_fallback: bool = True,
        prefer_software: bool = True,
    ) -> str:
        """Rotate the camera (360° by default) and record a GIF or a video.

        The format follows the output_path extension (.gif => GIF, else video;
        no extension => .mp4). Rendering is always off-screen.
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp4")
        gif = output_path.suffix.lower() == ".gif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if step is None:
            step = 360.0 / n_frames

        previous_off_screen = self.off_screen
        if not self.off_screen:
            self.off_screen = True
            self._invalidate()  # the current plotter is on screen
        try:
            with self._temporary_mesh_rotation(*mesh_rotation):
                self._ensure_plotter()
                self._configure_render(window_size, supersample, anti_aliasing)
                frames = self._capture_rotation_frames(n_frames, step, elevation)
            if gif:
                recording.write_gif(frames, output_path, fps=fps)
            else:
                recording.write_video(
                    frames, output_path, fps=fps, quality=quality, codec=codec,
                    bitrate=bitrate, crf=crf, auto_codec_fallback=auto_codec_fallback,
                    prefer_software=prefer_software,
                )
        finally:
            self.off_screen = previous_off_screen
            self._invalidate()  # camera / window size changed
        return str(output_path)

    def record_rotation(
        self,
        output_path: Union[str, Path],
        n_frames: int = 180,
        step: float = None,
        elevation: float = 0.0,
        gif: bool = False,
        fps: int = 30,
        quality: int = 5,
        window_size: Optional[Tuple[int, int]] = None,
        supersample: int = 1,
        anti_aliasing: bool = True,
        codec: str = "mpeg4",
        bitrate: Optional[str] = None,
        crf: Optional[int] = None,
        auto_codec_fallback: bool = True,
        prefer_software: bool = True,
        off_screen: Optional[bool] = True,
        rotation_x: float = -90.0,
        rotation_y: float = 0.0,
        rotation_z: float = 0.0,
    ):
        """Deprecated: use record(). `gif` only matters when output_path has no extension."""
        warn_deprecated("TractViewer.record_rotation",
                        "TractViewer.record(path, mesh_rotation=(x, y, z))")
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".gif" if gif else ".mp4")
        return self.record(
            output_path, n_frames=n_frames, step=step, elevation=elevation, fps=fps,
            mesh_rotation=(rotation_x, rotation_y, rotation_z), window_size=window_size,
            supersample=supersample, anti_aliasing=anti_aliasing, quality=quality, codec=codec,
            bitrate=bitrate, crf=crf, auto_codec_fallback=auto_codec_fallback,
            prefer_software=prefer_software,
        )

    def _configure_render(self, window_size, supersample, anti_aliasing):
        if window_size:
            w, h = int(window_size[0]), int(window_size[1])
            if supersample > 1:
                w, h = w * supersample, h * supersample
            with contextlib.suppress(Exception):
                self._plotter.window_size = (w, h)
        if anti_aliasing:
            with contextlib.suppress(Exception):
                self._plotter.enable_anti_aliasing()

    def _capture_rotation_frames(self, n_frames, step, elevation) -> List[np.ndarray]:
        frames = []
        for i in range(n_frames):
            if elevation:
                elev_step = elevation if i < n_frames / 2 else -elevation
            else:
                elev_step = 0.0
            self._apply_camera_rotation(azimuth_deg=step, elevation_deg=elev_step)
            frames.append(self._plotter.screenshot(return_img=True))
        return frames

    # ------------------------------
    # Web export (vtk.js, rendered in the browser)
    # ------------------------------
    def export_html(self, output_path: Union[str, Path], title: str = "TractViewer",
                    max_streamlines: Optional[int] = None, max_voxels: Optional[int] = None) -> str:
        """Write a standalone HTML page (vtk.js) with every dataset and all their arrays.

        Array, colormap, range, opacity, slices (NIfTI volumes) and rotation GIFs
        are then handled in the browser. max_streamlines subsamples large
        tractograms, max_voxels decimates large volumes.
        """
        from tractviewer.web import vtkjs
        return vtkjs.export_html(self, output_path, title=title, max_streamlines=max_streamlines,
                                 max_voxels=max_voxels or vtkjs.DEFAULT_MAX_VOXELS)

    def serve(self, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = False,
              title: str = "TractViewer", max_streamlines: Optional[int] = None,
              max_voxels: Optional[int] = None):
        """vtk.js web interface with file uploads (blocking, Ctrl+C to stop)."""
        from tractviewer.web import server, vtkjs
        server.serve(self, host=host, port=port, open_browser=open_browser, title=title,
                     max_streamlines=max_streamlines,
                     max_voxels=max_voxels or vtkjs.DEFAULT_MAX_VOXELS)

    # ------------------------------
    # Helpers
    # ------------------------------
    @contextlib.contextmanager
    def _temporary_mesh_rotation(self, rx, ry, rz):
        """Rotate the meshes for the duration of the block, then restore them."""
        backup = None
        if any([rx, ry, rz]):
            backup = self._backup_mesh_state()
            self._current_mesh_rotation = (rx, ry, rz)
            self.rotate_mesh(rx, ry, rz)
        try:
            yield
        finally:
            self._current_mesh_rotation = (0.0, 0.0, 0.0)
            if backup is not None:
                self._restore_mesh_state(backup)
                self._invalidate()

    def _env_info(self) -> str:
        return env_info(backend_class=self._vtk_backend_class, off_screen=self.off_screen)
