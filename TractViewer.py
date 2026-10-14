from __future__ import annotations
from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence, Tuple
import os
import warnings
import math
import numpy as np
import contextlib
import pyvista as pv
import subprocess
import shutil
import tempfile

try:
    import imageio.v2 as imageio  # GIF option
except Exception:  # pragma: no cover
    imageio = None


DataInput = Union[str, Path, pv.DataSet]
ParamDict = Dict[str, object]

warnings.warn(
    "Deprecated: the root module 'TractViewer.py' (old standalone version). "
    "Use 'from tractviewer import TractViewer' instead.",
    DeprecationWarning,
    stacklevel=2,
)


class TractViewer:
    """
    Simplified visualization of VTK files/datasets with pyvista.

    Per-dataset parameters (key -> meaning):
        - display_array: str | None  name of the scalar array to display
      - cmap: str | Sequence[str]  colormap matplotlib
        - clim: Tuple[float,float]   (min, max) display range
        - threshold: (str, (min,max))  filter on an array => applies a threshold
      - opacity: float | str | Sequence[float]
      - show_edges: bool
      - show_scalar_bar: bool
      - point_size / line_width: tailles optionnelles
        - ambient / specular / diffuse: material settings
        - style: 'surface' | 'wireframe' | 'points'   (new; default surface)
        - points_as_spheres: bool (useful with style='points')
      - name: nom interne (sinon auto)
    Global parameters (constructor):
        - background: background color (default "white")
        - off_screen: bool (render without a window, e.g. on a server)
    """
    def __init__(self, background: str = "white", off_screen: bool = False):
        print('COUCOUCOUCOCUCOU')
        self._datasets: List[Tuple[pv.DataSet, ParamDict]] = []
        self.background = background
        self.off_screen = off_screen
        self._plotter: Optional[pv.Plotter] = None
        self._show_scalar_bar_added = False
        self._font_color = self._choose_font_color(self.background)
        self._reference_affine: Optional[np.ndarray] = None  # 4x4 from the first NIfTI
        # Automatically switch to headless without a DISPLAY
        if not self.off_screen and not os.environ.get("DISPLAY"):
            warnings.warn("No DISPLAY detected -> switching to off_screen mode.")
            self.off_screen = True

    # ------------------------------
    # Loading / adding datasets
    # ------------------------------
    @staticmethod
    def _load(obj: DataInput) -> pv.DataSet:
        """Load an object according to its extension.

        Additional supported extensions:
          - .tck / .trk : tractography -> vtkPolyData (lines)
          - .nii / .nii.gz : NIfTI -> projection surface (isosurface)
        """
        if isinstance(obj, pv.DataSet):
            return obj
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        double_suffix = ''.join(path.suffixes[-2:]).lower()  # for .nii.gz
        if suffix in (".tck", ".trk"):
            return TractViewer._load_tract_file(path)
        if suffix in (".nii",) or double_suffix == ".nii.gz":
            return TractViewer._load_nifti_surface(path)
        return pv.read(path)

    # --- Tractography loading ---
    @staticmethod
    def _load_tract_file(path: Path) -> pv.PolyData:
        """Read a tractography file (.tck or .trk) and return a PolyData.

        Uses nibabel when available. Each streamline becomes a polyline.
        Adds arrays:
          - 'streamline_id' (cell data)
            - 'point_index' (point data, local index along the streamline)
            - 'length_mm' (cell data, streamline length in mm)
        """
        try:
            import nibabel as nib
        except Exception as e:  # pragma: no cover
            raise ImportError("nibabel is required to load tractography files (.tck/.trk): 'pip install nibabel'") from e

        tractogram = nib.streamlines.load(str(path))
        streamlines = tractogram.streamlines
        # Build points / connectivity
        all_points: List[np.ndarray] = []
        lines_idx: List[int] = []
        cell_data_lengths: List[float] = []
        cell_stream_ids: List[int] = []
        point_stream_index: List[int] = []
        running_index = 0
        for sid, sl in enumerate(streamlines):
            if sl.shape[0] < 2:
                continue
            npts = sl.shape[0]
            all_points.append(sl.astype(np.float32))
            # VTK polyline: [npts, p0, p1, ...]
            lines_idx.append(npts)
            lines_idx.extend(range(running_index, running_index + npts))
            # Cell data
            dists = np.linalg.norm(np.diff(sl, axis=0), axis=1)
            cell_data_lengths.append(float(dists.sum()))
            cell_stream_ids.append(sid)
            # Point data (local index)
            point_stream_index.extend(list(range(npts)))
            running_index += npts
        if not all_points:
            raise ValueError(f"No valid streamline in {path}.")
        points = np.vstack(all_points)
        poly = pv.PolyData()
        poly.points = points
        poly.lines = np.array(lines_idx, dtype=np.int32)
        # Cell data
        poly.cell_data["streamline_id"] = np.array(cell_stream_ids, dtype=np.int32)
        poly.cell_data["length_mm"] = np.array(cell_data_lengths, dtype=np.float32)
        # Point data
        poly.point_data["point_index"] = np.array(point_stream_index, dtype=np.int32)
        return poly

    # --- NIfTI loading / projection ---
    @staticmethod
    def _load_nifti_surface(path: Path) -> pv.PolyData:
        """Load a NIfTI and return an isosurface (marching cubes).

        Mechanism:
          - Charge volume via nibabel
            - If 4D, take the first component
            - Compute an iso_value:
                    * TRACTVIEWER_NIFTI_ISO environment variable if set
                    * otherwise the mean (min+max)/2 after excluding NaNs
            - Build a pv.ImageData then apply contour
          - Applies the voxel->world affine (RAS+)
            - Optional conversion to LPS if TRACTVIEWER_NIFTI_COORD=LPS
        """
        try:
            import nibabel as nib
        except Exception as e:  # pragma: no cover
            raise ImportError("nibabel is required to load NIfTI files (.nii/.nii.gz): 'pip install nibabel'") from e
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        affine = img.affine  # voxel -> world
        dims = data.shape  # (nx, ny, nz)
        # Build an ImageData in voxel space (origin 0,0,0; spacing 1,1,1)
        grid = pv.ImageData()
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.dimensions = tuple([d if d > 1 else 1 for d in dims])
        # C ravel (ImageData directly in voxel coordinates) -> matches (x,y,z)
        grid.point_data["intensity"] = data.ravel(order="F")
        # Iso value
        iso_env = os.environ.get("TRACTVIEWER_NIFTI_ISO")
        try:
            iso_value = float(iso_env) if iso_env is not None else float((data.min() + data.max()) / 2.0)
        except ValueError:
            iso_value = float((data.min() + data.max()) / 2.0)
        if data.max() - data.min() <= 1e-6:
            warnings.warn("Nearly constant NIfTI volume: no surface can be extracted.")
            empty = grid.extract_points(grid.points)
            # Transform points to world anyway
            pts = empty.points
            pts_h = np.c_[pts, np.ones(len(pts))]
            empty.points = (pts_h @ affine.T)[:, :3]
            return empty
        # Isosurface
        try:
            surf = grid.contour([iso_value], scalars="intensity")
        except Exception as e:
            warnings.warn(f"Contour failed: {e}; falling back to extract_surface (coarse).")
            surf = grid.extract_surface()
        # Apply the full affine to the points (rotation + translation + anisotropy)
        pts = surf.points
        pts_h = np.c_[pts, np.ones(len(pts))]
        surf.points = (pts_h @ affine.T)[:, :3]
        # Store the flattened affine for later use (projection of other datasets)
        surf.field_data["nifti_affine"] = affine.astype(np.float32).ravel()
        surf.field_data["nifti_dims"] = np.array(dims, dtype=np.int32)
        # Optional RAS->LPS or LPS->RAS conversion (the affine is assumed to give RAS+)
        target_coord = os.environ.get("TRACTVIEWER_NIFTI_COORD", "RAS").upper()
        if target_coord not in ("RAS", "LPS"):
            target_coord = "RAS"
        if target_coord == "LPS":
            surf.points[:, 0] *= -1.0  # R -> L
            surf.points[:, 1] *= -1.0  # A -> P
            surf.field_data["coord_system"] = np.array(["LPS"], dtype=object)
        else:
            surf.field_data["coord_system"] = np.array(["RAS"], dtype=object)
        surf.field_data["nifti_iso_value"] = np.array([iso_value], dtype=np.float32)
        return surf

    def add(self, data: DataInput, params: Optional[ParamDict] = None):
        ds = self._load(data)
        params = dict(params or {})
        if "name" not in params:
            params["name"] = f"ds{len(self._datasets)}"
        # Remember the reference affine if NIfTI
        if self._reference_affine is None and "nifti_affine" in ds.field_data:
            flat = np.array(ds.field_data["nifti_affine"]).astype(float)
            if flat.size == 16:
                self._reference_affine = flat.reshape(4, 4)
                # Apply it right away to any already loaded, untransformed tractogram
                for existing_ds, _ in self._datasets:
                    self._maybe_project_tract(existing_ds)
        # Add the dataset
        self._datasets.append((ds, params))
        # If the affine already exists, project this tract when relevant
        self._maybe_project_tract(ds)
        # Invalidate the plotter if already built
        self._plotter = None
        return self

    def _maybe_project_tract(self, ds: pv.DataSet):
        """Project the points of a tractogram through the reference affine when available.

        Heuristic criteria for treating a dataset as a tract:
            - cell_data['streamline_id'] present AND non-empty lines.
        A field_data['tract_transformed'] flag prevents applying it twice.
        """
        if self._reference_affine is None:
            return
        if not isinstance(ds, pv.PolyData):  # safety
            return
        if "streamline_id" not in ds.cell_data:
            return
        if "tract_transformed" in ds.field_data:
            return
        try:
            pts = ds.points
            # Heuristic: detect whether the points look like voxel indices (integers within the volume bounds)
            dims = None
            for ref_ds, _ in self._datasets:
                if "nifti_dims" in ref_ds.field_data:
                    dims = np.array(ref_ds.field_data["nifti_dims"]).astype(int)
                    break
            is_voxel = False
            if dims is not None and dims.size >= 3:
                if np.all(pts.min(axis=0) >= -1) and np.all(pts.max(axis=0) <= dims[:3] + 2):
                    frac = np.abs(pts - np.round(pts))
                    frac_flat = frac.ravel()
                    if frac_flat.size > 0:
                        ratio_int_like = np.mean(frac_flat < 0.05)
                        if ratio_int_like > 0.8:
                            is_voxel = True
            if is_voxel:
                pts_h = np.c_[pts, np.ones(len(pts))]
                ds.points = (pts_h @ self._reference_affine.T)[:, :3]
            # Optional flip to LPS if requested and not done yet
            target_coord = os.environ.get("TRACTVIEWER_NIFTI_COORD", "RAS").upper()
            if target_coord == "LPS" and ds.field_data.get("coord_system", ["RAS"]) [0] != "LPS":
                # Apply the R->L, A->P flips
                ds.points[:, 0] *= -1.0
                ds.points[:, 1] *= -1.0
                ds.field_data["coord_system"] = np.array(["LPS"], dtype=object)
            ds.field_data["tract_transformed"] = np.array([1 if is_voxel else 0], dtype=np.int8)
        except Exception as e:
            warnings.warn(f"Tract projection failed: {e}")

    @classmethod
    def from_paths(cls, paths: Sequence[DataInput], params_list: Optional[Sequence[ParamDict]] = None, **kwargs):
        vis = cls(**kwargs)
        params_list = params_list or [{}] * len(paths)
        for p, prm in zip(paths, params_list):
            vis.add(p, prm)
        return vis

    # ------------------------------
    # Scene building
    # ------------------------------
    def _ensure_plotter(self):
        # Rebuild if the plotter is missing or already closed
        if self._plotter is not None and not getattr(self._plotter, "_closed", False):
            return
        # If an old closed plotter is still referenced, forget it
        if self._plotter is not None and getattr(self._plotter, "_closed", False):
            self._plotter = None
        self._plotter = pv.Plotter(off_screen=self.off_screen)
        self._plotter.set_background(self.background)
        # Recompute (in case the background changed before a new ensure)
        self._font_color = self._choose_font_color(self.background)
        self._show_scalar_bar_added = False
        for ds, prm in self._datasets:
            mesh = ds
            # Threshold if requested
            if "threshold" in prm and prm["threshold"]:
                arr_name, (vmin, vmax) = prm["threshold"]
                if arr_name not in mesh.array_names:
                    raise ValueError(f"Array '{arr_name}' not found for threshold.")
                mesh = mesh.threshold(value=(vmin, vmax), scalars=arr_name, invert=False)

            display_array = prm.get("display_array")
            if display_array and display_array not in mesh.array_names:
                raise ValueError(f"Array '{display_array}' not found in dataset ({mesh.array_names}).")
            if display_array:
                # Make sure the array is active (helps show_scalar_bar)
                try:
                    mesh.set_active_scalars(display_array)
                except Exception:
                    pass

            # Map parameters to add_mesh
            add_kwargs = dict(
                scalars=display_array,
                cmap=prm.get("cmap"),
                clim=prm.get("clim"),
                opacity=prm.get("opacity", 1.0),
                show_edges=prm.get("show_edges", False),
                # A single global scalar bar (first eligible dataset)
                show_scalar_bar=bool(display_array) and prm.get("show_scalar_bar", True) and not self._show_scalar_bar_added,
                name=prm.get("name"),
                color=prm.get("color"),
                style=prm.get("style"),  # 'surface' | 'wireframe' | 'points'
            )
            # Points rendering without point_size: apply a default value
            if add_kwargs.get("style") == "points" and "point_size" not in prm:
                add_kwargs["point_size"] = 5
            # Spherical point rendering option
            if "points_as_spheres" in prm:
                add_kwargs["points_as_spheres"] = prm["points_as_spheres"]
            if add_kwargs["show_scalar_bar"]:
                sb_args = prm.get("show_scalar_bar_args") or {}
                sb_defaults = {
                    "title": display_array or "",
                    "n_labels": 5,
                    "fmt": "%.2f",
                    # Unified text color (ticks + title)
                    "color": self._font_color,
                }
                sb_defaults.update(sb_args)
                add_kwargs["show_scalar_bar_args"] = sb_defaults
            for opt in ("point_size", "line_width", "ambient", "specular", "diffuse"):
                if opt in prm:
                    add_kwargs[opt] = prm[opt]
            self._plotter.add_mesh(mesh, **{k: v for k, v in add_kwargs.items() if v is not None})
            if add_kwargs.get("show_scalar_bar"):
                self._show_scalar_bar_added = True

        # Adjust the global camera
        if self._datasets:
            self._plotter.camera_position = "xy"  # simple initial orientation

    # ------------------------------
    # Robust camera rotations
    # ------------------------------
    def _rotate_camera(self, azimuth_deg: float = 0.0, elevation_deg: float = 0.0):
        """
        Rotate the camera with the VTK methods first,
        otherwise fall back to rotating the position around the focal point.
        """
        cam = self._plotter.camera
        # Azimuth
        if azimuth_deg:
            if callable(getattr(cam, "Azimuth", None)):
                cam.Azimuth(azimuth_deg)
            elif callable(getattr(cam, "azimuth", None)):
                cam.azimuth(azimuth_deg)
            else:
                self._rotate_camera_fallback(axis='z', angle_deg=azimuth_deg)
        # Elevation
        if elevation_deg:
            if callable(getattr(cam, "Elevation", None)):
                cam.Elevation(elevation_deg)
            elif callable(getattr(cam, "elevation", None)):
                cam.elevation(elevation_deg)
            else:
                self._rotate_camera_fallback(axis='x', angle_deg=elevation_deg)

    def _rotate_camera_fallback(self, axis: str, angle_deg: float):
        """Fallback: rotate the camera position around the focal point."""
        cam = self._plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        vec = pos - fp
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        if axis == 'z':
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
        elif axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]])
        elif axis == 'y':
            R = np.array([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]])
        else:
            return
        new_pos = fp + R.dot(vec)
        cam.position = new_pos.tolist()
        # Optional: view_up is left unchanged here (keeps things stable)

    def _apply_rotation(self, azimuth_deg: float, elevation_deg: float):
        """Rotate + render (required off-screen for the camera to reach the captures)."""
        self._rotate_camera(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        self._plotter.render()

    # ------------------------------
    # Interactive display
    # ------------------------------
    def show(self, **show_kwargs):
        self._ensure_plotter()
        if self.off_screen:
            warnings.warn("off_screen=True: no interactive window will open. Pass off_screen=False for interaction.")
        return self._plotter.show(**show_kwargs)

    # ------------------------------
    # Screenshot
    # ------------------------------
    def capture_screenshot(self, output_path: Union[str, Path], transparent: bool = False):
        self._ensure_plotter()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = self._plotter.screenshot(filename=str(output_path), transparent_background=transparent)
        return img

    # ------------------------------
    # Video / GIF recording
    # ------------------------------
    @staticmethod
    def _ffmpeg_encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
        """Return True if the ffmpeg encoder is available."""
        try:
            out = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode != 0:
                return False
            # Each encoders line starts with 2 flag letters + space + name
            token = f" {encoder}"
            return any(line.strip().endswith(encoder) or f" {encoder} " in line for line in out.stdout.splitlines())
        except Exception:
            return False

    @staticmethod
    def _select_ffmpeg_codec(ffmpeg_bin: str, requested: str) -> Tuple[str, Optional[str]]:
        """
        (KEPT for compatibility) -> returns the first available codec or a fallback.
        Now superseded by _codec_candidates, kept in case it is called elsewhere.
        """
        order = [
            requested,
            "libx264", "h264_nvenc", "h264_qsv", "h264_v4l2m2m", "h264_vaapi",
            "libx265", "hevc_nvenc", "hevc_qsv", "hevc_v4l2m2m", "hevc_vaapi",
            "libvpx_vp9", "libvpx_vp8",
            "mpeg4",
            "ffv1",   # lossless if available
        ]
        seen = set()
        for c in order:
            if c in seen:
                continue
            seen.add(c)
            if TractViewer._ffmpeg_encoder_available(ffmpeg_bin, c):
                if c == requested:
                    return c, None
                return c, f"Codec '{requested}' indisponible -> fallback '{c}'."
        # No codec from the list: let ffmpeg choose (no -c:v)
        return "", f"Codec '{requested}' unavailable, no fallback found -> using the ffmpeg default codec."

    @staticmethod
    def _codec_candidates(ffmpeg_bin: str, requested: str, prefer_software: bool = True) -> List[str]:
        """
        Return an ordered list of codecs to try.
        - With prefer_software=True, hardware codecs (nvenc, qsv, vaapi, v4l2m2m) go last.
        - The explicitly requested codec stays first (even if hardware).
        """
        hw_suffix = ("_nvenc", "_qsv", "_vaapi", "_v4l2m2m")
        # Software base (order of preference)
        software_set = [
            "libx264", "libopenh264", "mpeg4", "libxvid",
            "libvpx_vp9", "libvpx_vp8", "ffv1", "rawvideo"
        ]
        # Potential hardware encoders
        hardware_set = [
            "h264_nvenc", "hevc_nvenc",
            "h264_qsv", "hevc_qsv",
            "h264_vaapi", "hevc_vaapi",
            "h264_v4l2m2m", "hevc_v4l2m2m",
        ]
        ordered = [requested] if requested else []
        if prefer_software:
            ordered += [c for c in software_set if c not in ordered]
            ordered += [c for c in hardware_set if c not in ordered]
        else:
            # Mix requested -> hw -> software
            ordered += [c for c in hardware_set if c not in ordered]
            ordered += [c for c in software_set if c not in ordered]
        # Remove duplicates
        uniq = []
        for c in ordered:
            if c and c not in uniq:
                uniq.append(c)
        # Keep only the available ones
        available = [c for c in uniq if TractViewer._ffmpeg_encoder_available(ffmpeg_bin, c)]
        # Always add a fallback without codec (marker "")
        available.append("")  # "" => let ffmpeg decide
        return available

    def record_rotation(
        self,
        output_path: Union[str, Path],
        n_frames: int = 180,
        step: float = 2.0,
        elevation: float = 0.0,
        gif: bool = False,
        fps: int = 30,
        quality: int = 5,
        window_size: Optional[Tuple[int, int]] = None,
        supersample: int = 1,
        anti_aliasing: bool = True,
        codec: str = "libx264",
        bitrate: Optional[str] = None,
        crf: Optional[int] = None,
        auto_codec_fallback: bool = True,
        prefer_software: bool = True,
    ):
        """
        Rotate the camera in azimuth and record.
        gif=True => GIF via imageio.
        gif=False => MP4/AVI via:
            1) imageio-ffmpeg if available (direct stream)
            2) otherwise the ffmpeg binary if present
          3) sinon erreur explicite

        Additional quality parameters:
            window_size: (w,h) desired final size (without supersampling). If None, current size.
            supersample: factor (>1) to render at (w*factor, h*factor) before encoding (sharper).
            anti_aliasing: enable anti-aliasing (if supported).
            codec: ffmpeg/imageio video encoder (e.g. libx264, mpeg4, libvpx_vp9).
            bitrate: e.g. '8M' (if set, ffmpeg uses this bitrate).
            crf: target quality (0-51, lower = better) for x264/x265-like codecs (ignored if bitrate is set).
            quality: imageio internal setting (0-10), keep high (8-10) to limit loss.
          auto_codec_fallback: if True, automatically pick an available codec when the requested one is missing.
        """
        self._ensure_plotter()
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".gif" if gif else ".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolution / supersampling
        original_size = getattr(self._plotter, "window_size", None)
        target_size = None
        if window_size:
            w, h = window_size
            w = int(w)
            h = int(h)
            if supersample > 1:
                target_size = (w * supersample, h * supersample)
            else:
                target_size = (w, h)
            try:
                self._plotter.window_size = target_size
            except Exception:
                target_size = None  # silent fallback

        if anti_aliasing:
            with contextlib.suppress(Exception):
                self._plotter.enable_anti_aliasing()

        if gif and imageio is None:
            gif = False  # fallback (imageio is still needed for video)

        if gif:
            frames = []
            for i in range(n_frames):
                elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                frames.append(self._plotter.screenshot(return_img=True))
            imageio.mimsave(output_path, frames, fps=fps)
            # Restore the size
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # --- Video branch ---
        if imageio is None:
            raise RuntimeError("imageio unavailable. Install: pip install imageio imageio-ffmpeg, or a system ffmpeg.")

        # 1) Try imageio-ffmpeg
        has_imageio_ffmpeg = False
        try:
            import imageio_ffmpeg  # noqa: F401
            has_imageio_ffmpeg = True
        except Exception:
            pass

        if has_imageio_ffmpeg:
            # Direct stream encoding
            writer = None
            try:
                writer_kwargs = dict(fps=fps, quality=quality, codec=codec)
                if bitrate:
                    writer_kwargs["bitrate"] = bitrate
                out_params = []
                if crf is not None and not bitrate:
                    out_params += ["-crf", str(crf)]
                if out_params:
                    writer_kwargs["output_params"] = out_params
                writer = imageio.get_writer(str(output_path), fps=fps, quality=quality)
                # Replace with a writer using kwargs (compatibility with older versions)
                with contextlib.suppress(TypeError):
                    writer.close()
                    writer = imageio.get_writer(str(output_path), **writer_kwargs)
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    writer.append_data(frame)
            finally:
                if writer is not None:
                    with contextlib.suppress(Exception):
                        writer.close()
                self._plotter.close()
                # Allow a later rebuild (e.g. vis.show())
                self._plotter = None
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 2) ffmpeg binary fallback (temporary PNG frames)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            temp_dir = Path(tempfile.mkdtemp(prefix="vtkvis_frames_"))
            try:
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    imageio.imwrite(temp_dir / f"frame{i:05d}.png", frame)
                # Try several codecs
                if auto_codec_fallback:
                    candidates = self._codec_candidates(ffmpeg_bin, codec, prefer_software=prefer_software)
                else:
                    candidates = [codec]
                last_err = None
                for idx, cdc in enumerate(candidates):
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-framerate", str(fps),
                        "-i", str(temp_dir / "frame%05d.png"),
                    ]
                    if cdc:
                        if idx == 0 and cdc != codec:
                            warnings.warn(f"Codec '{codec}' indisponible -> fallback '{cdc}'.")
                        elif idx > 0:
                            warnings.warn(f"Tentative codec fallback '{cdc}'.")
                        cmd += ["-c:v", cdc]
                    if crf is not None and not bitrate and (cdc.startswith("libx264") or cdc.startswith("libx265") or cdc in ("libvpx_vp9", "")):
                        cmd += ["-crf", str(crf)]
                    if bitrate:
                        cmd += ["-b:v", bitrate]
                    if cdc.startswith("libx264") or cdc.startswith("libx265"):
                        cmd += ["-preset", "slow"]
                    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        last_err = None
                        break
                    last_err = f"Codec '{cdc or 'auto'}' failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                if last_err:
                    raise RuntimeError("ffmpeg failed after all fallbacks:\n" + last_err)
            finally:
                self._plotter.close()
                self._plotter = None
                with contextlib.suppress(Exception):
                    shutil.rmtree(temp_dir)
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 3) Explicit error
        raise RuntimeError(
            "No working video backend.\n"
            "- Install imageio-ffmpeg: pip install imageio-ffmpeg\n"
            "or\n"
            "- Install ffmpeg (e.g. apt-get install ffmpeg)\n"
            "Otherwise use gif=True to write a GIF."
        )

    # ------------------------------
    # Utilities
    # ------------------------------
    def list_arrays(self) -> Dict[str, List[str]]:
        out = {}
        for ds, prm in self._datasets:
            out[prm.get("name")] = list(ds.array_names)
        return out

    @staticmethod
    def _choose_font_color(bg) -> str:
        """
        Return 'black' or 'white' depending on the background luminance.
        Robust implementation that does not rely on pyvista.parse_color (missing in some versions).
        Accepte:
            - color names (if matplotlib is installed)
            - hex code (#RRGGBB or #RGB)
            - tuple/list (r,g,b[,a]) with 0-1 or 0-255 components
        """
        def _norm_rgb(rgb):
            if max(rgb) > 1.0:
                return [c / 255.0 for c in rgb]
            return rgb

        # Try matplotlib
        try:
            import matplotlib.colors as mcolors
            try:
                r, g, b = mcolors.to_rgb(bg)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                pass
        except Exception:
            pass

        # Tuples / lists
        if isinstance(bg, (tuple, list)) and 3 <= len(bg) <= 4:
            try:
                r, g, b = _norm_rgb(list(bg)[:3])
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                return "black"

        # Manual hex
        if isinstance(bg, str) and bg.startswith("#"):
            h = bg.lstrip("#")
            if len(h) == 3:
                h = "".join([c*2 for c in h])
            if len(h) == 6:
                try:
                    r = int(h[0:2], 16) / 255.0
                    g = int(h[2:4], 16) / 255.0
                    b = int(h[4:6], 16) / 255.0
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    return "black" if lum > 0.6 else "white"
                except Exception:
                    return "black"

        # Default fallback
        return "black"


# -------------------------------------------------
# Usage example (guarded by __main__)
# -------------------------------------------------
if __name__ == "__main__":
    #Set TRACTVIEWER_NIFTI_COORD=LPS in env to have nifti in LPS

    os.environ["TRACTVIEWER_NIFTI_COORD"] = "LPS"
    vis = TractViewer(background="black", off_screen=False)
    vis.add(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-associations_model-MCM_space-HCP_tracto.vtk",
        {
            "display_array": "point_index",
            "cmap": "viridis",
            "threshold": ("point_index", (0, 24)),
            "opacity": 0.9,
            "show_scalar_bar": True,  # mapped to show_scalar_bar
            "name": "associations",
        }
    ).add(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-centroids_model-MCM_space-subject_tracto.vtk",
        {
            "display_array": None,
            "color": "red",
            "point_size": 20,
            "opacity": 1.0,
            "name": "centroids",
            "style": "points",
            "points_as_spheres": True,
        }
    ).add(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz",
        {
            "display_array": "intensity",
            "cmap": "gray",
            "clim": (200, 800),
            "opacity": 0.3,
            "show_scalar_bar": False,
            "name": "anatomy",
            "ambient": 0.6,
            "specular": 0.1,
            "diffuse": 0.8,
            "style": "surface",
        }
    #Cli version : display_array=intensity,cmap=gray,opacity=0.3,show_scalar_bar=0,name=anatomy,ambient=0.6,specular=0.1,diffuse=0.8,style=surface
    ).add(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_AF_left.tck",
        {
            "display_array": "length_mm",
            "cmap": "plasma",
            "clim": (50, 150),
            "opacity": 0.6,
            "show_scalar_bar": True,
            "name": "summed_AF",
            "style": "surface",
            "ambient": 0.3,
            "specular": 0.2,
            "diffuse": 0.5,
        }
    )
    
    # vis.capture_screenshot("capture.png")
    # vis.record_rotation("rotation.mp4", n_frames=240, step=1.5, quality=10)
    vis.show()  # if interactive rendering is possible
    # vis.show()  # if interactive rendering is possible
