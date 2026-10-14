"""Former private methods of TractViewer (from before the split into modules).

Kept for existing scripts: each one delegates to the module that now holds the
implementation and emits a TractViewerDeprecationWarning.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from tractviewer import backend, debug, geometry, loaders, recording
from tractviewer._deprecation import deprecated


class LegacyMethodsMixin:
    # ---- Loading ----
    @staticmethod
    @deprecated("tractviewer.loaders.load", old="TractViewer._load")
    def _load(obj, marching_cubes: bool = True, smooth_iterations: int = 0):
        from tractviewer.core import default_coord_system, default_iso_value
        return loaders.load(obj, marching_cubes=marching_cubes, smooth_iterations=smooth_iterations,
                            iso_value=default_iso_value(), coord_system=default_coord_system())

    @staticmethod
    @deprecated("tractviewer.loaders.load_tractogram", old="TractViewer._load_tract_file")
    def _load_tract_file(path):
        return loaders.load_tractogram(Path(path))

    @staticmethod
    @deprecated("tractviewer.loaders.load_nifti", old="TractViewer._load_nifti_surface")
    def _load_nifti_surface(path, marching_cubes: bool = True, smooth_iterations: int = 0):
        from tractviewer.core import default_coord_system, default_iso_value
        return loaders.load_nifti(Path(path), marching_cubes=marching_cubes,
                                  smooth_iterations=smooth_iterations,
                                  iso_value=default_iso_value(), coord_system=default_coord_system())

    @staticmethod
    @deprecated("tractviewer.loaders.load_png", old="TractViewer._load_png_as_vtk")
    def _load_png_as_vtk(path):
        return loaders.load_png(Path(path))

    # ---- Recording ----
    @staticmethod
    @deprecated("tractviewer.recording.write_gif", old="TractViewer._write_gif_with_ffmpeg")
    def _write_gif_with_ffmpeg(frames, output_path, fps: int = 10, scale_width=None, optimize: bool = True):
        return recording._gif_with_ffmpeg(frames, output_path, fps=fps, scale_width=scale_width)

    @staticmethod
    @deprecated("tractviewer.recording", old="TractViewer._ffmpeg_encoder_available")
    def _ffmpeg_encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
        return recording._encoder_available(ffmpeg_bin, encoder)

    @staticmethod
    @deprecated("tractviewer.recording", old="TractViewer._codec_candidates")
    def _codec_candidates(ffmpeg_bin: str, requested: str, prefer_software: bool = True):
        return recording._codec_candidates(ffmpeg_bin, requested, prefer_software)

    @staticmethod
    @deprecated("tractviewer.recording", old="TractViewer._select_ffmpeg_codec")
    def _select_ffmpeg_codec(ffmpeg_bin: str, requested: str):
        codec = recording._codec_candidates(ffmpeg_bin, requested)[0]
        if codec == requested:
            return codec, None
        return codec, f"Codec '{requested}' indisponible -> fallback '{codec or 'auto'}'."

    # ---- Backend / debug ----
    @staticmethod
    @deprecated("tractviewer.backend.choose_font_color", old="TractViewer._choose_font_color")
    def _choose_font_color(bg) -> str:
        return backend.choose_font_color(bg)

    @staticmethod
    @deprecated("tractviewer.backend.detect_vtk_backend", old="TractViewer._detect_vtk_backend")
    def _detect_vtk_backend():
        return backend.detect_vtk_backend()

    @deprecated("tractviewer.debug.warn_debug", old="TractViewer._warn_debug")
    def _warn_debug(self, msg: str, exc=None, extra=None):
        debug.warn_debug(msg, exc=exc, extra=extra)

    @deprecated("tractviewer.debug.env_info", old="TractViewer._debug_env_info")
    def _debug_env_info(self) -> str:
        return self._env_info()

    # ---- Rotations / scene ----
    @deprecated("tractviewer.geometry.rotate_camera", old="TractViewer._rotate_camera")
    def _rotate_camera(self, azimuth_deg: float = 0.0, elevation_deg: float = 0.0):
        geometry.rotate_camera(self._plotter.camera, azimuth_deg, elevation_deg)

    @deprecated("tractviewer.geometry.rotate_camera", old="TractViewer._rotate_camera_fallback")
    def _rotate_camera_fallback(self, axis: str = "z", angle_deg: float = 0.0):
        geometry._rotate_camera_manual(self._plotter.camera, axis, angle_deg)

    @deprecated("TractViewer._apply_camera_rotation", old="TractViewer._apply_rotation")
    def _apply_rotation(self, azimuth_deg: float, elevation_deg: float):
        self._apply_camera_rotation(azimuth_deg, elevation_deg)

    @deprecated("tractviewer.geometry.rotate_points", old="TractViewer._apply_mesh_rotation")
    def _apply_mesh_rotation(self, dataset, rx: float, ry: float, rz: float):
        dataset.points = geometry.rotate_points(np.asarray(dataset.points).copy(), rx, ry, rz)

    @deprecated("TractViewer._populate", old="TractViewer._populate_plotter")
    def _populate_plotter(self):
        self._populate()

    @deprecated("TractViewer._populate", old="TractViewer._rebuild_actors")
    def _rebuild_actors(self):
        if self._plotter is None:
            return
        camera = self._plotter.camera_position
        self._plotter.clear()
        self._populate()
        self._plotter.camera_position = camera
