"""Mesh and camera rotations (pure numpy geometry)."""
from __future__ import annotations
import math
import numpy as np
import pyvista as pv


def _rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # z


def rotate_points(points: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """Apply successive X, Y, Z rotations (degrees) to a point cloud."""
    for axis, angle in (("x", rx), ("y", ry), ("z", rz)):
        if angle:
            points = points @ _rotation_matrix(axis, angle).T
    return points


def rotate_camera(camera, azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> None:
    """Rotate the camera with VTK when available, else manually around the focal point."""
    if azimuth_deg:
        if callable(getattr(camera, "Azimuth", None)):
            camera.Azimuth(azimuth_deg)
        elif callable(getattr(camera, "azimuth", None)):
            camera.azimuth(azimuth_deg)
        else:
            _rotate_camera_manual(camera, "z", azimuth_deg)
    if elevation_deg:
        if callable(getattr(camera, "Elevation", None)):
            camera.Elevation(elevation_deg)
        elif callable(getattr(camera, "elevation", None)):
            camera.elevation(elevation_deg)
        else:
            _rotate_camera_manual(camera, "x", elevation_deg)


def _rotate_camera_manual(camera, axis: str, angle_deg: float) -> None:
    """Rotate the camera position around the focal point about a world axis."""
    pos = np.array(camera.position)
    focal = np.array(camera.focal_point)
    new_vec = _rotation_matrix(axis, angle_deg) @ (focal - pos)
    camera.position = focal - new_vec
