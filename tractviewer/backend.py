"""VTK backend detection and font color selection based on the background."""
from __future__ import annotations
from typing import Tuple


def detect_vtk_backend() -> Tuple[str, bool]:
    """Return (class_name, is_headless). is_headless=True for OSMesa/EGL."""
    try:
        import vtk
        rw = vtk.vtkRenderWindow()
        name = rw.GetClassName() or "unknown"
        cls = name.lower()
        if "osopengl" in cls or "osmesa" in cls or "egl" in cls:
            return name, True
        return name, False
    except Exception:
        return "unknown", False


def choose_font_color(bg) -> str:
    """Return 'black' or 'white' depending on the background luminance."""
    rgb = _to_rgb(bg)
    if rgb is None:
        return "black"
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.6 else "white"


def _to_rgb(bg):
    """Convert a color (name, hex, tuple) to normalized (r, g, b) in [0, 1], or None."""
    try:
        import matplotlib.colors as mcolors
        return mcolors.to_rgb(bg)
    except Exception:
        pass

    if isinstance(bg, (tuple, list)) and 3 <= len(bg) <= 4:
        vals = list(bg)[:3]
        if max(vals) > 1.0:
            vals = [c / 255.0 for c in vals]
        return tuple(vals)

    if isinstance(bg, str) and bg.startswith("#"):
        h = bg.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                return None
    return None
