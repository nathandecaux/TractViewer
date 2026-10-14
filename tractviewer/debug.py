"""Debug helpers: enabled by default, set TRACTVIEWER_DEBUG=0 to disable."""
from __future__ import annotations
import os
import sys
import traceback


def debug_enabled() -> bool:
    return os.environ.get("TRACTVIEWER_DEBUG", "1") != "0"


def warn_debug(msg: str, exc=None, extra=None) -> None:
    """Print a debug message (plus traceback if `exc` is given) to stderr."""
    if not debug_enabled():
        return
    full = f"[TractViewer DEBUG] {msg}"
    if extra:
        full += f" | {extra}"
    print(full, file=sys.stderr)
    if exc:
        traceback.print_exception(*exc, file=sys.stderr)


def env_info(backend_class=None, off_screen=None) -> str:
    """One-line summary of the environment (Python, pyvista, backend, DISPLAY)."""
    parts = [f"Python {sys.version.split()[0]}"]
    try:
        import pyvista
        parts.append(f"pyvista {pyvista.__version__}")
    except Exception:
        parts.append("pyvista not installed")
    if backend_class:
        parts.append(f"VTK backend: {backend_class}")
    if off_screen is not None:
        parts.append(f"off_screen={off_screen}")
    display = os.environ.get("DISPLAY")
    parts.append(f"DISPLAY={display}" if display else "no DISPLAY")
    return " | ".join(parts)
