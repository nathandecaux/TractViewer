"""Memory budget of tractograms: automatic subsampling, like the default
subsampling of MI-Brain, so that a large tractogram never exhausts the RAM.

The number of points of a .trk/.tck file is estimated from its header and its
size (no full read), and compared with what the machine can afford:

    affordable points = MEMORY_FRACTION * available RAM / BYTES_PER_POINT

BYTES_PER_POINT is the measured peak memory of the whole web pipeline (Python
loading and serialization + browser: decoded arrays, GPU buffers), on synthetic
tractograms of 1 to 40 M points. If the file holds more points, streamlines are
drawn at random (fixed seed) while the file is read lazily: the full tractogram
is never held in memory.

Override: per file `max_points=N` (0 = no limit), on the command line
`--max-points N`, or the TRACTVIEWER_MAX_POINTS environment variable.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Peak memory of the viewer per streamline point, Python + browser (measured: ~216 B/point
# between 10 M and 40 M points, browser with software WebGL, i.e. GPU buffers in RAM), + margin.
BYTES_PER_POINT = 250
# Share of the available RAM the viewer may use (the rest stays for the system).
MEMORY_FRACTION = 0.5
# Never subsample below this (a tractogram this small always fits).
MIN_POINTS = 1_000_000
# Upper bound whatever the RAM: beyond it loading and rendering are no longer interactive
# (40 M points: ~11 s to load, ~7 GB in the browser).
MAX_POINTS = 30_000_000


def available_memory() -> Optional[int]:
    """Available RAM in bytes (None if unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    try:  # macOS / other POSIX: free pages only (pessimistic)
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def affordable_points(available: Optional[int] = None) -> int:
    """Number of tractogram points the viewer can afford for one tractogram."""
    available = available_memory() if available is None else available
    if available is None:
        available = 8 * 2**30  # unknown: assume a modest machine
    points = int(MEMORY_FRACTION * available / BYTES_PER_POINT)
    return max(MIN_POINTS, min(MAX_POINTS, points))


def estimate_points(path: Path, header) -> Tuple[Optional[int], Optional[int]]:
    """(streamlines, points) of a .trk/.tck file from its header and size, without reading it."""
    size = Path(path).stat().st_size
    count = header.get("nb_streamlines") or header.get("count")  # .tck: "count" field
    try:
        count = int(count) if int(count) > 0 else None
    except (TypeError, ValueError):
        count = None
    suffix = Path(path).suffix.lower()
    if suffix == ".trk":
        n_scalars = int(header.get("nb_scalars_per_point", 0) or 0)
        n_props = int(header.get("nb_properties_per_streamline", 0) or 0)
        data = size - 1000 - (count or 0) * 4 * (1 + n_props)
        return count, max(0, data // (4 * (3 + n_scalars)))
    if suffix == ".tck":
        offset = int(header.get("_offset_data", 0) or 0)
        # float32 triplets, one NaN triplet after each streamline, an Inf triplet at the end
        return count, max(0, (size - offset) // 12 - (count or 0) - 1)
    return count, size // 12
