"""Desktop window (pywebview) around the vtk.js web interface.

The web server runs on 127.0.0.1 (free port) in a background thread and the
page is shown in a native window using the system webview (WebKitGTK / Qt on
Linux, WebView2 on Windows, WKWebView on macOS): no browser needed.

The JS libraries (vtk.js, gifenc) are downloaded once into a user cache folder
and served locally, so the app works offline afterwards. If they cannot be
downloaded, the page falls back to the CDN.

Requires: pip install "tractviewer[desktop]"
"""
from __future__ import annotations
import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Optional

from tractviewer.web import vtkjs
from tractviewer.web.server import WebApp, make_server

VENDOR_LIBS = {
    "vtkjs_url": vtkjs.VTKJS_URL,
    "gifenc_url": vtkjs.GIFENC_URL,
}


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "tractviewer" / "vendor"


def _local_name(url: str) -> str:
    # e.g. https://cdn.jsdelivr.net/npm/vtk.js@37.1.1/vtk.js -> vtk.js@37.1.1-vtk.js
    parts = url.split("/npm/", 1)[-1].split("/")
    return f"{parts[0]}-{parts[-1]}"


def ensure_vendor(folder: Path) -> dict:
    """Download missing libraries into `folder`. Returns {kwarg: url} for WebApp
    (local /vendor/ URL when available, CDN URL otherwise)."""
    folder.mkdir(parents=True, exist_ok=True)
    urls = {}
    for key, url in VENDOR_LIBS.items():
        name = _local_name(url)
        target = folder / name
        if not target.is_file():
            try:
                print(f"Downloading {url} ...")
                tmp = target.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                    fh.write(resp.read())
                tmp.replace(target)
            except OSError as exc:
                print(f"Warning: could not download {url} ({exc}); using the CDN.", file=sys.stderr)
                urls[key] = url
                continue
        urls[key] = f"/vendor/{name}"
    return urls


def _has(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:  # parent package missing
        return False


def _pick_gui() -> Optional[str]:
    """Linux: GTK when PyGObject is installed, otherwise Qt (pip wheels). None = pywebview default."""
    if not sys.platform.startswith("linux") or _has("gi"):
        return None
    # qtpy picks PyQt5 first when installed, often without WebEngine.
    if "QT_API" not in os.environ and _has("PyQt6.QtWebEngineWidgets"):
        os.environ["QT_API"] = "pyqt6"
    return "qt"


def run(viewer, title: str = "TractViewer", max_streamlines: Optional[int] = None,
        max_voxels: int = vtkjs.DEFAULT_MAX_VOXELS, width: int = 1400, height: int = 900,
        debug: bool = False) -> None:
    """Open the web interface in a native window (blocking until it is closed)."""
    gui = _pick_gui()
    if not debug:  # Qt / Chromium are very verbose on some setups (conda, Wayland)
        os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.wayland.textinput=false")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=3")
    try:
        import webview
    except ImportError:
        raise SystemExit("pywebview is required for the desktop mode: "
                         "pip install \"tractviewer[desktop]\"") from None

    vendor = cache_dir()
    urls = ensure_vendor(vendor)
    app = WebApp(viewer, title=title, max_streamlines=max_streamlines, max_voxels=max_voxels,
                 vendor_dir=vendor, **urls)
    httpd = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # PNG / GIF / HTML exports use <a download>: disabled by default in pywebview.
    if hasattr(webview, "settings"):
        webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window(title, f"http://127.0.0.1:{httpd.server_port}/",
                          width=width, height=height)
    try:
        webview.start(gui=gui, debug=debug)
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.cleanup()
