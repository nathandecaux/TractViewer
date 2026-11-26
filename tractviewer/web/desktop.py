"""Desktop window (pywebview) around the vtk.js web interface.

The web server runs on 127.0.0.1 (free port) in a background thread and the
page is shown in a native window using the system webview (WebKitGTK / Qt on
Linux, WebView2 on Windows, WKWebView on macOS): no browser needed.

The JS libraries (vtk.js, gifenc) are downloaded once into a user cache folder
and served locally, so the app works offline afterwards. If they cannot be
downloaded, the page falls back to the CDN.

In the pre-compiled app, the libraries are bundled (packaging/fetch_vendor.py) and a newer
GitHub release is offered at startup (tractviewer.updater).

Requires pywebview (installed with tractviewer).
"""
from __future__ import annotations
import json
import os
import shutil
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


def bundled_vendor_dir() -> Optional[Path]:
    """JS libraries shipped inside the pre-compiled app (PyInstaller), if any."""
    base = getattr(sys, "_MEIPASS", None)
    folder = Path(base) / "tractviewer_vendor" if base else None
    return folder if folder is not None and folder.is_dir() else None


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


# Filters of the save dialog, by extension.
FILE_TYPES = {".png": "PNG image", ".gif": "GIF animation", ".html": "HTML page"}


def _save_dialog(filename: str, tmp: Path) -> Optional[str]:
    """Native "Save as" dialog (called from a server thread: pywebview marshals it to the GUI
    thread), then move the file made by the page there. Returns the path, None if cancelled."""
    import webview
    suffix = Path(filename).suffix.lower()
    file_types = ((f"{FILE_TYPES[suffix]} (*{suffix})",) if suffix in FILE_TYPES else ()) + ("All files (*.*)",)
    dialog = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or webview.SAVE_DIALOG
    window = webview.windows[0]
    result = window.create_file_dialog(dialog, directory=str(Path.home()), save_filename=filename,
                                       file_types=file_types)
    if not result:
        return None
    target = Path(result if isinstance(result, str) else result[0])
    if suffix and not target.suffix:
        target = target.with_suffix(suffix)
    shutil.move(str(tmp), str(target))
    return str(target)


def _banner(window, text: Optional[str]) -> None:
    """Small message in the bottom-right corner of the page (None removes it)."""
    js = """(function(t){var b=document.getElementById('tv-update');
if(t===null){if(b)b.remove();return;}
if(!b){b=document.createElement('div');b.id='tv-update';
b.style.cssText='position:fixed;right:12px;bottom:12px;z-index:2147483647;padding:8px 12px;'+
'border-radius:6px;background:#222;color:#fff;font:13px sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.3)';
document.body.appendChild(b);}b.textContent=t;})(%s)""" % json.dumps(text)
    try:
        window.evaluate_js(js)
    except Exception:  # noqa: BLE001 - window closed meanwhile
        pass


def _appimage_integration(window) -> None:
    """Linux AppImage, first launch: offer to install it in a folder + application menu entry."""
    from tractviewer import linux_integration as li
    if not li.should_offer():
        li.refresh()
        return
    window.events.loaded.wait(30)
    if not window.create_confirmation_dialog(
            "Install TractViewer",
            "Install TractViewer on this computer?\n\n"
            "You will choose the installation folder (default: ~/Applications), and TractViewer "
            "will be added to the applications menu.\n\n"
            "Cancel: run without installing (you will not be asked again)."):
        li.decline()
        return
    import webview
    default = li.default_dir()
    default.mkdir(parents=True, exist_ok=True)
    dialog = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
    result = window.create_file_dialog(dialog, directory=str(default))
    if not result:
        _banner(window, "Installation cancelled (you will be asked again at the next launch).")
        return
    folder = Path(result if isinstance(result, str) else result[0])
    source = li.current_appimage()
    try:
        _banner(window, f"Installing in {folder}...")
        target = li.install(folder)
    except Exception as exc:  # noqa: BLE001 - read-only folder, disk full...
        _banner(window, f"Installation failed: {exc}")
        return
    _banner(window, f"TractViewer installed: {target} (applications menu entry created).")
    if source is not None and source.resolve() != target.resolve() and window.create_confirmation_dialog(
            "Installation done",
            f"TractViewer is installed in {target} and available in the applications menu.\n\n"
            f"Delete the downloaded file {source}?"):
        try:
            source.unlink()
        except OSError:
            pass


def _startup(window, check_updates: bool) -> None:
    """Background thread started with the window: AppImage integration, then update check."""
    try:
        _appimage_integration(window)
    except Exception as exc:  # noqa: BLE001 - never break the app for this
        print(f"Warning: desktop integration failed ({exc})", file=sys.stderr)
    if check_updates:
        _update_check(window)


def _update_check(window) -> None:
    """Pre-compiled app only: offer to install a newer GitHub release (background thread)."""
    from tractviewer import updater
    kind = updater.install_kind()
    if kind is None:
        return
    update = updater.check(kind)
    if update is None:
        return
    window.events.loaded.wait(30)
    size = f" ({update.size / 1e6:.0f} MB)" if update.size else ""
    if not window.create_confirmation_dialog(
            "Update available",
            f"TractViewer {update.version} is available (current: {updater.__version__}).\n\n"
            f"Download{size} and install it now? The application will restart."):
        return
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else -1
        if pct != last[0]:
            last[0] = pct
            _banner(window, f"Downloading TractViewer {update.version}... "
                            + (f"{pct}%" if pct >= 0 else f"{done / 1e6:.0f} MB"))

    try:
        file = updater.download(update, progress)
        _banner(window, "Installing the update...")
        updater.apply(file, kind)
    except Exception as exc:  # noqa: BLE001
        _banner(window, f"Update failed: {exc}")
        return
    window.destroy()


def run(viewer, title: str = "TractViewer", max_streamlines: Optional[int] = None,
        max_voxels: int = vtkjs.DEFAULT_MAX_VOXELS, width: int = 1400, height: int = 900,
        debug: bool = False, check_updates: bool = True) -> None:
    """Open the web interface in a native window (blocking until it is closed)."""
    gui = _pick_gui()
    if not debug:  # Qt / Chromium are very verbose on some setups (conda, Wayland)
        os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.wayland.textinput=false")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=3")
    try:
        import webview
    except ImportError:
        raise SystemExit("pywebview is required for the desktop mode: "
                         "pip install pywebview  (Linux: \"pywebview[qt]\")") from None

    vendor = bundled_vendor_dir() or cache_dir()
    urls = ensure_vendor(vendor)
    app = WebApp(viewer, title=title, max_streamlines=max_streamlines, max_voxels=max_voxels,
                 vendor_dir=vendor, save_handler=_save_dialog, **urls)
    httpd = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    window = webview.create_window(title, f"http://127.0.0.1:{httpd.server_port}/",
                                   width=width, height=height)
    try:
        webview.start(_startup, (window, check_updates), gui=gui, debug=debug)
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.cleanup()
