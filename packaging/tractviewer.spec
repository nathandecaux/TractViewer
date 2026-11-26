# PyInstaller spec of the desktop app (onedir build; .app bundle on macOS).
#   python packaging/fetch_vendor.py           # optional: bundle vtk.js (offline first launch)
#   pyinstaller --noconfirm packaging/tractviewer.spec
# Output: dist/TractViewer/ (Windows, Linux) or dist/TractViewer.app (macOS).
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
PKG = Path(SPECPATH)
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "tractviewer" / "_version.py").read_text()).group(1)

# The checkout must be importable here: the collect_* helpers import the package to locate it,
# and silently return nothing when it is not installed (CI uninstalls it).
sys.path.insert(0, str(ROOT))

# Package data files (vtkjs_template.html...), taken from the checkout by path.
datas = [(str(f), str(f.parent.relative_to(ROOT)))
         for f in (ROOT / "tractviewer").rglob("*")
         if f.is_file() and f.suffix not in (".py", ".pyc") and "__pycache__" not in f.parts]
if not any(Path(src).name == "vtkjs_template.html" for src, _ in datas):
    raise SystemExit("tractviewer/web/vtkjs_template.html not found: the interface would not load")
binaries = []
hiddenimports = collect_submodules("tractviewer") + collect_submodules("imageio.plugins")

vendor = PKG / "build" / "vendor"
if vendor.is_dir() and any(vendor.iterdir()):
    datas.append((str(vendor), "tractviewer_vendor"))
else:
    print("WARNING: packaging/build/vendor missing: vtk.js will be downloaded at first launch")

if sys.platform != "win32":
    d, b, h = collect_all("imageio_ffmpeg")  # ffmpeg binary for MP4 export
    datas += d; binaries += b; hiddenimports += h

# Only one Qt binding may be collected. Linux uses PyQt6 (Qt WebEngine) for pywebview;
# Windows (WebView2) and macOS (WKWebView) need no Qt at all.
excludes = ["PyQt5", "PySide2", "PySide6", "tkinter", "IPython", "jupyter", "notebook",
            "trame", "trame_vtk", "trame_vuetify", "trame_server", "pytest",
            # optional imports of pyvista / imageio / matplotlib that are never used here
            # (only matter when building from a large environment, e.g. conda)
            "torch", "triton", "nvidia", "tensorflow", "jax", "onnxruntime", "numba", "llvmlite",
            "SimpleITK", "itk", "pyarrow", "pandas", "panel", "bokeh", "playwright", "vl_convert",
            "av", "dask", "distributed", "sklearn", "skimage", "cv2", "sympy", "h5py", "zmq"]
if sys.platform.startswith("linux"):
    hiddenimports += ["webview.platforms.qt", "qtpy", "PyQt6.QtWebEngineWidgets",
                      "PyQt6.QtWebEngineCore", "PyQt6.QtWebChannel", "PyQt6.QtNetwork"]
else:
    excludes += ["PyQt6", "qtpy"]

icon = str(PKG / "icon.png")  # converted to .ico / .icns by PyInstaller (needs Pillow)

a = Analysis(
    [str(PKG / "tractviewer_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TractViewer",
    console=sys.platform.startswith("linux"),  # no console window on Windows / macOS
    icon=icon,
    upx=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="TractViewer", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="TractViewer.app",
        icon=icon,
        bundle_identifier="io.github.nathandecaux.tractviewer",
        version=VERSION,
        info_plist={
            "CFBundleName": "TractViewer",
            "CFBundleDisplayName": "TractViewer",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
