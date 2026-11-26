"""Download the JS libraries (vtk.js, gifenc) into packaging/build/vendor so that the
pre-compiled app works offline from the first launch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # checkout, not installed
from tractviewer.web.desktop import ensure_vendor  # noqa: E402

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "build" / "vendor"
    urls = ensure_vendor(folder)
    missing = [u for u in urls.values() if not u.startswith("/vendor/")]
    if missing:
        raise SystemExit(f"could not download: {missing}")
    print(f"vendor libraries in {folder}: {sorted(p.name for p in folder.iterdir())}")
