"""Regenerate the images of the README and docs.

    python docs/make_screenshots.py                       # synthetic demo data
    python docs/make_screenshots.py anat.nii.gz:cmap=gray bundle.tck:display_array=length_mm,cmap=plasma

Outputs (overwritten):
    docs/web-interface.png   the interface (desktop app / web server), captured in headless Chrome
    docs/rotation.gif        headless pyvista rotation (tractograms and meshes only)

Files use the command line syntax (path:key=val,...). Without files, a synthetic volume and
tractogram are generated, so the script runs anywhere; real data gives nicer pictures.

Requirements: pip install playwright, plus a Chromium-based browser: an installed Chrome / Chromium
/ Edge is found automatically (or --browser PATH), otherwise `python -m playwright install chromium`.
"""
from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # the checkout, even if another version is installed

from tractviewer import TractViewer  # noqa: E402
from tractviewer.cli import _add_inputs  # noqa: E402
from tractviewer.params import parse_spec  # noqa: E402
from tractviewer.web.server import WebApp, make_server  # noqa: E402

DOCS = ROOT / "docs"
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe")


# ------------------------------
# Synthetic demo data
# ------------------------------
def demo_data(folder: Path) -> list:
    """Brain-like volume (2 mm) and a tractogram of three bundles, in the same RAS frame."""
    import nibabel as nib
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    shape = (80, 96, 80)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -np.array(shape) + 1  # centered on 0
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = (ijk @ affine[:3, :3].T + affine[:3, 3]).reshape(*shape, 3)
    r = np.sqrt((xyz[..., 0] / 68) ** 2 + (xyz[..., 1] / 85) ** 2 + ((xyz[..., 2] - 5) / 62) ** 2)
    folds = gaussian_filter(rng.standard_normal(shape), 2.0)
    brain = (r < 1).astype(float)
    wm = (r + 0.08 * folds < 0.72).astype(float)
    ventricles = (np.sqrt((np.abs(xyz[..., 0]) - 8) ** 2 / 36 + xyz[..., 1] ** 2 / 600
                          + (xyz[..., 2] - 12) ** 2 / 80) < 1).astype(float)
    img = gaussian_filter(0.35 * brain + 0.45 * wm - 0.6 * ventricles * brain, 1.0)
    img = np.clip(img + 0.03 * rng.standard_normal(shape) * brain, 0, None).astype(np.float32)
    vol = folder / "demo_t1.nii.gz"
    nib.save(nib.Nifti1Image(img, affine), vol)

    t = np.linspace(0, 1, 60)[:, None]
    lines = []
    for _ in range(1500):  # corpus callosum: left-right arcs
        y, z0, h = rng.normal(0, 20), rng.normal(18, 3), rng.uniform(15, 30)
        x = (t - 0.5) * rng.uniform(90, 115)
        lines.append(np.hstack([x, np.full_like(t, y) + rng.normal(0, 1.5),
                                z0 + h * np.sin(np.pi * t) - 10]))
    for side in (-1, 1):  # corticospinal tracts: vertical fans
        for _ in range(900):
            top = np.array([side * rng.uniform(15, 45), rng.normal(-10, 10), rng.uniform(50, 58)])
            bottom = np.array([side * 10 + rng.normal(0, 2), rng.normal(-15, 2), -55])
            lines.append(bottom + (top - bottom) * t ** 1.3 + rng.normal(0, 0.4, (len(t), 3)))
    for side in (-1, 1):  # arcuate fasciculus: front-back hooks
        for _ in range(700):
            a = np.pi * (0.15 + 1.1 * t)
            c = np.array([side * rng.uniform(32, 40), -10, 5])
            lines.append(c + np.hstack([np.zeros_like(t), -48 * np.cos(a) + rng.normal(0, 1.5),
                                        32 * np.sin(a) - 15 + rng.normal(0, 1.5)]))
    tck = folder / "demo_bundles.tck"
    tractogram = nib.streamlines.Tractogram([s.astype(np.float32) for s in lines],
                                            affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(tck))
    return [f"{vol}:cmap=gray", f"{tck}:display_array=length_mm,cmap=plasma"]


# ------------------------------
# Captures
# ------------------------------
def find_browser(explicit) -> str | None:
    if explicit:
        return explicit
    for name in BROWSERS:
        path = shutil.which(name) or (name if Path(name).is_file() else None)
        if path:
            return path
    return None  # Playwright's own Chromium


def capture_interface(inputs, output: Path, size, scale: float, browser, wait_ms: int) -> None:
    from playwright.sync_api import sync_playwright

    vis = TractViewer(off_screen=True)
    _add_inputs(vis, inputs, marching_cubes=False)  # like `tractviewer web`: NIfTI as slices
    app = WebApp(vis, title="TractViewer")
    httpd = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            chrome = pw.chromium.launch(
                executable_path=find_browser(browser),
                # software WebGL: works without a GPU (CI, servers)
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            page = chrome.new_page(viewport={"width": size[0], "height": size[1]},
                                   device_scale_factor=scale)
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(f"http://127.0.0.1:{httpd.server_port}/")
            page.wait_for_selector("#loading", state="hidden", timeout=180_000)
            page.wait_for_timeout(wait_ms)  # 2D views (slab extraction) and last frames
            if errors:
                print("page errors:\n  " + "\n  ".join(errors), file=sys.stderr)
            page.screenshot(path=str(output))
            chrome.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.cleanup()
    print(f"interface: {output}")


def record_rotation(inputs, output: Path, size, frames: int, fps: int) -> None:
    meshes = [i for i in inputs if not parse_spec(i)[0].lower().endswith((".nii", ".nii.gz"))]
    if not meshes:
        print("rotation: skipped (no tractogram or mesh)")
        return
    vis = TractViewer(background="white", off_screen=True, window_size=size)
    _add_inputs(vis, meshes)
    vis.record(output, n_frames=frames, fps=fps)
    print(f"rotation: {output}")


def _size(text: str):
    w, h = map(int, text.lower().split("x"))
    return w, h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="files (path:key=val,...); default: synthetic demo data")
    p.add_argument("--only", choices=("interface", "rotation"), help="generate one image only")
    p.add_argument("--size", type=_size, default=(1440, 860), help="interface viewport (default 1440x860)")
    p.add_argument("--scale", type=float, default=1.0, help="device pixel ratio of the capture (2 = retina)")
    p.add_argument("--wait", type=int, default=2500, help="ms to wait after loading (default 2500)")
    p.add_argument("--browser", help="Chrome / Chromium executable (default: auto-detected)")
    p.add_argument("--gif-size", type=_size, default=(480, 480))
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--out-dir", type=Path, default=DOCS)
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="tractviewer_docs_") as tmp:
        inputs = args.inputs or demo_data(Path(tmp))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.only in (None, "interface"):
            capture_interface(inputs, args.out_dir / "web-interface.png", args.size, args.scale,
                              args.browser, args.wait)
        if args.only in (None, "rotation"):
            record_rotation(inputs, args.out_dir / "rotation.gif", args.gif_size, args.frames, args.fps)


if __name__ == "__main__":
    main()
