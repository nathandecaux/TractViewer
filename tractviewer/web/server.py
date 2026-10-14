"""vtk.js web server (standard library only): page, dataset upload and removal.

  GET    /                      vtk.js page (current scene embedded)
  POST   /api/upload            body = file content
                                header: X-Filename (URI encoded)
                                -> {layers, volumes, updated, colormaps}
  DELETE /api/datasets/<name>   remove a dataset

Uploaded files go through TractViewer.add: same loading as the CLI, including
the projection of tractograms onto the affine of the first NIfTI (reprojected
layers are returned in `updated`).
"""
from __future__ import annotations
import json
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from tractviewer import loaders
from tractviewer.debug import warn_debug
from tractviewer.web import vtkjs

CHUNK = 1 << 20


class WebApp:
    """Shared server state: viewer, lock, upload folder, export options."""

    def __init__(self, viewer, title: str = "TractViewer", max_streamlines: Optional[int] = None,
                 max_voxels: int = vtkjs.DEFAULT_MAX_VOXELS, vtkjs_url: str = vtkjs.VTKJS_URL):
        self.viewer = viewer
        self.title = title
        self.max_streamlines = max_streamlines
        self.max_voxels = max_voxels
        self.vtkjs_url = vtkjs_url
        self.lock = threading.Lock()
        self.upload_dir = Path(tempfile.mkdtemp(prefix="tractviewer_upload_"))
        vtkjs.make_names_unique(viewer)

    def page(self) -> bytes:
        with self.lock:
            scene = vtkjs.build_scene(self.viewer, self.max_streamlines, self.max_voxels, server=True)
        return vtkjs.render_html(scene, title=self.title, vtkjs_url=self.vtkjs_url).encode("utf-8")

    def add_file(self, path: Path) -> dict:
        """Load an uploaded file. NIfTI files are loaded as volumes (slices); the page
        computes isosurfaces on demand."""
        is_nifti = path.name.lower().endswith((".nii", ".nii.gz"))
        with self.lock:
            vis = self.viewer
            name = vtkjs.unique_name(vis, loaders.stem(path.name))
            params = {"name": name}
            if is_nifti:
                params["marching_cubes"] = False
            untransformed = {n for n in vis.names() if "tract_transformed" not in vis.get_dataset(n).field_data}
            vis.add(str(path), params)

            ds, prm = vis._find(name)
            layer, volume = vtkjs.serialize_dataset(vis, ds, prm, self.max_streamlines, self.max_voxels)
            cmaps = {}
            updated = []
            for other in untransformed - {name}:  # tracts reprojected onto the NIfTI affine
                if "tract_transformed" in vis.get_dataset(other).field_data:
                    other_layer, _ = vtkjs.serialize_dataset(vis, *vis._find(other), self.max_streamlines)
                    if other_layer is not None:
                        cmaps.update(other_layer.pop("_cmap_def", {}))
                        updated.append(other_layer)
            if layer is not None:
                cmaps.update(layer.pop("_cmap_def", {}))
        return {
            "layers": [layer] if layer is not None else [],
            "volumes": [volume] if volume is not None else [],
            "updated": updated,
            "colormaps": vtkjs.sample_colormaps(cmaps),
        }

    def remove(self, name: str) -> None:
        with self.lock:
            self.viewer.remove(name)

    def cleanup(self):
        shutil.rmtree(self.upload_dir, ignore_errors=True)


def make_server(app: WebApp, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self._send(200, app.page(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/upload":
                self.send_error(404)
                return
            filename = Path(unquote(self.headers.get("X-Filename", ""))).name
            if not filename or filename.startswith("."):
                self._json(400, {"error": "Missing or invalid file name."})
                return
            target = app.upload_dir / uuid.uuid4().hex / filename
            target.parent.mkdir(parents=True)
            self._save_body(target)
            try:
                result = app.add_file(target)
            except Exception as exc:
                warn_debug(f"Upload of '{filename}' failed.", exc=sys.exc_info())
                self._json(400, {"error": f"{filename}: {exc}"})
                return
            self._json(200, result)

        def do_DELETE(self):
            prefix = "/api/datasets/"
            if not self.path.startswith(prefix):
                self.send_error(404)
                return
            app.remove(unquote(self.path[len(prefix):]))
            self._json(200, {"ok": True})

        def _save_body(self, target: Path):
            remaining = int(self.headers.get("Content-Length", 0))
            with open(target, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                       "application/json")

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(viewer, host: str = "127.0.0.1", port: int = 8080, open_browser: bool = False,
          title: str = "TractViewer", max_streamlines: Optional[int] = None,
          max_voxels: int = vtkjs.DEFAULT_MAX_VOXELS) -> None:
    """Serve the interface on http://host:port/ until Ctrl+C."""
    app = WebApp(viewer, title=title, max_streamlines=max_streamlines, max_voxels=max_voxels)
    httpd = make_server(app, host, port)
    url = f"http://{host}:{httpd.server_port}/"
    print(f"TractViewer web: {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.cleanup()
