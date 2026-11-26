# Remote web server and HTML export

## Web server

`tractviewer web` serves the [interface](interface.md) over HTTP. Typical use: data on a compute
server or a workstation, viewed from a laptop browser. Rendering happens in the browser (WebGL),
so the server needs no GPU and no display.

```bash
pip install git+https://github.com/nathandecaux/TractViewer.git

tractviewer web                                   # empty page, add files from the browser
tractviewer web anat.nii.gz bundle.trk            # start with some files
tractviewer web --port 9000 --title "Subject 01"
tractviewer web anat.nii.gz bundle.trk --open     # local use: also opens the browser
```

The page is served on <http://0.0.0.0:8080> by default. Files can be added from the browser
(button or drag and drop); they are uploaded to the server and loaded like on the command line.

From Python:

```python
vis = TractViewer()
vis.add("bundle.tck", display_array="length_mm")
vis.serve(host="127.0.0.1", port=8080)
```

Options: `--host` (default `0.0.0.0`), `--port` (default 8080), `--open`, `--title`,
`--max-streamlines`, `--max-voxels`, plus the [common options](cli.md#common-options).

### Security: prefer an SSH tunnel

The server has **no authentication**: anyone who reaches the port can view the data and upload
files. With the default `--host 0.0.0.0`, it is reachable from the whole network. Only do this on
a trusted network.

On a shared or exposed server, listen on localhost only and use an SSH tunnel:

```bash
# on the server
tractviewer web data/*.trk --host 127.0.0.1 --port 8080

# on your computer
ssh -N -L 8080:127.0.0.1:8080 user@server
# then open http://localhost:8080
```

### Notes

- The page loads vtk.js (and the GIF encoder, on demand) from the jsdelivr CDN: the browser needs
  internet access.
- Large tractograms are subsampled according to the server's available RAM, see
  [Large tractograms](interface.md#large-tractograms).
- Captures and GIFs are downloaded by the browser.

## Standalone HTML pages

`tractviewer html` writes the scene into a single self-contained HTML file: open it locally or
share it (email, website, report), no server needed.

```bash
tractviewer html anat.nii.gz bundle.vtk -o scene.html
tractviewer html bundle.trk:display_array=length_mm --max-streamlines 20000 -o bundle.html
```

From Python: `vis.export_html("scene.html")`. The *Export* button of the interface does the same
with the current scene (uploaded files, display settings, isosurfaces, slices and camera).

The page keeps the full interface (coloring, slices, isosurface, GIF export) except adding files.
It loads vtk.js from a CDN. All data is embedded in the page, so keep it reasonably small:
`--max-streamlines` and `--max-voxels` reduce the size; very large tractograms (above ~19 M points)
cannot be embedded.
