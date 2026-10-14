"""Command line interface.

    tractviewer web        [FILES...]            web interface (vtk.js): upload, slices, GIF
    tractviewer html       FILES... -o x.html    standalone vtk.js page
    tractviewer show       FILES...              desktop window (pyvista)
    tractviewer screenshot FILES... -o x.png     PNG capture
    tractviewer record     FILES... -o x.mp4     rotation video / GIF
    tractviewer info       FILES...              arrays and sizes of the datasets

Each FILE accepts parameters: ``path:key=val,key=val`` (see tractviewer.params).
The former syntax without a subcommand (``tractviewer f.vtk --screenshot x.png``)
is still accepted but deprecated.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tractviewer._deprecation import deprecated
from tractviewer.core import TractViewer
from tractviewer.loaders import stem
from tractviewer.params import parse_spec

COMMANDS = ("web", "html", "show", "screenshot", "record", "info")

INPUT_HELP = (
    "Files to load, with optional parameters: path:key=val,key=val. "
    "E.g. anat.nii.gz:cmap=gray,opacity=0.3 bundle.trk:display_array=length_mm,clim=(50,150)"
)


# ------------------------------
# Parser (subcommands)
# ------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tractviewer",
        description="Tractography / surface / NIfTI viewer.",
        epilog="Former syntax (deprecated): tractviewer FILES [--screenshot ...] [--rotate N] [--web]",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    options = argparse.ArgumentParser(add_help=False)
    g = options.add_argument_group("scene")
    g.add_argument("--background", default="white", help="Background color (default: white)")
    g.add_argument("--window-size", type=_window_size, metavar="WxH", help="Window size (e.g. 1280x720)")
    g.add_argument("--color", help="Default color of datasets without 'color'")
    g.add_argument("--ras-axes", action="store_true", help="Show RAS axes")
    g = options.add_argument_group("NIfTI")
    g.add_argument("--no-marching-cubes", action="store_true",
                   help="Load the raw volume (no isosurface). Always the case for web/html, "
                        "unless a file sets marching_cubes=true")
    g.add_argument("--smooth", type=int, help="Smoothing iterations after marching cubes (default: 200, 0=none)")
    g.add_argument("--iso", type=float, help="Iso value (default: TRACTVIEWER_NIFTI_ISO or 0.2)")
    g.add_argument("--coord", choices=("LPS", "RAS"), help="Target frame (default: TRACTVIEWER_NIFTI_COORD or LPS)")

    common = argparse.ArgumentParser(add_help=False, parents=[options])
    common.add_argument("inputs", nargs="+", help=INPUT_HELP)

    rotation = argparse.ArgumentParser(add_help=False)
    rotation.add_argument("--mesh-rotation", type=float, nargs=3, metavar=("X", "Y", "Z"),
                          default=(-90.0, 0.0, 0.0),
                          help="Mesh rotation (degrees) before rendering (default: -90 0 0)")

    sp = sub.add_parser("web", parents=[options], help="Web interface (vtk.js): upload, slices, GIF")
    sp.add_argument("inputs", nargs="*", help=INPUT_HELP + " Optional: files can be uploaded from the page.")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--open", action="store_true", help="Open the browser")
    sp.add_argument("--title", default="TractViewer")
    sp.add_argument("--max-streamlines", type=int, help="Subsample large tractograms")
    sp.add_argument("--max-voxels", type=int, help="Decimate large volumes (default: 16e6)")

    sp = sub.add_parser("html", parents=[common], help="Standalone vtk.js HTML page")
    sp.add_argument("-o", "--output", type=Path, required=True, help=".html file")
    sp.add_argument("--title", default="TractViewer")
    sp.add_argument("--max-streamlines", type=int, help="Subsample large tractograms")
    sp.add_argument("--max-voxels", type=int, help="Decimate large volumes (default: 16e6)")

    sub.add_parser("show", parents=[common], help="Desktop window (pyvista)")

    sp = sub.add_parser("screenshot", parents=[common, rotation], help="PNG capture")
    sp.add_argument("-o", "--output", type=Path, required=True, help="PNG file")
    sp.add_argument("--transparent", action="store_true", help="Transparent background")

    sp = sub.add_parser("record", parents=[common, rotation], help="Rotation video / GIF")
    sp.add_argument("-o", "--output", type=Path, required=True, help=".mp4 / .gif file (format from the extension)")
    sp.add_argument("--frames", type=int, default=180, help="Number of frames (default: 180)")
    sp.add_argument("--step", type=float, help="Azimuth step in degrees (default: 360/frames)")
    sp.add_argument("--elevation", type=float, default=0.0, help="Elevation oscillation per frame (degrees)")
    sp.add_argument("--fps", type=int, default=30)

    sub.add_parser("info", parents=[common], help="List the arrays of each dataset")
    return p


def _window_size(value: str) -> Tuple[int, int]:
    try:
        w, h = map(int, value.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("expected format: WxH (e.g. 1280x720)")


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        return legacy_main(argv)

    args = build_parser().parse_args(argv)
    cmd = args.command

    vis = TractViewer(background=args.background, off_screen=(cmd != "show"),
                      window_size=args.window_size, show_ras_axes=args.ras_axes,
                      coord_system=args.coord, iso_value=args.iso)
    # Web pages show NIfTI files as slices; the isosurface is generated on demand in the page
    # (or in Python with the per-file parameter marching_cubes=true).
    marching_cubes = not args.no_marching_cubes and cmd not in ("web", "html")
    _add_inputs(vis, args.inputs, color=args.color, marching_cubes=marching_cubes, smooth=args.smooth)

    if cmd == "web":
        vis.serve(host=args.host, port=args.port, open_browser=args.open, title=args.title,
                  max_streamlines=args.max_streamlines, max_voxels=args.max_voxels)
    elif cmd == "html":
        out = vis.export_html(args.output, title=args.title, max_streamlines=args.max_streamlines,
                              max_voxels=args.max_voxels)
        print(f"vtk.js page: {out}")
    elif cmd == "show":
        vis.show()
    elif cmd == "screenshot":
        vis.screenshot(args.output, transparent=args.transparent, mesh_rotation=tuple(args.mesh_rotation))
        print(f"Screenshot: {args.output}")
    elif cmd == "record":
        out = vis.record(args.output, n_frames=args.frames, step=args.step, elevation=args.elevation,
                         fps=args.fps, mesh_rotation=tuple(args.mesh_rotation))
        print(f"Rotation: {out}")
    elif cmd == "info":
        print(describe(vis))


def _add_inputs(vis, inputs, color=None, marching_cubes=True, smooth=None):
    for raw in inputs:
        path_str, meta = parse_spec(raw)
        meta.setdefault("name", stem(path_str))
        if not marching_cubes:
            meta.setdefault("marching_cubes", False)
        if smooth is not None:
            meta.setdefault("smooth", smooth)
        if color:
            meta.setdefault("color", color)
        vis.add(path_str, meta)


def describe(vis: TractViewer) -> str:
    """Text summary of the datasets: type, sizes, arrays (location, dtype, range)."""
    lines = []
    for name in vis.names():
        ds = vis.get_dataset(name)
        sizes = f"{ds.n_points} points, {ds.n_cells} cells"
        if getattr(ds, "n_lines", 0):
            sizes += f" ({ds.n_lines} lines)"
        lines.append(f"{name}  [{type(ds).__name__}]  {sizes}")
        for location, store in (("point", ds.point_data), ("cell", ds.cell_data)):
            for arr_name in store.keys():
                values = np.asarray(store[arr_name])
                comps = "" if values.ndim == 1 else f"x{values.shape[1]}"
                if values.dtype.kind in "biuf" and values.size:
                    rng = f"[{np.nanmin(values):.4g}, {np.nanmax(values):.4g}]"
                else:
                    rng = ""
                lines.append(f"  {location:<5} {arr_name:<32} {str(values.dtype) + comps:<10} {rng}")
    return "\n".join(lines)


# ------------------------------
# Former syntax (deprecated)
# ------------------------------
def build_legacy_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tractviewer", description="(deprecated syntax, see 'tractviewer -h')")
    p.add_argument("inputs", nargs="+", help=INPUT_HELP)
    p.add_argument("--background", default="white", help="Background color (default: white)")
    p.add_argument("--off-screen", action="store_true", help="Force off-screen rendering")
    p.add_argument("--interactive", action="store_true", help="Force the interactive window even with capture/rotation")
    p.add_argument("--web", action="store_true", help="Start the web interface (vtk.js)")
    p.add_argument("--web-host", type=str, default="127.0.0.1", help="Web server host")
    p.add_argument("--web-port", type=int, default=8080, help="Web server port")
    p.add_argument("--open-browser", action="store_true", help="Open the browser on the web interface")
    p.add_argument("--screenshot", type=Path, help="PNG capture path after loading")
    p.add_argument("--rotate", type=int, default=0, help="Number of rotation frames to record (0=no rotation)")
    p.add_argument("--rotation-output", type=Path, help="Rotation video or GIF file (e.g. rotation.mp4)")
    p.add_argument("--gif", action="store_true", help="Force GIF for the rotation")
    p.add_argument("--step", type=float, default=2.0, help="Azimuth step (deg) between rotation frames")
    p.add_argument("--window-size", type=str, help="Window size WxH (e.g. 1280x720)")
    p.add_argument("--rotation-x", type=float, default=-90.0, help="Mesh rotation around X in degrees (default: -90)")
    p.add_argument("--rotation-y", type=float, default=0.0, help="Mesh rotation around Y in degrees (default: 0)")
    p.add_argument("--rotation-z", type=float, default=0.0, help="Mesh rotation around Z in degrees (default: 0)")
    p.add_argument("--no-marching-cubes", action="store_true", help="Disable marching cubes for NIfTI (load the raw volume)")
    p.add_argument("--smooth", type=int, default=0, help="Laplacian smoothing iterations after marching cubes (0=default)")
    p.add_argument("--color", type=str, default=None, help="NIfTI surface color (e.g. white, #ff8800, red)")
    p.add_argument("--ras-axes", action="store_true", help="Show RAS axes in the scene")
    return p


def legacy_main(argv: List[str]):
    parser = build_legacy_parser()
    args = parser.parse_args(argv)
    print("Deprecated: the 'tractviewer FILES --option' syntax. Use a subcommand: "
          "tractviewer {web,html,show,screenshot,record,info} FILES ... (see 'tractviewer -h').",
          file=sys.stderr)

    if args.off_screen and args.interactive:
        parser.error("--off-screen and --interactive are mutually exclusive")
    if args.web and args.interactive:
        parser.error("--web and --interactive are mutually exclusive")

    off_screen = args.off_screen or args.web or (not args.interactive and bool(args.screenshot or args.rotate))
    print("Off-screen mode" if off_screen else "Interactive mode")
    window_size = None
    if args.window_size:
        try:
            window_size = _window_size(args.window_size)
        except argparse.ArgumentTypeError:
            parser.error("Expected --window-size format: WxH (e.g. 1280x720)")

    vis = TractViewer(background=args.background, off_screen=off_screen,
                      window_size=window_size, show_ras_axes=args.ras_axes)

    # --smooth 0 used to mean "not specified" (=> add() default).
    _add_inputs(vis, args.inputs, color=args.color, marching_cubes=not args.no_marching_cubes,
                smooth=args.smooth or None)

    mesh_rotation = (args.rotation_x, args.rotation_y, args.rotation_z)
    if args.screenshot:
        vis.screenshot(args.screenshot, mesh_rotation=mesh_rotation)

    if args.rotate and args.rotate > 0:
        output = args.rotation_output or Path("rotation.mp4")
        if not output.suffix:
            output = output.with_suffix(".gif" if args.gif else ".mp4")
        vis.record(output, n_frames=args.rotate, step=args.step, mesh_rotation=mesh_rotation)

    if args.web:
        vis.serve(host=args.web_host, port=args.web_port, open_browser=args.open_browser)
    elif args.interactive or (not off_screen and not (args.screenshot or args.rotate)):
        vis.show()


@deprecated("tractviewer.params.parse_spec", old="tractviewer.cli._parse_input_spec")
def _parse_input_spec(spec: str):
    return parse_spec(spec)


if __name__ == "__main__":
    main()
