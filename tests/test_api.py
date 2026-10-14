import base64
import json
import re
import threading
import urllib.request

import numpy as np
import pytest
import pyvista as pv

from tractviewer import TractViewer, TractViewerDeprecationWarning
from tractviewer import cli, params
from tractviewer.web import server, vtkjs


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("TRACTVIEWER_DEBUG", "0")


def _bundle(n_lines=20, n_pts=10):
    """Small synthetic bundle: parallel lines + point and cell arrays."""
    pts, lines = [], []
    for i in range(n_lines):
        t = np.linspace(0, 50, n_pts)
        pts.append(np.c_[t, np.full(n_pts, i * 0.5), np.sin(t / 10) * 5])
        lines += [n_pts, *range(i * n_pts, (i + 1) * n_pts)]
    poly = pv.PolyData(np.vstack(pts).astype(np.float32), lines=np.array(lines))
    poly.point_data["point_index"] = np.tile(np.arange(n_pts), n_lines).astype(float)
    poly.cell_data["streamline_id"] = np.arange(n_lines)
    return poly


@pytest.fixture
def bundle_file(tmp_path):
    path = tmp_path / "bundle.vtk"
    _bundle().save(path)
    return path


# ---------- params ----------
def test_parse_spec_types():
    path, prm = params.parse_spec(
        "b.vtk:display_array=FA,clim=(0,1),threshold=(FA,0.2,0.8),opacity=0.5,visible=false,contour,cmap=gray"
    )
    assert path == "b.vtk"
    assert prm == {
        "display_array": "FA", "clim": (0.0, 1.0), "threshold": ("FA", (0.2, 0.8)),
        "opacity": 0.5, "visible": False, "contour": True, "cmap": "gray",
    }
    assert params.parse_spec("b.vtk:clim=1:2,threshold=FA;0;1")[1] == {
        "clim": (1.0, 2.0), "threshold": ("FA", (0.0, 1.0)),
    }
    assert params.parse_spec("plain.trk") == ("plain.trk", {})


def test_normalize_alias_and_unknown_key():
    with pytest.warns(UserWarning, match="opacity"):
        out = params.normalize({"show_scalar_bar": False, "opactiy": 0.3})
    assert out["scalar_bar"] is False and "show_scalar_bar" not in out


def test_envelope_french_spelling_is_deprecated_alias():
    with pytest.warns(TractViewerDeprecationWarning):
        out = params.normalize({"style": "enveloppe", "enveloppe_smooth": 5})
    assert out == {"style": "envelope", "envelope_smooth": 5}
    from tractviewer import enveloppe
    with pytest.warns(TractViewerDeprecationWarning):
        assert enveloppe.enveloppe_minimale(pv.Sphere()).n_points > 0


# ---------- API ----------
def test_new_api_roundtrip(tmp_path):
    vis = TractViewer(off_screen=True)
    vis.add(_bundle(), name="b", display_array="point_index").add(pv.Sphere(), name="s")
    assert vis.names() == ["b", "s"]
    vis.edit("s", color="red", contour=False)
    assert vis.get_params("s")["color"] == "red"
    img = vis.screenshot(tmp_path / "a.png")
    assert (tmp_path / "a.png").exists() and img.ndim == 3
    vis.remove("s")
    assert vis.names() == ["b"]
    with pytest.raises(ValueError):
        vis.edit("absent", color="red")


def test_record_gif(tmp_path):
    vis = TractViewer(off_screen=True).add(_bundle())
    out = vis.record(tmp_path / "rot.gif", n_frames=3)
    assert out.endswith(".gif") and (tmp_path / "rot.gif").stat().st_size > 0


def test_legacy_methods_still_work_with_warning(tmp_path):
    vis = TractViewer(off_screen=True).add(_bundle(), {"name": "b"}).add(pv.Sphere(), {"name": "s"})
    with pytest.warns(TractViewerDeprecationWarning, match="capture_screenshot"):
        vis.capture_screenshot(tmp_path / "old.png", rotation_x=0)
    assert (tmp_path / "old.png").exists()
    with pytest.warns(TractViewerDeprecationWarning, match="record_rotation"):
        out = vis.record_rotation(tmp_path / "old_rot", n_frames=2, gif=True)
    assert out.endswith(".gif")
    with pytest.warns(TractViewerDeprecationWarning, match=r"\.rm"):
        vis.rm("s")
    assert vis.names() == ["b"]
    with pytest.warns(TractViewerDeprecationWarning):
        assert TractViewer._choose_font_color("black") == "white"
    with pytest.warns(TractViewerDeprecationWarning):
        assert TractViewer._load(_bundle()).n_lines == 20


def test_mesh_rotation_is_restored():
    vis = TractViewer(off_screen=True).add(_bundle(), name="b")
    before = vis.get_dataset("b").points.copy()
    vis.screenshot(mesh_rotation=(-90, 30, 0))
    np.testing.assert_allclose(vis.get_dataset("b").points, before)


@pytest.fixture
def nifti_file(tmp_path):
    nib = pytest.importorskip("nibabel")
    grid = np.mgrid[:20, :20, :20]
    vol = (np.linalg.norm(grid - 10, axis=0) < 6).astype(np.float32)
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([2.0, 2.0, 2.0, 1.0])), path)
    return path


def test_nifti_iso_and_coord_system(nifti_file):
    path = nifti_file

    ras = TractViewer(off_screen=True, coord_system="RAS").add(path, name="m", smooth=0).get_dataset("m")
    lps = TractViewer(off_screen=True).add(path, name="m", smooth=0, iso=0.5).get_dataset("m")
    assert ras.field_data["coord_system"][0] == "RAS"
    assert lps.field_data["coord_system"][0] == "LPS"
    assert float(lps.field_data["nifti_iso_value"][0]) == pytest.approx(0.5)
    np.testing.assert_allclose(ras.center[0], -lps.center[0], atol=1.0)


# ---------- export vtk.js ----------
def test_build_scene_contains_all_arrays():
    vis = TractViewer(off_screen=True)
    vis.add(_bundle(), name="b", display_array="point_index", cmap=["red", "blue"], clim=(0, 5))
    vis.add(pv.Sphere(), name="s", color="tomato", visible=False)
    scene = vtkjs.build_scene(vis)
    b, s = scene["layers"]
    assert {(a["location"], a["name"]) for a in b["arrays"]} == {
        ("point", "point_index"), ("cell", "streamline_id"),
    }
    assert b["activeArray"] == "point:point_index" and b["clim"] == [0.0, 5.0]
    assert b["cmap"] in scene["colormaps"] and b["lines"] and not b["polys"]
    assert s["polys"] and s["visible"] is False and s["color"] == "#ff6347"


def test_export_html_subsample(tmp_path):
    vis = TractViewer(off_screen=True).add(_bundle(n_lines=50), name="b")
    out = vis.export_html(tmp_path / "scene", max_streamlines=10)
    html = open(out).read()
    assert out.endswith(".html") and vtkjs.VTKJS_URL in html and vtkjs.GIFENC_URL in html
    assert "__" + "GIFENC_URL__" not in html
    scene = json.loads(re.search(r'id="scene" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert scene["layers"][0]["nCells"] == 10


# ---------- CLI ----------
def test_cli_subcommands(bundle_file, tmp_path, capsys):
    out = tmp_path / "shot.png"
    cli.main(["screenshot", f"{bundle_file}:display_array=point_index", "-o", str(out)])
    assert out.exists()
    cli.main(["info", str(bundle_file)])
    assert "streamline_id" in capsys.readouterr().out
    cli.main(["html", str(bundle_file), "-o", str(tmp_path / "s.html")])
    assert (tmp_path / "s.html").exists()


def test_cli_legacy_syntax_warns(bundle_file, tmp_path, capsys):
    out = tmp_path / "legacy.png"
    cli.main([str(bundle_file), "--screenshot", str(out), "--rotation-x", "0"])
    assert out.exists()
    assert "Deprecated" in capsys.readouterr().err


def test_cli_legacy_parse_input_spec_alias():
    with pytest.warns(TractViewerDeprecationWarning):
        assert cli._parse_input_spec("a.vtk:opacity=0.2") == ("a.vtk", {"opacity": 0.2})


# ---------- volumes / slices ----------
def test_nifti_volume_matches_isosurface(nifti_file):
    vis = TractViewer(off_screen=True).add(nifti_file, name="m", smooth=0, iso=0.5)
    layer, volume = vtkjs.serialize_dataset(vis, *vis._find("m"))
    assert layer is not None and volume["id"] == "m" and volume["dims"] == [20, 20, 20]
    data = np.frombuffer(base64.b64decode(volume["data"]), np.float32).reshape(volume["dims"], order="F")
    inside = np.argwhere(data > 0.5)
    lo = np.array(volume["origin"]) + inside.min(0) * volume["spacing"]
    hi = np.array(volume["origin"]) + inside.max(0) * volume["spacing"]
    surf = np.array(vis.get_dataset("m").bounds).reshape(3, 2)
    np.testing.assert_allclose(lo, surf[:, 0], atol=1.5)  # interpolated isosurface: < 1 voxel
    np.testing.assert_allclose(hi, surf[:, 1], atol=1.5)


def test_nifti_without_marching_cubes_is_volume_only(nifti_file):
    vis = TractViewer(off_screen=True).add(nifti_file, name="m", marching_cubes=False)
    layer, volume = vtkjs.serialize_dataset(vis, *vis._find("m"))
    assert layer is None and volume is not None


# ---------- web server ----------
def _request(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def test_server_upload_and_delete(bundle_file, nifti_file):
    vis = TractViewer(off_screen=True).add(str(bundle_file), name="bundle")
    app = server.WebApp(vis)
    httpd = server.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        page = _request(base + "/").decode()
        assert '"server":true' in page

        result = json.loads(_request(base + "/api/upload", "POST", nifti_file.read_bytes(),
                                     {"X-Filename": "mask.nii.gz"}))
        assert result["layers"] == [] and result["volumes"][0]["id"] == "mask"

        result = json.loads(_request(base + "/api/upload", "POST", bundle_file.read_bytes(),
                                     {"X-Filename": "bundle.vtk"}))
        assert result["layers"][0]["id"] == "bundle_2"  # unique name
        assert vis.names() == ["bundle", "mask", "bundle_2"]

        _request(base + "/api/datasets/bundle_2", "DELETE")
        assert vis.names() == ["bundle", "mask"]

        with pytest.raises(urllib.error.HTTPError):
            _request(base + "/api/upload", "POST", b"x", {"X-Filename": "../.."})
    finally:
        httpd.shutdown()
        app.cleanup()


def test_server_upload_nifti_reprojects_existing_tract(tmp_path, nifti_file):
    nib = pytest.importorskip("nibabel")
    tck = tmp_path / "tract.tck"
    lines = [np.c_[np.linspace(5, 30, 8), np.full(8, 10.0), np.full(8, 12.0)].astype(np.float32)]
    nib.streamlines.save(nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)), str(tck))
    vis = TractViewer(off_screen=True).add(str(tck), name="tract")
    x_before = vis.get_dataset("tract").points[:, 0].copy()

    app = server.WebApp(vis)
    try:
        result = app.add_file(nifti_file)
    finally:
        app.cleanup()
    assert [layer["id"] for layer in result["updated"]] == ["tract"]
    np.testing.assert_allclose(vis.get_dataset("tract").points[:, 0], -x_before)  # RAS -> LPS


def test_cli_html_loads_nifti_as_slices(tmp_path, nifti_file):
    out = tmp_path / "s.html"
    cli.main(["html", str(nifti_file), "-o", str(out)])
    html = out.read_text(encoding="utf-8")
    scene = json.loads(re.search(r'id="scene" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert scene["layers"] == [] and [v["id"] for v in scene["volumes"]] == ["mask"]
    assert 'id="iso-generate"' in html and 'id="loading"' in html

    cli.main(["html", f"{nifti_file}:marching_cubes=true,smooth=0", "-o", str(out)])
    scene = json.loads(re.search(r'id="scene" type="application/json">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1))
    assert [layer["id"] for layer in scene["layers"]] == ["mask"]


def test_cli_web_accepts_no_inputs():
    args = cli.build_parser().parse_args(["web", "--port", "9000"])
    assert args.inputs == [] and args.port == 9000
