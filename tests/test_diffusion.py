"""Diffusion models (DTI, ODF, Anima MCM): readers, conventions and glyph payload."""
import base64
import gzip
import io
import zipfile

import nibabel as nib
import numpy as np
import pytest

from tractviewer import TractViewer
from tractviewer.diffusion import io as dio
from tractviewer.diffusion import sh
from tractviewer.diffusion.glyphs import glyph_payload
from tractviewer.web import vtkjs

# Oblique voxel -> RAS affine (20° around z, 2 mm voxels, x flipped like a radiological image).
ANGLE = np.deg2rad(20)
ROT = np.array([[np.cos(ANGLE), -np.sin(ANGLE), 0], [np.sin(ANGLE), np.cos(ANGLE), 0], [0, 0, 1]])
AFFINE = np.eye(4)
AFFINE[:3, :3] = ROT @ np.diag([-2.0, 2.0, 2.0])
AFFINE[:3, 3] = [40, -30, -20]


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("TRACTVIEWER_DEBUG", "0")


def _stick(direction, l1=1.7e-3, l2=0.3e-3):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return l2 * np.eye(3) + (l1 - l2) * np.outer(d, d)


def _anima6(m):
    return np.array([m[0, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2], m[2, 2]])


def _main_axis(payload, entry):
    m = payload["matrix"][entry].reshape(3, 3)
    w, v = np.linalg.eigh(m)
    return v[:, 2]


# ---------- spherical harmonics ----------
def test_sh_basis_matches_dipy_mrtrix_basis():
    shm = pytest.importorskip("dipy.reconst.shm")
    verts, faces = sh.icosphere(2)
    assert verts.shape == (162, 3) and faces.shape == (320, 3)
    theta, phi = np.arccos(verts[:, 2]), np.arctan2(verts[:, 1], verts[:, 0])
    ref, _, _ = shm.real_sh_tournier(8, theta, phi, legacy=False)
    assert np.allclose(sh.sh_basis(8, verts), ref, atol=1e-10)
    assert sh.lmax_from_ncoeffs(45) == 8
    with pytest.raises(ValueError):
        sh.lmax_from_ncoeffs(44)


# ---------- NRRD / MCM ----------
def _nrrd(data, kinds, directions="(2,0,0) (0,2,0) (0,0,2)", origin="(-10,-20,30)"):
    """Gzip NRRD bytes of `data` (shape = NRRD sizes, fastest axis first)."""
    vec = ["none"] * (len(kinds) - 3)
    header = "\n".join([
        "NRRD0004", "type: double", f"dimension: {data.ndim}", "space: left-posterior-superior",
        f"sizes: {' '.join(map(str, data.shape))}", f"space directions: {' '.join(vec)} {directions}".strip(),
        f"kinds: {' '.join(kinds)}", "endian: little", "encoding: gzip", f"space origin: {origin}",
    ])
    body = gzip.compress(np.asarray(data, "<f8").tobytes(order="F"))
    return header.encode() + b"\n\n" + body


def test_read_nrrd_vector_image():
    data = np.random.default_rng(0).random((6, 3, 4, 5))
    nrrd = dio.read_nrrd(_nrrd(data, ["vector", "domain", "domain", "domain"]))
    assert np.allclose(nrrd.data, data)
    assert nrrd.voxels_last().shape == (3, 4, 5, 6)
    # LPS directions/origin -> RAS affine
    assert np.allclose(nrrd.affine_ras(), [[-2, 0, 0, 10], [0, -2, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1]])


def _mcmx(tmp_path, shape=(4, 3, 2)):
    """MCM with a free water compartment and two crossing tensors (voxel x and voxel y)."""
    t_x = np.broadcast_to(_anima6(_stick([1, 0, 0])), shape + (6,))
    t_y = np.broadcast_to(_anima6(_stick([0, 1, 0])), shape + (6,))
    weights = np.broadcast_to([0.2, 0.5, 0.3], shape + (3,))
    vec = ["vector", "domain", "domain", "domain"]
    xml = ("<?xml version=\"1.0\"?><Model><Weights>mcm_weights.nrrd</Weights>"
           "<Compartment><Type>FreeWater</Type><FileName>mcm_0.nrrd</FileName></Compartment>"
           "<Compartment><Type>Tensor</Type><FileName>mcm_1.nrrd</FileName></Compartment>"
           "<Compartment><Type>Tensor</Type><FileName>mcm_2.nrrd</FileName></Compartment></Model>")
    path = tmp_path / "model.mcmx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mcm.mcm", xml)
        z.writestr("mcm/mcm_weights.nrrd", _nrrd(np.moveaxis(weights, -1, 0), vec))
        z.writestr("mcm/mcm_0.nrrd", _nrrd(np.full(shape, 3e-3), ["domain"] * 3))
        z.writestr("mcm/mcm_1.nrrd", _nrrd(np.moveaxis(t_x, -1, 0), vec))
        z.writestr("mcm/mcm_2.nrrd", _nrrd(np.moveaxis(t_y, -1, 0), vec))
    return path


def test_mcm_glyphs(tmp_path):
    vis = TractViewer(off_screen=True, coord_system="LPS")
    vis.add(str(_mcmx(tmp_path)), name="mcm")
    ds = vis.get_dataset("mcm")
    assert str(ds.field_data["diffusion_model"][0]) == "mcm"
    assert np.allclose(ds.point_data["intensity"], 0.8)          # anisotropic fraction
    assert np.allclose(ds.point_data["weight_FreeWater"], 0.2)
    p = glyph_payload(ds, "LPS")
    assert p["kind"] == "ellipsoid" and p["count"] == 2 * 24
    assert sorted(set(p["compartment"].tolist())) == [0, 1]
    # NRRD voxel axes = LPS axes here: voxel x -> display (LPS) x, voxel y -> display y.
    first = np.flatnonzero(p["compartment"] == 0)[0]
    second = np.flatnonzero(p["compartment"] == 1)[0]
    assert abs(_main_axis(p, first)[0]) > 0.999 and abs(_main_axis(p, second)[1]) > 0.999
    assert np.allclose(p["amp"][p["compartment"] == 0], 0.5)
    # signed principal direction (projection on tracts): main axis of the ellipsoid
    assert p["direction"].shape == (48, 3)
    assert abs(p["direction"][first] @ _main_axis(p, first)) > 0.999


def test_mcm_scalar_maps(tmp_path):
    vis = TractViewer(off_screen=True, coord_system="LPS")
    vis.add(str(_mcmx(tmp_path)), name="mcm", display_array="weight_FreeWater")
    volume = vtkjs.diffusion_volume(vis.get_dataset("mcm"), "LPS", display_array="weight_FreeWater")
    maps = volume["maps"]
    assert [m["key"] for m in maps] == ["intensity", "weight_FreeWater", "weight_0", "weight_1"]
    assert [m["label"] for m in maps] == ["Anisotropic fraction", "FreeWater weight", "Tensor 1 weight", "Tensor 2 weight"]
    assert "data" not in maps[0] and volume["map"] == "weight_FreeWater"
    free_water = np.frombuffer(base64.b64decode(maps[1]["data"]), np.float32)
    assert free_water.size == 24 and np.allclose(free_water, 0.2)
    # pyvista (desktop) rendering of the chosen map
    from tractviewer.scene import world_volume
    assert np.allclose(world_volume(vis.get_dataset("mcm"), "weight_FreeWater").point_data["intensity"], 0.2)
    vis.screenshot(tmp_path / "mcm.png")  # scene._add_volume with display_array


# ---------- DTI ----------
def test_anima_dti_is_in_voxel_frame(tmp_path):
    """Anima tensors are expressed in the voxel frame: a voxel-x stick points along the affine's x column."""
    data = np.zeros((5, 5, 3, 1, 6), np.float32)
    data[..., 0, :] = _anima6(_stick([1, 0, 0]))
    path = tmp_path / "dti.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), path)

    vis = TractViewer(off_screen=True, coord_system="RAS")
    vis.add(str(path), name="dti")
    ds = vis.get_dataset("dti")
    assert str(ds.field_data["diffusion_model"][0]) == "dti"
    fa = ds.point_data["intensity"]
    assert np.allclose(fa, fa[0]) and 0.6 < fa[0] < 0.8
    p = glyph_payload(ds, "RAS")
    expected = AFFINE[:3, 0] / np.linalg.norm(AFFINE[:3, 0])
    assert abs(_main_axis(p, 0) @ expected) > 0.999
    assert np.allclose(p["color"][0], np.rint(np.abs(expected) * 255), atol=1)
    # LPS display: x and y flipped, same axis up to the sign
    p_lps = glyph_payload(ds, "LPS")
    assert abs(_main_axis(p_lps, 0) @ (np.diag([-1, -1, 1]) @ expected)) > 0.999


def test_mrtrix_dti_world_frame_and_plain_4d(tmp_path):
    m = _stick([0, 0, 1])
    data = np.zeros((4, 4, 4, 6), np.float32)
    data[...] = [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]
    nib.save(nib.Nifti1Image(data, AFFINE), tmp_path / "tensor.nii.gz")
    dwi = np.random.default_rng(0).uniform(100, 1000, (4, 4, 4, 6)).astype(np.float32)
    nib.save(nib.Nifti1Image(dwi, AFFINE), tmp_path / "dwi.nii.gz")

    vis = TractViewer(off_screen=True, coord_system="RAS")
    vis.add(str(tmp_path / "tensor.nii.gz"), name="t", marching_cubes=False)
    vis.add(str(tmp_path / "dwi.nii.gz"), name="dwi", marching_cubes=False)
    p = glyph_payload(vis.get_dataset("t"), "RAS")
    assert abs(_main_axis(p, 0)[2]) > 0.999                      # world z, whatever the affine
    assert "diffusion_model" not in vis.get_dataset("dwi").field_data
    with pytest.raises(ValueError, match="Unknown diffusion model"):
        vis.add(str(tmp_path / "tensor.nii.gz"), model="csd")


# ---------- ODF ----------
def test_mrtrix_odf_detection_and_payload(tmp_path):
    verts, _ = sh.icosphere(3)
    basis = sh.sh_basis(8, verts)
    sharp = np.exp(-((verts @ [0, 1, 0]) ** 2 - 1) ** 2 / 0.02)  # lobes along world y
    coeffs = np.linalg.lstsq(basis, sharp, rcond=None)[0]
    data = np.zeros((4, 4, 4, 45), np.float32)
    data[1:3, 1:3, 1:3] = coeffs
    nib.save(nib.Nifti1Image(data, AFFINE), tmp_path / "fod.nii.gz")

    vis = TractViewer(off_screen=True, coord_system="LPS")
    vis.add(str(tmp_path / "fod.nii.gz"), name="fod")
    ds = vis.get_dataset("fod")
    assert str(ds.field_data["diffusion_model"][0]) == "odf"
    p = glyph_payload(ds, "LPS")
    assert p["kind"] == "sh" and p["nCoeffs"] == 45 and p["count"] >= 8
    assert [len(s["vertices"]) for s in p["spheres"]] == [42, 162, 642]  # coarse to fine (LOD)
    for sphere in p["spheres"]:  # basis evaluated at display-frame vertices: peak along ±y
        v = sphere["vertices"].reshape(-1, 3)
        half = sphere["half"]
        assert len(half) == len(v) and sphere["basis"].size == 45 * len(v) // 2
        radii = (sphere["basis"].reshape(-1, 45) @ p["coeffs"][0])[half]
        assert abs(v[np.argmax(radii)][1]) > 0.95
        # antipodal half: same radii as the basis evaluated at every vertex
        full = sh.sh_basis(8, v.astype(np.float64) @ np.diag([-1, -1, 1])) @ p["coeffs"][0]
        assert np.allclose(radii, full, atol=1e-4 * np.abs(full).max())


def test_scene_and_server_carry_glyphs(tmp_path):
    vis = TractViewer(off_screen=True)
    vis.add(str(_mcmx(tmp_path)), name="mcm")
    scene = vtkjs.build_scene(vis)
    (volume,) = scene["volumes"]
    assert scene["layers"] == []
    glyphs = volume["glyphs"]
    assert glyphs["model"] == "mcm" and isinstance(glyphs["matrix"], str)  # base64
    assert len(glyphs["spheres"]) == 2 and isinstance(glyphs["spheres"][0]["faces"], str)
    html = vtkjs.render_html(scene)
    assert "glyph-controls" in html
