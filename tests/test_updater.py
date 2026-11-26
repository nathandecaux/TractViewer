import io
import json

from tractviewer import updater


def test_parse_version():
    assert updater.parse_version("v1.2.3") == (1, 2, 3)
    assert updater.parse_version("1.10.0") > updater.parse_version("1.9.9")
    assert updater.parse_version("0.2.0-rc1") == (0, 2, 0)


def test_is_newer():
    assert updater.is_newer("v0.2.0", "0.1.9")
    assert not updater.is_newer("v0.1.0", "0.1.0")
    assert not updater.is_newer("v0.0.9", "0.1.0")


def test_not_frozen_never_updates(monkeypatch):
    monkeypatch.delenv("APPIMAGE", raising=False)
    assert updater.install_kind() is None
    assert updater.check() is None


def test_check_picks_platform_asset(monkeypatch):
    release = {
        "tag_name": "v99.0.0", "draft": False, "prerelease": False, "html_url": "https://x/rel",
        "assets": [
            {"name": "TractViewer-99.0.0-windows-x64-setup.exe", "browser_download_url": "https://x/w", "size": 3},
            {"name": "TractViewer-99.0.0-linux-x86_64.AppImage", "browser_download_url": "https://x/l", "size": 5},
        ],
    }
    monkeypatch.setattr(updater, "_get", lambda url, timeout=10: io.BytesIO(json.dumps(release).encode()))
    monkeypatch.setattr(updater, "_arch", lambda: "x86_64")
    up = updater.check("appimage")
    assert up.version == "99.0.0" and up.url == "https://x/l" and up.size == 5
    assert updater.check("windows-installer").url == "https://x/w"
    assert updater.check("macos-app") is None  # no asset for this platform
    release["prerelease"] = True
    assert updater.check("appimage") is None


def test_check_offline(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(updater, "_get", boom)
    assert updater.check("appimage") is None


def test_bump_version(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "bump", Path(__file__).resolve().parent.parent / "packaging" / "bump_version.py")
    bump = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bump)
    f = tmp_path / "_version.py"
    f.write_text('__version__ = "1.2.3"\n')
    monkeypatch.setattr(bump, "FILE", f)
    for arg, expected in (("patch", "1.2.4"), ("minor", "1.3.0"), ("major", "2.0.0"), ("v5.0.1", "5.0.1")):
        bump.main([arg])
        assert f.read_text() == f'__version__ = "{expected}"\n'
