import sys

import pytest

from tractviewer import linux_integration as li

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")


@pytest.fixture
def env(tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    appdir = tmp_path / "mnt"
    appdir.mkdir()
    (appdir / "tractviewer.png").write_bytes(b"png")
    monkeypatch.setenv("APPDIR", str(appdir))
    download = tmp_path / "Downloads" / "TractViewer-1.0.0-linux-x86_64.AppImage"
    download.parent.mkdir()
    download.write_bytes(b"appimage")
    monkeypatch.setenv("APPIMAGE", str(download))
    monkeypatch.delenv("TRACTVIEWER_NO_INTEGRATION", raising=False)
    return tmp_path


def test_not_appimage(monkeypatch):
    monkeypatch.delenv("APPIMAGE", raising=False)
    assert not li.should_offer()


def test_install_and_uninstall(env):
    assert li.should_offer()
    target = li.install(env / "My Apps")
    assert target == env / "My Apps" / "TractViewer.AppImage"
    assert target.read_bytes() == b"appimage" and target.stat().st_mode & 0o111
    entry = li.desktop_file().read_text()
    assert f'Exec="{target}" %F' in entry and "Icon=tractviewer" in entry
    assert f'Exec="{target}" --uninstall' in entry
    assert li.icon_file().read_bytes() == b"png"
    assert li.installed_path() == target
    assert not li.should_offer()
    import os
    assert os.environ["APPIMAGE"] == str(target)  # the updater now targets the installed copy

    li.desktop_file().unlink()
    li.refresh()
    assert li.desktop_file().is_file()

    li.uninstall()
    assert not target.exists() and not li.desktop_file().exists() and not li.icon_file().exists()


def test_decline_is_remembered(env):
    li.decline()
    assert not li.should_offer()


def test_exec_quoting(tmp_path):
    assert li._quote_exec(tmp_path / 'a "b" $c') == '"' + str(tmp_path) + '/a \\"b\\" \\$c"'
