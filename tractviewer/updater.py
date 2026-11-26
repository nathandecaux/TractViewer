"""Auto-updater of the pre-compiled desktop app (GitHub Releases).

Only active in the frozen builds made by the CI (PyInstaller), and only for the install
types it knows how to replace:

  Windows  installed with the Inno Setup installer   -> runs the new installer silently
  macOS    TractViewer.app                            -> replaces the bundle, relaunches
  Linux    AppImage ($APPIMAGE)                       -> replaces the file, relaunches

Other cases (pip install, source checkout, portable copies) are never modified.

Release assets are matched by name (see .github/workflows/desktop.yml):
  TractViewer-<version>-windows-x64-setup.exe
  TractViewer-<version>-macos-<arm64|x86_64>.zip
  TractViewer-<version>-linux-x86_64.AppImage

Disable with the environment variable TRACTVIEWER_NO_UPDATE=1 (or `desktop --no-update-check`).
"""
from __future__ import annotations
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from tractviewer._version import __version__

REPO = os.environ.get("TRACTVIEWER_UPDATE_REPO", "nathandecaux/TractViewer")
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
USER_AGENT = f"TractViewer-updater/{__version__}"

Progress = Callable[[int, int], None]  # (downloaded bytes, total bytes or 0)


@dataclass
class Update:
    version: str
    url: str           # asset download URL
    name: str          # asset file name
    size: int
    notes_url: str     # release page


# ------------------------------
# Version / install detection
# ------------------------------
def parse_version(text: str) -> Tuple[int, ...]:
    """'v1.2.3' / '1.2.3rc1' -> (1, 2, 3). Non-numeric suffixes are ignored."""
    nums = re.findall(r"\d+", text.lstrip("vV").split("-")[0].split("+")[0])
    return tuple(int(n) for n in nums[:3]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def _arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def install_kind() -> Optional[str]:
    """'windows-installer', 'macos-app', 'appimage' or None (no self-update possible)."""
    if os.environ.get("TRACTVIEWER_NO_UPDATE"):
        return None
    if sys.platform.startswith("linux") and os.environ.get("APPIMAGE"):
        return "appimage"
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    if sys.platform == "win32" and any(exe.parent.glob("unins*.exe")):
        return "windows-installer"
    if sys.platform == "darwin" and _app_bundle(exe) is not None:
        return "macos-app"
    return None


def _app_bundle(exe: Path) -> Optional[Path]:
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None


def _asset_suffix(kind: str) -> str:
    return {
        "windows-installer": "-windows-x64-setup.exe",
        "macos-app": f"-macos-{_arch()}.zip",
        "appimage": f"-linux-{_arch()}.AppImage",
    }[kind]


# ------------------------------
# Check / download
# ------------------------------
def _get(url: str, timeout: float = 10):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)


def check(kind: Optional[str] = None) -> Optional[Update]:
    """Latest release newer than this version with an asset for this install, or None.
    Never raises (no network, rate limit...)."""
    kind = kind or install_kind()
    if kind is None:
        return None
    try:
        with _get(API_LATEST) as resp:
            release = json.load(resp)
    except Exception:  # noqa: BLE001 - offline, rate limit, bad JSON: silently skip
        return None
    tag = release.get("tag_name", "")
    if release.get("draft") or release.get("prerelease") or not is_newer(tag):
        return None
    suffix = _asset_suffix(kind)
    for asset in release.get("assets", []):
        if asset.get("name", "").endswith(suffix):
            return Update(version=tag.lstrip("vV"), url=asset["browser_download_url"],
                          name=asset["name"], size=int(asset.get("size") or 0),
                          notes_url=release.get("html_url", ""))
    return None


def download(update: Update, progress: Optional[Progress] = None) -> Path:
    folder = Path(tempfile.mkdtemp(prefix="tractviewer_update_"))
    target = folder / update.name
    with _get(update.url, timeout=60) as resp, open(target, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or update.size or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if update.size and target.stat().st_size != update.size:
        raise OSError(f"incomplete download ({target.stat().st_size} / {update.size} bytes)")
    return target


# ------------------------------
# Install (the caller must quit the app right after)
# ------------------------------
def apply(file: Path, kind: Optional[str] = None) -> None:
    """Start the installation of a downloaded update. The application must exit right after:
    the replacement happens (Windows, macOS) once this process is gone, then the app restarts."""
    kind = kind or install_kind()
    if kind == "windows-installer":
        _apply_windows(file)
    elif kind == "macos-app":
        _apply_macos(file)
    elif kind == "appimage":
        _apply_appimage(file)
    else:
        raise RuntimeError("this installation cannot be updated automatically")


def _apply_windows(installer: Path) -> None:
    # Per-user Inno Setup installer: no UAC prompt. It waits for this process to exit
    # (CloseApplications), installs over the current version and relaunches the app.
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                      "/CLOSEAPPLICATIONS", "/SP-"], creationflags=flags, close_fds=True)


def _apply_macos(archive: Path) -> None:
    app = _app_bundle(Path(sys.executable).resolve())
    if app is None:
        raise RuntimeError("TractViewer.app not found")
    staging = archive.parent / "new"
    subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(staging)], check=True)
    new_app = next(staging.glob("*.app"))
    # Swap the bundles once this process has exited, then relaunch.
    script = f"""#!/bin/sh
while kill -0 {os.getpid()} 2>/dev/null; do sleep 0.5; done
rm -rf {shlex.quote(str(app))}.old
mv {shlex.quote(str(app))} {shlex.quote(str(app))}.old && \\
mv {shlex.quote(str(new_app))} {shlex.quote(str(app))} && \\
rm -rf {shlex.quote(str(app))}.old
/usr/bin/xattr -dr com.apple.quarantine {shlex.quote(str(app))} 2>/dev/null
open {shlex.quote(str(app))}
rm -rf {shlex.quote(str(archive.parent))}
"""
    path = archive.parent / "update.sh"
    path.write_text(script)
    subprocess.Popen(["/bin/sh", str(path)], start_new_session=True, close_fds=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _apply_appimage(new: Path) -> None:
    current = Path(os.environ["APPIMAGE"])
    # Same directory as the target so that os.replace is atomic (same filesystem).
    tmp = current.with_name(f".{current.name}.update")
    with open(new, "rb") as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    tmp.chmod(current.stat().st_mode | 0o111)
    os.replace(tmp, current)  # the running instance keeps its (unlinked) inode
    subprocess.Popen([str(current), *sys.argv[1:]], start_new_session=True, close_fds=True)
