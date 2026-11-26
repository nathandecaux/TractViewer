"""Desktop integration of the Linux AppImage.

On the first launch of the AppImage ($APPIMAGE set by the AppImage runtime), the desktop app
offers to install it: the user picks a folder (default ~/Applications), the AppImage is copied
there as TractViewer.AppImage, and a menu entry is created:

  ~/.local/share/applications/tractviewer.desktop
  ~/.local/share/icons/hicolor/256x256/apps/tractviewer.png

The answer is remembered in ~/.config/tractviewer/appimage.json (delete it to be asked again).
The menu entry has an "Uninstall" action (`TractViewer.AppImage --uninstall`).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

APP_NAME = "TractViewer"
FILE_NAME = "TractViewer.AppImage"  # fixed name: the auto-updater replaces it in place
DESKTOP_ID = "tractviewer.desktop"
ICON_NAME = "tractviewer"


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


def config_file() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "tractviewer" / "appimage.json"


def desktop_file() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "applications" / DESKTOP_ID


def icon_file() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "icons" / "hicolor" / "256x256" / "apps" / f"{ICON_NAME}.png"


def default_dir() -> Path:
    return Path.home() / "Applications"


def current_appimage() -> Optional[Path]:
    path = os.environ.get("APPIMAGE")
    return Path(path) if sys.platform.startswith("linux") and path else None


def _load() -> dict:
    try:
        return json.loads(config_file().read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    f = config_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=2))


def installed_path() -> Optional[Path]:
    path = _load().get("installed")
    return Path(path) if path and Path(path).is_file() else None


def should_offer() -> bool:
    """True on the first launch of an AppImage that is neither installed nor declined."""
    if current_appimage() is None or os.environ.get("TRACTVIEWER_NO_INTEGRATION"):
        return False
    state = _load()
    return not state.get("declined") and installed_path() is None


def decline() -> None:
    _save({**_load(), "declined": True})


# ------------------------------
# Install / uninstall
# ------------------------------
def _quote_exec(path: Path) -> str:
    """Quoting of a path in an Exec= key (Desktop Entry Specification)."""
    s = str(path)
    for ch in ("\\", '"', "`", "$"):
        s = s.replace(ch, "\\" + ch)
    return f'"{s}"'


def _icon_source() -> Optional[Path]:
    for base in (os.environ.get("APPDIR"), getattr(sys, "_MEIPASS", None)):
        if base:
            for name in (f"{ICON_NAME}.png", ".DirIcon"):
                p = Path(base) / name
                if p.is_file():
                    return p
    return None


def write_desktop_entry(appimage: Path) -> Path:
    icon = _icon_source()
    if icon is not None:
        icon_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(icon, icon_file())
    exe = _quote_exec(appimage)
    entry = f"""[Desktop Entry]
Type=Application
Name={APP_NAME}
Comment=Tractography, surface and NIfTI viewer
Exec={exe} %F
TryExec={appimage}
Icon={ICON_NAME if icon is not None else "applications-science"}
Terminal=false
Categories=Science;MedicalSoftware;Graphics;
StartupWMClass={APP_NAME}
Actions=Uninstall;

[Desktop Action Uninstall]
Name=Uninstall {APP_NAME}
Exec={exe} --uninstall
"""
    target = desktop_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(entry)
    target.chmod(0o755)
    _refresh_menus(target.parent)
    return target


def _refresh_menus(applications: Path) -> None:
    for cmd in (["update-desktop-database", str(applications)],
                ["gtk-update-icon-cache", "-q", "-t", str(icon_file().parents[2])]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def install(folder: Path) -> Path:
    """Copy the running AppImage into `folder`, create the menu entry. Returns the new path.
    $APPIMAGE is updated so that the auto-updater targets the installed copy."""
    source = current_appimage()
    if source is None:
        raise RuntimeError("not running from an AppImage")
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / FILE_NAME
    if source.resolve() != target.resolve():
        tmp = target.with_name(f".{FILE_NAME}.part")
        shutil.copyfile(source, tmp)
        tmp.chmod(0o755)
        os.replace(tmp, target)
    write_desktop_entry(target)
    _save({"installed": str(target), "declined": False})
    os.environ["APPIMAGE"] = str(target)
    return target


def refresh() -> None:
    """Installed copy: rewrite the menu entry if it was removed (e.g. menu cleanup)."""
    path = installed_path()
    if path is not None and not desktop_file().is_file():
        write_desktop_entry(path)


def uninstall() -> str:
    """Remove the menu entry, the icon and the installed AppImage."""
    removed = []
    path = installed_path()
    for f in (desktop_file(), icon_file(), path, config_file()):
        if f is not None and f.is_file():
            f.unlink()
            removed.append(str(f))
    _refresh_menus(desktop_file().parent)
    return "Removed:\n  " + "\n  ".join(removed) if removed else "Nothing to remove."
