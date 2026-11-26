# Desktop app

The desktop app shows the TractViewer [interface](interface.md) in a native window. Everything
runs locally: no browser, no network needed (vtk.js is bundled in the app).

## Installation

Download the file for your system from the
[latest release](https://github.com/nathandecaux/TractViewer/releases/latest).

### Windows

Run `TractViewer-<version>-windows-x64-setup.exe`. It installs for the current user only
(in `%LOCALAPPDATA%\Programs\TractViewer`), without admin rights, and adds a Start menu entry
(and optionally a desktop shortcut). Uninstall from *Settings › Apps*.

If SmartScreen warns about an unknown publisher (the installer is not signed): *More info ›
Run anyway*.

### macOS

Open `TractViewer-<version>-macos-arm64.dmg` (Apple Silicon) or `-macos-x86_64.dmg` (Intel) and
drag TractViewer into *Applications*.

The app is not notarized by Apple. The first time, right click on TractViewer.app › *Open*, then
confirm. Or, in a terminal:

```bash
xattr -dr com.apple.quarantine /Applications/TractViewer.app
```

Uninstall: move TractViewer.app to the trash.

### Linux

```bash
chmod +x TractViewer-<version>-linux-x86_64.AppImage
./TractViewer-<version>-linux-x86_64.AppImage
```

At the first launch, the app offers to install itself:

1. choose a folder (default `~/Applications`): the AppImage is copied there as
   `TractViewer.AppImage`;
2. an entry is added to the applications menu (`~/.local/share/applications/tractviewer.desktop`,
   icon in `~/.local/share/icons/`), which can also open files with TractViewer;
3. the downloaded file can be deleted.

If you decline, the app runs without installing and does not ask again (delete
`~/.config/tractviewer/appimage.json` to be asked again, or set `TRACTVIEWER_NO_INTEGRATION=1`
to never be asked).

Uninstall: right click on the menu entry › *Uninstall TractViewer*, or
`~/Applications/TractViewer.AppImage --uninstall`. It removes the menu entry, the icon and the
installed AppImage.

Requirements: a 64-bit distribution with glibc ≥ 2.35 (Ubuntu 22.04, Debian 12, Fedora 36 or
newer). Some distributions need `libfuse2` to run AppImages
(`sudo apt install libfuse2`); otherwise run it with `--appimage-extract-and-run`.

## Updates

At startup, the app checks the latest release on GitHub. When a newer version exists, it asks
whether to install it, downloads it (progress in the bottom-right corner), replaces the app and
restarts. Nothing happens without network access.

Disable the check with `TRACTVIEWER_NO_UPDATE=1` (environment variable) or
`--no-update-check`. Pre-releases (`v1.2.0-rc1`) are never offered.

## Using the app

- Open files with *Add files* or by dropping them on the view. They can also be passed on the
  command line (`TractViewer anat.nii.gz bundle.trk`) or, on Linux once installed, opened with
  TractViewer from the file manager (*Open with*).
- PNG captures, GIFs and HTML exports open a native *Save as* dialog.
- The pre-compiled app also accepts the [command line](cli.md) subcommands, e.g.
  `TractViewer info bundle.trk` (Linux / macOS terminal).

See the [interface guide](interface.md) for all features.

## From the Python package

With the Python package installed (`pip install git+https://github.com/nathandecaux/TractViewer.git`):

```bash
tractviewer desktop                          # empty window, add files from the interface
tractviewer desktop anat.nii.gz bundle.trk   # start with some files
```

or from Python: `vis.desktop()` (see the [Python API](python-api.md)). This mode never updates
itself.

The window uses [pywebview](https://pywebview.flowrl.com/). The server only listens on 127.0.0.1
(free port) and stops when the window is closed. vtk.js and the GIF encoder are downloaded once
into `~/.cache/tractviewer/vendor/` and then served locally, so it works offline afterwards.
Webview downloads are unreliable (e.g. broken with Qt 6 in pywebview), so the page sends exported
files to the local server, which shows the *Save as* dialog.

Backend: the system webview on Windows (WebView2) and macOS (WKWebView). On Linux, the Qt
backend is installed as a dependency (PyQt6 + Qt WebEngine wheels). GTK/WebKitGTK is used instead
when PyGObject is installed (e.g. `conda install pygobject` or the distribution package).

Options: `--title`, `--max-streamlines`, `--max-voxels`, `--debug` (developer tools),
`--no-update-check`, plus the [common options](cli.md#common-options).

## Troubleshooting

- **Blank window on Linux**: try `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu`, or run with
  `--debug` and open the developer tools.
- **Large tractograms**: they are subsampled at loading according to the available RAM, see
  [Large tractograms](interface.md#large-tractograms).
