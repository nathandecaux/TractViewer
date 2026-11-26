#!/usr/bin/env bash
# Build an AppImage from the PyInstaller output.
#   packaging/linux/build_appimage.sh dist/TractViewer dist/TractViewer-1.2.3-linux-x86_64.AppImage
set -euo pipefail
SRC=$(realpath "$1"); OUT=$(realpath -m "$2")
HERE=$(dirname "$(realpath "$0")")
WORK=$(mktemp -d)
APPDIR="$WORK/TractViewer.AppDir"

mkdir -p "$APPDIR/usr/lib"
cp -a "$SRC" "$APPDIR/usr/lib/TractViewer"
cp "$HERE/tractviewer.desktop" "$APPDIR/tractviewer.desktop"
cp "$HERE/../icon.png" "$APPDIR/tractviewer.png"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/TractViewer/TractViewer" "$@"
RUN
chmod +x "$APPDIR/AppRun"

TOOL="${APPIMAGETOOL:-$WORK/appimagetool}"
if [ ! -x "$TOOL" ]; then
  curl -fsSL -o "$TOOL" \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$(uname -m).AppImage"
  chmod +x "$TOOL"
fi
# --appimage-extract-and-run: no FUSE needed (CI containers)
ARCH=$(uname -m) "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"
rm -rf "$WORK"
echo "$OUT"
