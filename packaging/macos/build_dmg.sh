#!/usr/bin/env bash
# Package dist/TractViewer.app:
#   <name>.dmg  drag-and-drop installer (first install)
#   <name>.zip  same app, used by the auto-updater
#   packaging/macos/build_dmg.sh dist/TractViewer.app dist/TractViewer-1.2.3-macos-arm64
set -euo pipefail
APP=$1; BASE=$2
# Ad-hoc signature of the whole bundle (required on Apple Silicon; no Apple ID involved).
codesign --force --deep --sign - "$APP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$BASE.zip"
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname TractViewer -srcfolder "$STAGE" -ov -format UDZO "$BASE.dmg"
rm -rf "$STAGE"
