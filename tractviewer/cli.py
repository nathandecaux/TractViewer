import argparse
from pathlib import Path
from .core import TractViewer
from typing import Dict

def build_parser():
    p = argparse.ArgumentParser(prog="tractviewer", description="Visualisation headless de tractographies / surfaces")
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Fichiers à charger. Paramètres par fichier possibles avec syntaxe: "
            "chemin:clé=val,clé=val (ex: bundle.vtk:scalar=FA,opacity=0.4,color=red)"
        ),
    )
    p.add_argument("--background", default="white", help="Couleur de fond (par défaut: white)")
    p.add_argument("--off-screen", action="store_true", help="Force le mode hors écran")
    p.add_argument("--interactive", action="store_true", help="Force l'ouverture de la fenêtre interactive même si capture/rotation")
    p.add_argument("--screenshot", type=Path, help="Chemin de capture PNG après chargement")
    p.add_argument("--rotate", type=int, default=0, help="Nombre de frames de rotation à enregistrer (0=pas de rotation)")
    p.add_argument("--rotation-output", type=Path, help="Fichier vidéo ou gif de rotation (ex: rotation.mp4)")
    p.add_argument("--gif", action="store_true", help="Forcer GIF pour la rotation")
    p.add_argument("--step", type=float, default=2.0, help="Pas azimut (deg) entre frames de rotation")
    p.add_argument("--window-size", type=str, help="Taille fenêtre WxH (ex: 1280x720)")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.off_screen and args.interactive:
        parser.error("--off-screen et --interactive sont mutuellement exclus")

    off_screen = args.off_screen or (not args.interactive and bool(args.screenshot or args.rotate))

    window_size = None
    if args.window_size:
        try:
            w, h = map(int, args.window_size.lower().split("x"))
            window_size = (w, h)
        except Exception:
            parser.error("Format --window-size attendu: WxH (ex: 1280x720)")

    vis = TractViewer(background=args.background, off_screen=off_screen, window_size=window_size)
    for raw in args.inputs:
        path_str, meta = _parse_input_spec(raw)
        meta.setdefault("name", Path(path_str).stem)
        vis.add_dataset(path_str, meta)

    if args.screenshot:
        vis.capture_screenshot(args.screenshot)

    if args.rotate and args.rotate > 0:
        output = args.rotation_output or Path("rotation.mp4")
        vis.record_rotation(output_path=output, n_frames=args.rotate, step=args.step, gif=args.gif)

    if args.interactive or (not off_screen and not (args.screenshot or args.rotate)):
        vis.show()

if __name__ == "__main__":
    main()


def _parse_input_spec(spec: str):
    """Découpe 'chemin:clé=val,clé=val' -> (chemin, meta).

    Retourne (spec, {}) si aucun paramètre.
    """
    if ":" not in spec:
        return spec, {}
    path_part, param_part = spec.split(":", 1)
    if not param_part:
        return path_part, {}
    meta: Dict[str, str] = {}
    for chunk in param_part.split(","):
        if not chunk:
            continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            meta[k.strip()] = v.strip()
        else:  # flag style
            meta[chunk.strip()] = "true"
    return path_part, meta
