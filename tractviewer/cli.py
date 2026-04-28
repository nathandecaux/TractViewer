import argparse
from pathlib import Path
from .core import TractViewer
from typing import Dict
import re
import threading

def build_parser():
    p = argparse.ArgumentParser(prog="tractviewer", description="Visualisation headless de tractographies / surfaces")
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Fichiers à charger. Paramètres par fichier avec syntaxe étendue:\n"
            "chemin:clé=val,clé=val,...\n"
            "Types auto: int, float, bool (true/false), None.\n"
            "Cast de display_array: type=int ou type=float.\n"
            "Listes / tuples: utiliser parenthèses. Ex:\n"
            "  volume.nii.gz:display_array=intensity,cmap=gray,clim=(200,800),opacity=0.3,show_scalar_bar=false,ambient=0.6,specular=0.1,diffuse=0.8,style=surface\n"
            "Threshold: threshold=(array,min,max)  -> threshold=(FA,0.2,0.8)\n"
            "clim: clim=(min,max)\n"
            "Remarque: Les valeurs multi-parties DOIVENT être entre parenthèses."
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
    p.add_argument("--rotation-x", type=float, default=-90.0, help="Rotation du mesh autour de l'axe X en degrés (défaut: -90)")
    p.add_argument("--rotation-y", type=float, default=0.0, help="Rotation du mesh autour de l'axe Y en degrés (défaut: 0)")
    p.add_argument("--rotation-z", type=float, default=0.0, help="Rotation du mesh autour de l'axe Z en degrés (défaut: 0)")
    p.add_argument("--no-marching-cubes", action="store_true", help="Désactive le marching cubes pour les NIfTI (charge le volume brut)")
    p.add_argument("--smooth", type=int, default=0, help="Nombre d'itérations de smoothing Laplacien après marching cubes (0=désactivé)")
    p.add_argument("--color", type=str, default=None, help="Couleur des surfaces NIfTI (ex: white, #ff8800, red)")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.off_screen and args.interactive:
        parser.error("--off-screen et --interactive sont mutuellement exclus")

    off_screen = args.off_screen or (not args.interactive and bool(args.screenshot or args.rotate))
    print('Mode off-screen' if off_screen else 'Mode interactif')
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
        if args.no_marching_cubes and "marching_cubes" not in meta:
            meta["marching_cubes"] = False
        if args.smooth and "smooth" not in meta:
            meta["smooth"] = args.smooth
        if args.color and "color" not in meta:
            meta["color"] = args.color
        vis.add(path_str, meta)

    if args.screenshot:
        vis.capture_screenshot(
            args.screenshot,
            rotation_x=args.rotation_x,
            rotation_y=args.rotation_y,
            rotation_z=args.rotation_z
        )

    if args.rotate and args.rotate > 0:
        output = args.rotation_output or Path("rotation.mp4")
        vis.record_rotation(
            output_path=output,
            n_frames=args.rotate,
            step=args.step,
            gif=args.gif,
            rotation_x=args.rotation_x,
            rotation_y=args.rotation_y,
            rotation_z=args.rotation_z
        )

    if args.interactive or (not off_screen and not (args.screenshot or args.rotate)):
        vis.show()

def _run_plotter(vis: TractViewer):
    """
    Exécute le plotter dans un thread séparé.
    """
    try:
        vis.show()
    except Exception as e:
        print(f"Erreur dans le thread du plotter : {e}")

def _coerce_value(value: str):
    """
    Convertit une valeur de chaîne en un type approprié (int, float, bool, None, ou str).
    """
    value = value.strip()
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    if value.lower() in ("none", "null"):
        return None
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value  # Retourne la chaîne brute si aucune conversion n'est possible

if __name__ == "__main__":
    main()


def _parse_input_spec(spec: str):
    """Découpe 'chemin:clé=val,clé=val' -> (chemin, meta) avec parsing typé.

    Nouveautés:
      - bool: true/false/yes/no
      - None
      - int / float
      - tuples via '(a,b,...)'
      - threshold=(array,min,max) -> ('array', (min,max))
      - clim=(min,max) -> (min,max)

    Si aucun ':', retourne (spec, {}).
    """
    if ":" not in spec:
        return spec, {}
    path_part, param_part = spec.split(":", 1)
    if not param_part.strip():
        return path_part, {}

    def split_top_level(s: str):
        tokens = []
        current = []
        depth = 0
        in_quote = False
        quote_char = ''
        for ch in s:
            if in_quote:
                if ch == quote_char:
                    in_quote = False
                current.append(ch)
                continue
            if ch in ("'", '"'):
                in_quote = True
                quote_char = ch
                current.append(ch)
                continue
            if ch in "([{" :
                depth += 1
                current.append(ch)
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                current.append(ch)
                continue
            if ch == ',' and depth == 0:
                token = ''.join(current).strip()
                if token:
                    tokens.append(token)
                current = []
                continue
            current.append(ch)
        last = ''.join(current).strip()
        if last:
            tokens.append(last)
        return tokens

    def coerce_scalar(v: str):
        vl = v.strip()
        if re.fullmatch(r"(?i)none|null", vl):
            return None
        if re.fullmatch(r"(?i)true|yes|on", vl):
            return True
        if re.fullmatch(r"(?i)false|no|off", vl):
            return False
        # int
        if re.fullmatch(r"[+-]?\d+", vl):
            try:
                return int(vl)
            except Exception:
                pass
        # float
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", vl):
            try:
                return float(vl)
            except Exception:
                pass
        # strip quotes
        if (vl.startswith("'") and vl.endswith("'")) or (vl.startswith('"') and vl.endswith('"')):
            return vl[1:-1]
        return vl  # string brut

    def coerce_value(key: str, raw: str):
        raw = raw.strip()
        # type=<int|float> : directive de cast pour display_array
        if key == "type":
            if raw in ("int", "float"):
                return raw
            raise ValueError(f"Valeur non supportée pour 'type': {raw!r}. Utilisez 'int' ou 'float'.")
        if raw.startswith("(") and raw.endswith(")"):
            inner = raw[1:-1].strip()
            if not inner:
                return ()
            parts = [p.strip() for p in split_top_level(inner)]
            coerced = [coerce_scalar(p) for p in parts]
            # Cas spéciaux
            if key == "threshold" and len(coerced) == 3:
                arr, vmin, vmax = coerced
                return (str(arr), (float(vmin), float(vmax)))
            if key == "clim" and len(coerced) == 2:
                return (float(coerced[0]), float(coerced[1]))
            return tuple(coerced)
        # Cas threshold format simplifié arr,min,max sans parenthèses -> on tente (si séparé par ;)
        if key == "threshold" and ";" in raw:
            parts = [p.strip() for p in raw.split(";")]
            if len(parts) == 3:
                arr, vmin, vmax = [coerce_scalar(p) for p in parts]
                return (str(arr), (float(vmin), float(vmax)))
        # clim min:max
        if key == "clim" and ":" in raw:
            a, b = raw.split(":", 1)
            try:
                return (float(a), float(b))
            except Exception:
                pass
        return coerce_scalar(raw)

    meta: Dict[str, object] = {}
    tokens = split_top_level(param_part)
    for tok in tokens:
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip()
            v = v.strip()
            meta[k] = coerce_value(k, v)
        else:
            # flag -> True
            meta[tok.strip()] = True

    # Post-traitements supplémentaires
    # Normaliser certaines clés booléennes
    for bkey in ("show_scalar_bar", "show_edges"):
        if bkey in meta:
            meta[bkey] = bool(meta[bkey])

    # Opacity: si liste/tuple numérique -> laisser tel quel sinon convertir float
    if "opacity" in meta and isinstance(meta["opacity"], (int, float, str)):
        try:
            meta["opacity"] = float(meta["opacity"])
        except Exception:
            pass

    return path_part, meta
