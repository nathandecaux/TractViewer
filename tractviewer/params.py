"""Dataset parameters: known keys, aliases, CLI syntax parsing.

Shared by the API (`TractViewer.add/edit`), the CLI and the web interface.

CLI input syntax: ``path:key=val,key=val,...``
  - bool: true/false, yes/no, on/off ; None/null
  - int / float
  - tuples: ``(a,b,...)``
  - ``threshold=(array,min,max)`` or ``threshold=array;min;max``
  - ``clim=(min,max)`` or ``clim=min:max``
  - ``type=int|float``: cast of the displayed array
  - bare key (no ``=``) => True
"""
from __future__ import annotations
import difflib
import re
import warnings
from typing import Dict, List, Tuple

from tractviewer._deprecation import warn_deprecated

ParamDict = Dict[str, object]

# Display parameters (read by tractviewer.scene) and loading parameters (read by add()).
DISPLAY_PARAMS = {
    "name", "visible", "display_array", "type", "cmap", "clim", "color", "threshold",
    "opacity", "style", "show_edges", "edge_color", "contour", "contour_width",
    "scalar_bar", "scalar_bar_args", "legend", "categorical_legend",
    "categorical_max_values", "unique_values", "point_size", "line_width",
    "points_as_spheres", "ambient", "specular", "diffuse",
    "envelope_voxel_size", "envelope_smooth",
}
LOAD_PARAMS = {"marching_cubes", "smooth", "iso"}
KNOWN_PARAMS = DISPLAY_PARAMS | LOAD_PARAMS

# Alternative name -> canonical name.
ALIASES = {"show_scalar_bar": "scalar_bar"}
# Former (French) spellings -> canonical name, accepted with a deprecation warning.
DEPRECATED_ALIASES = {
    "enveloppe_voxel_size": "envelope_voxel_size",
    "enveloppe_smooth": "envelope_smooth",
}


def normalize(params: ParamDict) -> ParamDict:
    """Resolve aliases and warn about unknown keys (usually a typo)."""
    out = dict(params)
    for alias, canonical in ALIASES.items():
        if alias in out:
            out.setdefault(canonical, out.pop(alias))
    for alias, canonical in DEPRECATED_ALIASES.items():
        if alias in out:
            warn_deprecated(f"parameter '{alias}'", f"'{canonical}'", stacklevel=4)
            out.setdefault(canonical, out.pop(alias))
    if out.get("style") == "enveloppe":
        warn_deprecated("style='enveloppe'", "style='envelope'", stacklevel=4)
        out["style"] = "envelope"
    for key in out:
        if key not in KNOWN_PARAMS:
            close = difflib.get_close_matches(key, KNOWN_PARAMS, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            warnings.warn(f"Unknown parameter '{key}' ignored.{hint}", stacklevel=3)
    return out


def parse_threshold(value: str) -> Tuple[str, Tuple[float, float]]:
    """'FA,0.2,0.8' -> ('FA', (0.2, 0.8))."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("expected format: array,min,max")
    return parts[0], (float(parts[1]), float(parts[2]))


def parse_spec(spec: str) -> Tuple[str, ParamDict]:
    """Split 'path:key=val,key=val' -> (path, typed params).

    Without ':', return (spec, {}).
    """
    if ":" not in spec:
        return spec, {}
    path_part, param_part = spec.split(":", 1)

    params: ParamDict = {}
    for tok in _split_top_level(param_part):
        if "=" in tok:
            key, raw = (s.strip() for s in tok.split("=", 1))
            params[key] = _coerce_value(key, raw)
        else:
            params[tok] = True  # flag

    for bkey in ("show_scalar_bar", "scalar_bar", "show_edges"):
        if bkey in params:
            params[bkey] = bool(params[bkey])
    if isinstance(params.get("opacity"), (int, float, str)):
        try:
            params["opacity"] = float(params["opacity"])
        except ValueError:
            pass  # e.g. opacity=linear (pyvista preset)
    return path_part, params


def _split_top_level(s: str) -> List[str]:
    """Split on commas outside parentheses/brackets/quotes."""
    tokens, current = [], []
    depth = 0
    quote = ""
    for ch in s:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tokens.append("".join(current).strip())
    return [t for t in tokens if t]


def _coerce_scalar(v: str):
    v = v.strip()
    if re.fullmatch(r"(?i)none|null", v):
        return None
    if re.fullmatch(r"(?i)true|yes|on", v):
        return True
    if re.fullmatch(r"(?i)false|no|off", v):
        return False
    if re.fullmatch(r"[+-]?\d+", v):
        return int(v)
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", v):
        return float(v)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _coerce_value(key: str, raw: str):
    if key == "type":
        if raw in ("int", "float"):
            return raw
        raise ValueError(f"Unsupported value for 'type': {raw!r}. Use 'int' or 'float'.")

    if raw.startswith("(") and raw.endswith(")"):
        parts = [_coerce_scalar(p) for p in _split_top_level(raw[1:-1])]
        if key == "threshold" and len(parts) == 3:
            return str(parts[0]), (float(parts[1]), float(parts[2]))
        if key == "clim" and len(parts) == 2:
            return float(parts[0]), float(parts[1])
        return tuple(parts)

    if key == "threshold" and ";" in raw:
        parts = [p.strip() for p in raw.split(";")]
        if len(parts) == 3:
            return parts[0], (float(parts[1]), float(parts[2]))
    if key == "clim" and ":" in raw:
        a, b = raw.split(":", 1)
        try:
            return float(a), float(b)
        except ValueError:
            pass
    return _coerce_scalar(raw)
