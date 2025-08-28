"""Cœur du package tractviewer.

Fournit la classe TractViewer avec un petit wrapper simplifié autour de PyVista
pour charger des jeux de données et produire captures / rotations en mode
interactif ou hors-écran.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import os
try:  # import léger différé si possible
    import pyvista as pv
except Exception as e:  # pragma: no cover
    pv = None  # type: ignore
    _import_error = e  # stocker pour message explicite
else:
    _import_error = None

try:
    import imageio.v2 as imageio  # pour enregistrement rotation
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

try:
    import nibabel as nib  # pour .trk éventuels
except Exception:  # pragma: no cover
    nib = None  # type: ignore


@dataclass
class LoadedDataset:
    path: Path
    actor: Any
    meta: Dict[str, Any]


class TractViewer:
    """Visualisation simple headless.

    Paramètres
    ---------
    background : str | tuple
        Couleur de fond.
    off_screen : bool
        Force le rendu hors-écran (utile sur serveur sans X11).
    window_size : tuple[int, int] | None
        Taille de la fenêtre / rendu.
    """

    def __init__(
        self,
        background: str | Sequence[float] = "white",
        off_screen: bool = False,
        window_size: Optional[Sequence[int]] = None,
    ) -> None:
        if pv is None:  # pragma: no cover - dépendance externe
            raise RuntimeError(
                "PyVista/VTK introuvable: installez dépendances (pyvista, vtk-osmesa pour off-screen)."
            ) from _import_error

        # Collections internes
        self._datasets: List[LoadedDataset] = []
        self._plotter: Optional[Any] = None  # sera créé à la demande

        # Paramètres
        self.background = background
        self.off_screen = off_screen or os.environ.get("PYVISTA_OFF_SCREEN", "").lower() in {"1", "true"}
        self.window_size = tuple(map(int, window_size)) if window_size else None

        if self.off_screen:
            # activation globale (certaines versions pyvista le lisent)
            pv.OFF_SCREEN = True  # type: ignore[attr-defined]
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    # ------------------------------------------------------------------ utils internes
    def _ensure_plotter(self):
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=self.off_screen, window_size=self.window_size)
            self._plotter.set_background(self.background)
        return self._plotter

    # ------------------------------------------------------------------ chargement
    def add_dataset(self, path: str | Path, meta: Optional[Dict[str, Any]] = None) -> LoadedDataset:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        mesh = self._read_any(p)
        meta = dict(meta or {})

        # Sélection d'un scalaire pour coloration
        scalar_name = meta.get("scalar") or meta.get("scalars")
        if scalar_name and hasattr(mesh, "array_names"):
            try:
                if scalar_name in mesh.array_names:  # type: ignore[attr-defined]
                    mesh.set_active_scalars(scalar_name)  # type: ignore[call-arg]
            except Exception:  # silencieux
                pass

        # Gestion des points / sphères
        # Clés reconnues : points / as_points / spheres (bool), point_size (float), point_radius (float -> glyph sphères)
        wants_points = any(k in meta for k in ("points", "as_points", "spheres"))
        point_radius = None
        if "point_radius" in meta:
            try:
                point_radius = float(meta["point_radius"])  # type: ignore[arg-type]
            except Exception:
                point_radius = None
        # Si un radius explicite est fourni on glyph avec des sphères plutôt que simple point size
        if point_radius and hasattr(mesh, "n_points") and mesh.n_points > 0:  # type: ignore[attr-defined]
            try:
                import pyvista as _pv  # local
                sphere = _pv.Sphere(radius=point_radius)
                mesh = mesh.glyph(geom=sphere, scale=False)
                wants_points = False  # déjà converti en surface
            except Exception:
                pass

        plotter = self._ensure_plotter()
        add_kwargs: Dict[str, Any] = {}
        if meta.get("color"):
            add_kwargs["color"] = meta["color"]
        if meta.get("opacity"):
            try:
                add_kwargs["opacity"] = float(meta["opacity"])  # type: ignore[arg-type]
            except Exception:
                pass
        if meta.get("cmap"):
            add_kwargs["cmap"] = meta["cmap"]
        if meta.get("line_width"):
            try:
                add_kwargs["line_width"] = float(meta["line_width"])  # type: ignore[arg-type]
            except Exception:
                pass
        if wants_points:
            add_kwargs["render_points_as_spheres"] = True
        if meta.get("point_size"):
            try:
                add_kwargs["point_size"] = float(meta["point_size"])  # type: ignore[arg-type]
            except Exception:
                pass

        actor = plotter.add_mesh(mesh, name=p.stem, **add_kwargs)
        ld = LoadedDataset(p, actor, meta or {})
        self._datasets.append(ld)
        return ld

    def _read_any(self, p: Path):  # pragma: no cover - simple bascule
        # Tentative générique
        try:
            return pv.read(p)
        except Exception:
            if p.suffix.lower() == ".trk" and nib is not None:  # conversion simplifiée
                trk = nib.streamlines.load(str(p))
                from numpy import concatenate, asarray

                lines = trk.streamlines
                points_list = []
                cells = []
                offset = 0
                for s in lines:
                    n = len(s)
                    points_list.append(s)
                    cells.append([n, *range(offset, offset + n)])
                    offset += n
                import numpy as np

                # IMPORTANT: ne pas ré-importer "pyvista as pv" ici (créait UnboundLocalError)
                points = np.vstack(points_list)
                # VTK cell array format: [n, ids..., n, ids...]
                cells_arr = np.concatenate([asarray(c, dtype=np.int32) for c in cells])
                poly = pv.PolyData(points)
                poly.lines = cells_arr
                return poly
            raise

    # ------------------------------------------------------------------ actions
    def show(self):  # pragma: no cover - interactif
        self._ensure_plotter().show()

    def capture_screenshot(self, path: str | Path) -> Path:
        plotter = self._ensure_plotter()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out))
        return out

    def record_rotation(
        self,
        output_path: str | Path,
        n_frames: int = 180,
        step: float = 2.0,
        gif: bool = False,
    ) -> Path:
        """Enregistre une rotation azimutale simple.

        Paramètres
        ----------
        output_path : str | Path
            Fichier de sortie (.mp4, .gif, .png seq si souhaité).
        n_frames : int
            Nombre de frames générées.
        step : float
            Pas en degrés pour l'azimut par frame.
        gif : bool
            Force la sortie GIF (sinon déduit de l'extension).
        """
        if imageio is None:  # pragma: no cover
            raise RuntimeError("imageio manquant pour l'enregistrement de rotation")
        plotter = self._ensure_plotter()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        is_gif = gif or out.suffix.lower() == ".gif"
        frames: List[Any] = []
        # Rendu initial pour stabiliser l'exposition / auto-range
        plotter.render()
        cam = plotter.camera
        for i in range(n_frames):
            # Rotation azimutale
            try:
                cam.Azimuth(step)  # type: ignore[attr-defined]
            except Exception:
                # fallback: incrémenter attribut si présent
                if hasattr(cam, "azimuth") and isinstance(getattr(cam, "azimuth"), (int, float)):
                    try:
                        setattr(cam, "azimuth", getattr(cam, "azimuth") + step)
                    except Exception:
                        pass
            # Forcer rendu avant capture sinon certaines versions n'appliquent pas la modif caméra
            plotter.render()
            img = plotter.screenshot(return_img=True)
            frames.append(img)
        if is_gif:
            imageio.mimsave(out, frames, fps=max(1, int(360 / (n_frames * step))) )
        else:
            imageio.mimsave(out, frames, fps=25)
        return out

    # ------------------------------------------------------------------ contexte
    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        if self._plotter is not None:
            try:
                self._plotter.close()
            finally:
                self._plotter = None
        return False

__all__ = ["TractViewer", "LoadedDataset"]
