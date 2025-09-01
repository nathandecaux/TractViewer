from __future__ import annotations
from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence, Tuple
import os
import warnings
import math
import numpy as np
import contextlib
import pyvista as pv
import subprocess
import shutil
import tempfile
from tractviewer.enveloppe import enveloppe_minimale  # ajout

try:
    import imageio.v2 as imageio  # GIF option
except Exception:  # pragma: no cover
    imageio = None


DataInput = Union[str, Path, pv.DataSet]
ParamDict = Dict[str, object]

os.environ["TRACTVIEWER_NIFTI_COORD"] = "LPS"

class TractViewer:
    """
    Visualisation simplifiée de fichiers/datasets VTK via pyvista.

    Paramètres par dataset (clé -> signification):
      - display_array: str | None  nom de l'array scalaires à afficher
      - cmap: str | Sequence[str]  colormap matplotlib
      - clim: Tuple[float,float]   limites (min, max) pour l'affichage
      - threshold: (str, (min,max))  filtrage sur un array => applique un threshold
      - opacity: float | str | Sequence[float]
      - show_edges: bool
      - scalar_bar: bool
      - point_size / line_width: tailles optionnelles
      - ambient / specular / diffuse: réglages matériaux
      - style: 'surface' | 'wireframe' | 'points' | 'enveloppe'
        * enveloppe => calcule et affiche l'enveloppe convexe (ou alpha si futur)
      - render_points_as_spheres: bool (utile si style='points')
      - name: nom interne (sinon auto)
    Paramètres globaux (constructeur):
      - background: couleur du fond (default "white")
      - off_screen: bool (pour rendu sans interface, ex: serveur)
    """
    def __init__(self, background: str = "white", off_screen: bool = False,window_size: Optional[Tuple[int,int]] = None):
        self._datasets: List[Tuple[pv.DataSet, ParamDict]] = []
        self.background = background
        self.off_screen = off_screen
        self._plotter: Optional[pv.Plotter] = None
        self._scalar_bar_added = False
        self._font_color = self._choose_font_color(self.background)
        self._reference_affine: Optional[np.ndarray] = None  # 4x4 issue du premier NIfTI
        # Auto bascule headless si pas de DISPLAY
        if not self.off_screen and not os.environ.get("DISPLAY"):
            warnings.warn("Aucun DISPLAY détecté -> passage en mode off_screen.")
            self.off_screen = True

    # ------------------------------
    # Chargement / ajout de datasets
    # ------------------------------
    @staticmethod
    def _load(obj: DataInput) -> pv.DataSet:
        """Charge un objet selon son extension.

        Extensions supportées supplémentaires:
          - .tck / .trk : tractographie -> vtkPolyData (lignes)
          - .nii / .nii.gz : NIfTI -> projection surface (isosurface)
        """
        if isinstance(obj, pv.DataSet):
            return obj
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        double_suffix = ''.join(path.suffixes[-2:]).lower()  # pour .nii.gz
        if suffix in (".tck", ".trk"):
            return TractViewer._load_tract_file(path)
        if suffix in (".nii",) or double_suffix == ".nii.gz":
            return TractViewer._load_nifti_surface(path)
        return pv.read(path)

    # --- Chargement spécialisé tractographie ---
    @staticmethod
    def _load_tract_file(path: Path) -> pv.PolyData:
        """Lit un fichier tractographie (.tck ou .trk) et retourne un PolyData.

        Utilise nibabel si disponible. Chaque streamline devient une polyline.
        Ajoute des arrays:
          - 'streamline_id' (cell data)
          - 'point_index' (point data index local dans streamline)
          - 'length_mm' (cell data longueur de la streamline en mm)
        """
        try:
            import nibabel as nib
        except Exception as e:  # pragma: no cover
            raise ImportError("nibabel requis pour charger des fichiers tractographie (.tck/.trk). 'pip install nibabel'") from e

        tractogram = nib.streamlines.load(str(path))
        streamlines = tractogram.streamlines
        # Construction des points / connectivité
        all_points: List[np.ndarray] = []
        lines_idx: List[int] = []
        cell_data_lengths: List[float] = []
        cell_stream_ids: List[int] = []
        point_stream_index: List[int] = []
        running_index = 0
        for sid, sl in enumerate(streamlines):
            if sl.shape[0] < 2:
                continue
            npts = sl.shape[0]
            all_points.append(sl.astype(np.float32))
            # VTK polyline: [npts, p0, p1, ...]
            lines_idx.append(npts)
            lines_idx.extend(range(running_index, running_index + npts))
            # Cell data
            dists = np.linalg.norm(np.diff(sl, axis=0), axis=1)
            cell_data_lengths.append(float(dists.sum()))
            cell_stream_ids.append(sid)
            # Point data (index local)
            point_stream_index.extend(list(range(npts)))
            running_index += npts
        if not all_points:
            raise ValueError(f"Aucune streamline valide dans {path}.")
        points = np.vstack(all_points)
        poly = pv.PolyData()
        poly.points = points
        poly.lines = np.array(lines_idx, dtype=np.int32)
        # Cell data
        poly.cell_data["streamline_id"] = np.array(cell_stream_ids, dtype=np.int32)
        poly.cell_data["length_mm"] = np.array(cell_data_lengths, dtype=np.float32)
        # Point data
        poly.point_data["point_index"] = np.array(point_stream_index, dtype=np.int32)
        return poly

    # --- Chargement / projection NIfTI ---
    @staticmethod
    def _load_nifti_surface(path: Path) -> pv.PolyData:
        """Charge un NIfTI et renvoie une surface isosurface (marching cubes).

        Mécanisme:
          - Charge volume via nibabel
          - Si 4D, prend la première composante
          - Calcule un iso_value:
              * variable d'environnement TRACTVIEWER_NIFTI_ISO si définie
              * sinon moyenne (min+max)/2 après exclusion des NaNs
          - Crée un pv.ImageData puis applique contour
          - Applique l'affine voxel->monde (RAS+)
          - Conversion optionnelle vers LPS si variable TRACTVIEWER_NIFTI_COORD=LPS
        """
        try:
            import nibabel as nib
        except Exception as e:  # pragma: no cover
            raise ImportError("nibabel requis pour charger des fichiers NIfTI (.nii/.nii.gz). 'pip install nibabel'") from e
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        affine = img.affine  # voxel -> monde
        dims = data.shape  # (nx, ny, nz)
        # Construire une ImageData en espace voxel (origine 0,0,0; spacing 1,1,1)
        grid = pv.ImageData()
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.dimensions = tuple([d if d > 1 else 1 for d in dims])
        # Ravel C (car ImageData en coordonnées voxel directes) -> correspond (x,y,z)
        grid.point_data["intensity"] = data.ravel(order="F")
        # Détermination iso
        iso_env = os.environ.get("TRACTVIEWER_NIFTI_ISO")
        try:
            iso_value = float(iso_env) if iso_env is not None else float((data.min() + data.max()) / 2.0)
        except ValueError:
            iso_value = float((data.min() + data.max()) / 2.0)
        if data.max() - data.min() <= 1e-6:
            warnings.warn("Volume NIfTI quasi constant: surface impossible.")
            empty = grid.extract_points(grid.points)
            # Transformer points vers monde quand même
            pts = empty.points
            pts_h = np.c_[pts, np.ones(len(pts))]
            empty.points = (pts_h @ affine.T)[:, :3]
            return empty
        # Isosurface
        try:
            surf = grid.contour([iso_value], scalars="intensity")
        except Exception as e:
            warnings.warn(f"Echec contour: {e}; fallback extract_surface (coarse).")
            surf = grid.extract_surface()
        # Appliquer affine complet sur les points (rotation + translation + anisotropies)
        pts = surf.points
        pts_h = np.c_[pts, np.ones(len(pts))]
        surf.points = (pts_h @ affine.T)[:, :3]
        # Stocker affine aplatie pour usage ultérieur (projection d'autres datasets)
        surf.field_data["nifti_affine"] = affine.astype(np.float32).ravel()
        surf.field_data["nifti_dims"] = np.array(dims, dtype=np.int32)
        # Conversion optionnelle RAS->LPS ou LPS->RAS (ici on suppose affine donne RAS+)
        target_coord = os.environ.get("TRACTVIEWER_NIFTI_COORD", "RAS").upper()
        if target_coord not in ("RAS", "LPS"):
            target_coord = "RAS"
        if target_coord == "LPS":
            surf.points[:, 0] *= -1.0  # R -> L
            surf.points[:, 1] *= -1.0  # A -> P
            surf.field_data["coord_system"] = np.array(["LPS"], dtype=object)
        else:
            surf.field_data["coord_system"] = np.array(["RAS"], dtype=object)
        surf.field_data["nifti_iso_value"] = np.array([iso_value], dtype=np.float32)
        return surf

    def add_dataset(self, data: DataInput, params: Optional[ParamDict] = None):
        ds = self._load(data)
        params = dict(params or {})
        if "name" not in params:
            params["name"] = f"ds{len(self._datasets)}"
        # Enregistrer affine de référence si NIfTI
        if self._reference_affine is None and "nifti_affine" in ds.field_data:
            flat = np.array(ds.field_data["nifti_affine"]).astype(float)
            if flat.size == 16:
                self._reference_affine = flat.reshape(4, 4)
                # Appliquer immédiatement à toute tractographie déjà chargée non transformée
                for existing_ds, _ in self._datasets:
                    self._maybe_project_tract(existing_ds)
        # Ajouter dataset
        self._datasets.append((ds, params))
        # Si l'affine existe déjà, projeter ce tract le cas échéant
        self._maybe_project_tract(ds)
        # Invalidation plotter si déjà construit
        self._plotter = None
        return self

    def _maybe_project_tract(self, ds: pv.DataSet):
        """Projette les points d'une tractographie via l'affine de référence si disponible.

        Critères heuristiques pour considérer un dataset comme tract:
          - présence de cell_data['streamline_id'] ET lines non vides.
        On évite double-application via un flag field_data['tract_transformed'].
        """
        if self._reference_affine is None:
            return
        if not isinstance(ds, pv.PolyData):  # sécurité
            return
        if "streamline_id" not in ds.cell_data:
            return
        if "tract_transformed" in ds.field_data:
            return
        try:
            pts = ds.points
            # Heuristique: détecter si les points ressemblent à des indices voxel (entiers dans bornes du volume)
            dims = None
            for ref_ds, _ in self._datasets:
                if "nifti_dims" in ref_ds.field_data:
                    dims = np.array(ref_ds.field_data["nifti_dims"]).astype(int)
                    break
            is_voxel = False
            if dims is not None and dims.size >= 3:
                if np.all(pts.min(axis=0) >= -1) and np.all(pts.max(axis=0) <= dims[:3] + 2):
                    frac = np.abs(pts - np.round(pts))
                    frac_flat = frac.ravel()
                    if frac_flat.size > 0:
                        ratio_int_like = np.mean(frac_flat < 0.05)
                        if ratio_int_like > 0.8:
                            is_voxel = True
            if is_voxel:
                pts_h = np.c_[pts, np.ones(len(pts))]
                ds.points = (pts_h @ self._reference_affine.T)[:, :3]
            # Flip optionnel pour LPS si demandé et pas encore fait
            target_coord = os.environ.get("TRACTVIEWER_NIFTI_COORD", "RAS").upper()
            if target_coord == "LPS" and ds.field_data.get("coord_system", ["RAS"]) [0] != "LPS":
                # Appliquer flips R->L, A->P
                ds.points[:, 0] *= -1.0
                ds.points[:, 1] *= -1.0
                ds.field_data["coord_system"] = np.array(["LPS"], dtype=object)
            ds.field_data["tract_transformed"] = np.array([1 if is_voxel else 0], dtype=np.int8)
        except Exception as e:
            warnings.warn(f"Projection tract échouée: {e}")

    @classmethod
    def from_paths(cls, paths: Sequence[DataInput], params_list: Optional[Sequence[ParamDict]] = None, **kwargs):
        vis = cls(**kwargs)
        params_list = params_list or [{}] * len(paths)
        for p, prm in zip(paths, params_list):
            vis.add_dataset(p, prm)
        return vis

    # ------------------------------
    # Construction de la scène
    # ------------------------------
    def _ensure_plotter(self):
        # Reconstruit si plotter absent ou déjà fermé
        if self._plotter is not None and not getattr(self._plotter, "_closed", False):
            return
        # Si un ancien plotter fermé reste référencé, on l'oublie
        if self._plotter is not None and getattr(self._plotter, "_closed", False):
            self._plotter = None
        self._plotter = pv.Plotter(off_screen=self.off_screen)
        self._plotter.set_background(self.background)
        # Recalcule (au cas où background modifié avant nouvel ensure)
        self._font_color = self._choose_font_color(self.background)
        self._scalar_bar_added = False
        for ds, prm in self._datasets:
            mesh = ds
            # Threshold si demandé
            if "threshold" in prm and prm["threshold"]:
                arr_name, (vmin, vmax) = prm["threshold"]
                if arr_name not in mesh.array_names:
                    raise ValueError(f"Array '{arr_name}' introuvable pour threshold.")
                mesh = mesh.threshold(value=(vmin, vmax), scalars=arr_name, invert=False)

            display_array = prm.get("display_array")
            if display_array and display_array not in mesh.array_names:
                raise ValueError(f"Array '{display_array}' introuvable dans dataset ({mesh.array_names}).")
            if display_array:
                # Assure que l'array est active (facilite la génération correcte de la scalar_bar)
                try:
                    mesh.set_active_scalars(display_array)
                except Exception:
                    pass

            style = prm.get("style")
            if style == "enveloppe":
                try:
                    mesh = enveloppe_minimale(mesh)
                    # On force un style surface pour l'affichage
                    prm_style_forced = "surface"
                except Exception as e:
                    warnings.warn(f"Echec enveloppe: {e}; fallback surface originale.")
                    prm_style_forced = "surface"
            else:
                prm_style_forced = style

            add_kwargs = dict(
                scalars=display_array,
                cmap=prm.get("cmap"),
                clim=prm.get("clim"),
                opacity=prm.get("opacity", 1.0),
                show_edges=prm.get("show_edges", False),
                # Une seule scalar bar globale (premier dataset éligible)
                show_scalar_bar=bool(display_array) and prm.get("scalar_bar", True) and not self._scalar_bar_added,
                name=prm.get("name"),
                color=prm.get("color"),
                style=prm_style_forced,
            )
            # Si rendu en points sans point_size défini, appliquer une valeur par défaut
            if add_kwargs.get("style") == "points" and "point_size" not in prm:
                add_kwargs["point_size"] = 5
            # Option de rendu sphérique des points
            if "render_points_as_spheres" in prm:
                add_kwargs["render_points_as_spheres"] = prm["render_points_as_spheres"]
            if add_kwargs["show_scalar_bar"]:
                sb_args = prm.get("scalar_bar_args") or {}
                sb_defaults = {
                    "title": display_array or "",
                    "n_labels": 5,
                    "fmt": "%.2f",
                    # Couleur de texte unifiée (ticks + titre)
                    "color": self._font_color,
                }
                sb_defaults.update(sb_args)
                add_kwargs["scalar_bar_args"] = sb_defaults
            for opt in ("point_size", "line_width", "ambient", "specular", "diffuse"):
                if opt in prm:
                    add_kwargs[opt] = prm[opt]
            self._plotter.add_mesh(mesh, **{k: v for k, v in add_kwargs.items() if v is not None})
            if add_kwargs.get("show_scalar_bar"):
                self._scalar_bar_added = True

        # Ajuster caméra globale
        if self._datasets:
            self._plotter.camera_position = "xy"  # orientation initiale simple

    # ------------------------------
    # Rotations caméra robustes
    # ------------------------------
    def _rotate_camera(self, azimuth_deg: float = 0.0, elevation_deg: float = 0.0):
        """
        Applique une rotation caméra en essayant d'abord les méthodes VTK,
        sinon fallback par rotation de la position autour du focal point.
        """
        cam = self._plotter.camera
        # Azimuth
        if azimuth_deg:
            if callable(getattr(cam, "Azimuth", None)):
                cam.Azimuth(azimuth_deg)
            elif callable(getattr(cam, "azimuth", None)):
                cam.azimuth(azimuth_deg)
            else:
                self._rotate_camera_fallback(axis='z', angle_deg=azimuth_deg)
        # Elevation
        if elevation_deg:
            if callable(getattr(cam, "Elevation", None)):
                cam.Elevation(elevation_deg)
            elif callable(getattr(cam, "elevation", None)):
                cam.elevation(elevation_deg)
            else:
                self._rotate_camera_fallback(axis='x', angle_deg=elevation_deg)

    def _rotate_camera_fallback(self, axis: str, angle_deg: float):
        """Fallback: rotation de la position caméra autour du focal point."""
        cam = self._plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        vec = pos - fp
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        if axis == 'z':
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
        elif axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]])
        elif axis == 'y':
            R = np.array([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]])
        else:
            return
        new_pos = fp + R.dot(vec)
        cam.position = new_pos.tolist()
        # Optionnel: pas de modification du view_up ici (préserver stabilité)

    def _apply_rotation(self, azimuth_deg: float, elevation_deg: float):
        """Rotation + rendu (indispensable off_screen pour que la caméra se propage aux captures)."""
        self._rotate_camera(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        self._plotter.render()

    # ------------------------------
    # Affichage interactif
    # ------------------------------
    def show(self, **show_kwargs):
        self._ensure_plotter()
        if self.off_screen:
            warnings.warn("off_screen=True : aucune fenêtre interactive ne s'ouvrira. Passez off_screen=False pour interaction.")
        return self._plotter.show(**show_kwargs)

    # ------------------------------
    # Capture d'écran
    # ------------------------------
    def capture_screenshot(self, output_path: Union[str, Path], transparent: bool = False):
        self._ensure_plotter()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = self._plotter.screenshot(filename=str(output_path), transparent_background=transparent)
        return img

    # ------------------------------
    # Enregistrement vidéo / GIF
    # ------------------------------
    @staticmethod
    def _ffmpeg_encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
        """Retourne True si l'encodeur ffmpeg est disponible."""
        try:
            out = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode != 0:
                return False
            # Chaque ligne des encoders commence par 2 lettres (flags) + espace + nom
            token = f" {encoder}"
            return any(line.strip().endswith(encoder) or f" {encoder} " in line for line in out.stdout.splitlines())
        except Exception:
            return False

    @staticmethod
    def _select_ffmpeg_codec(ffmpeg_bin: str, requested: str) -> Tuple[str, Optional[str]]:
        """
        (CONSERVÉ pour compat) -> retourne premier codec dispo ou fallback.
        Désormais remplacé par _codec_candidates, gardé si appelé ailleurs.
        """
        order = [
            requested,
            "libx264", "h264_nvenc", "h264_qsv", "h264_v4l2m2m", "h264_vaapi",
            "libx265", "hevc_nvenc", "hevc_qsv", "hevc_v4l2m2m", "hevc_vaapi",
            "libvpx_vp9", "libvpx_vp8",
            "mpeg4",
            "ffv1",   # sans perte si dispo
        ]
        seen = set()
        for c in order:
            if c in seen:
                continue
            seen.add(c)
            if TractViewer._ffmpeg_encoder_available(ffmpeg_bin, c):
                if c == requested:
                    return c, None
                return c, f"Codec '{requested}' indisponible -> fallback '{c}'."
        # Aucun codec de la liste : laisser ffmpeg choisir (pas de -c:v)
        return "", f"Codec '{requested}' indisponible, aucun fallback trouvé -> utilisation du codec par défaut ffmpeg."

    @staticmethod
    def _codec_candidates(ffmpeg_bin: str, requested: str, prefer_software: bool = True) -> List[str]:
        """
        Retourne une liste ordonnée de codecs à essayer.
        - Si prefer_software=True, ne met les codecs hardware (nvenc, qsv, vaapi, v4l2m2m) qu’en fin.
        - Le codec explicitement demandé reste en tête (même si HW).
        """
        hw_suffix = ("_nvenc", "_qsv", "_vaapi", "_v4l2m2m")
        # Base logiciels (ordre de préférence)
        software_set = [
            "mpeg4","libx264", "libopenh264", "libxvid",
            "libvpx_vp9", "libvpx_vp8", "ffv1", "rawvideo"
        ]
        # Encoders hardware potentiels
        hardware_set = software_set
        # hardware_set = [
        #     "h264_vaapi","h264_nvenc", "hevc_nvenc",
        #     "h264_qsv", "hevc_qsv",
        #      "hevc_vaapi",
        #     "h264_v4l2m2m", "hevc_v4l2m2m",
        # ]
        ordered = [requested] if requested else []
        if prefer_software:
            ordered += [c for c in software_set if c not in ordered]
            ordered += [c for c in hardware_set if c not in ordered]
        else:
            # Mélange requested -> hw -> software
            ordered += [c for c in hardware_set if c not in ordered]
            ordered += [c for c in software_set if c not in ordered]
        # Filtrer doublons
        uniq = []
        for c in ordered:
            if c and c not in uniq:
                uniq.append(c)
        # Garder uniquement ceux disponibles
        available = [c for c in uniq if TractViewer._ffmpeg_encoder_available(ffmpeg_bin, c)]
        # Toujours ajouter fallback sans spécifier codec (marqueur "")
        available.append("")  # "" => laisser ffmpeg décider
        return available

    def record_rotation(
        self,
        output_path: Union[str, Path],
        n_frames: int = 180,
        step: float = 2.0,
        elevation: float = 0.0,
        gif: bool = False,
        fps: int = 30,
        quality: int = 5,
        window_size: Optional[Tuple[int, int]] = None,
        supersample: int = 1,
        anti_aliasing: bool = True,
        codec: str = "mpeg4",
        bitrate: Optional[str] = None,
        crf: Optional[int] = None,
        auto_codec_fallback: bool = True,
        prefer_software: bool = True,
    ):
        """
        Effectue une rotation azimutale de la caméra et enregistre.
        gif=True => GIF via imageio.
        gif=False => MP4/AVI via:
          1) imageio-ffmpeg si dispo (flux direct)
          2) sinon binaire ffmpeg si présent
          3) sinon erreur explicite

        Paramètres qualité supplémentaires:
          window_size: (w,h) taille finale souhaitée (sans supersample). Si None, taille actuelle.
          supersample: facteur (>1) pour rendre en (w*factor, h*factor) puis encoder (améliore netteté).
          anti_aliasing: active l'anti-aliasing (si supporté).
          codec: encodeur vidéo ffmpeg/imageio (ex: libx264, mpeg4, libvpx_vp9).
          bitrate: ex '8M' (si défini, ffmpeg utilisera ce débit).
          crf: qualité cible (0-51, plus bas = meilleur) si codec type x264/x265 (ignoré si bitrate défini).
          quality: param interne imageio (0-10) — conserver élevé (8-10) pour limiter la perte.
         auto_codec_fallback: si True, sélection automatique d'un codec disponible si le demandé manque.
        """
        self._ensure_plotter()
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".gif" if gif else ".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Gestion résolution / supersampling
        original_size = getattr(self._plotter, "window_size", None)
        target_size = None
        if window_size:
            w, h = window_size
            w = int(w)
            h = int(h)
            if supersample > 1:
                target_size = (w * supersample, h * supersample)
            else:
                target_size = (w, h)
            try:
                self._plotter.window_size = target_size
            except Exception:
                target_size = None  # fallback silencieux

        if anti_aliasing:
            with contextlib.suppress(Exception):
                self._plotter.enable_anti_aliasing()

        if gif and imageio is None:
            gif = False  # fallback (mais nécessitera quand même imageio pour vidéo)

        if gif:
            frames = []
            for i in range(n_frames):
                elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                frames.append(self._plotter.screenshot(return_img=True))
            imageio.mimsave(output_path, frames, fps=fps)
            # Restauration taille
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # --- Branche vidéo ---
        if imageio is None:
            raise RuntimeError("imageio indisponible. Installez: pip install imageio imageio-ffmpeg ou ffmpeg système.")

        # 1) Essai imageio-ffmpeg
        has_imageio_ffmpeg = False
        try:
            import imageio_ffmpeg  # noqa: F401
            has_imageio_ffmpeg = True
        except Exception:
            pass

        if has_imageio_ffmpeg:
            # Encodage en flux direct
            writer = None
            try:
                writer_kwargs = dict(fps=fps, quality=quality, codec=codec)
                if bitrate:
                    writer_kwargs["bitrate"] = bitrate
                out_params = []
                if crf is not None and not bitrate:
                    out_params += ["-crf", str(crf)]
                if out_params:
                    writer_kwargs["output_params"] = out_params
                writer = imageio.get_writer(str(output_path), fps=fps, quality=quality)
                # Remplacer par writer avec kwargs (compat anciennes versions)
                with contextlib.suppress(TypeError):
                    writer.close()
                    writer = imageio.get_writer(str(output_path), **writer_kwargs)
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    writer.append_data(frame)
            finally:
                if writer is not None:
                    with contextlib.suppress(Exception):
                        writer.close()
                self._plotter.close()
                # Permettre reconstruction ultérieure (ex: vis.show())
                self._plotter = None
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 2) Fallback binaire ffmpeg (frames temporaires PNG)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            temp_dir = Path(tempfile.mkdtemp(prefix="vtkvis_frames_"))
            try:
                for i in range(n_frames):
                    elev_step = elevation if (elevation and i < n_frames / 2) else (-elevation if elevation else 0.0)
                    self._apply_rotation(azimuth_deg=step, elevation_deg=elev_step)
                    frame = self._plotter.screenshot(return_img=True)
                    imageio.imwrite(temp_dir / f"frame{i:05d}.png", frame)
                # Essais multi-codecs
                if auto_codec_fallback:
                    candidates = self._codec_candidates(ffmpeg_bin, codec, prefer_software=prefer_software)
                else:
                    candidates = [codec]
                last_err = None
                for idx, cdc in enumerate(candidates):
                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        "-framerate", str(fps),
                        "-i", str(temp_dir / "frame%05d.png"),
                    ]
                    if cdc:
                        if idx == 0 and cdc != codec:
                            warnings.warn(f"Codec '{codec}' indisponible -> fallback '{cdc}'.")
                        elif idx > 0:
                            warnings.warn(f"Tentative codec fallback '{cdc}'.")
                        cmd += ["-c:v", cdc]
                    if crf is not None and not bitrate and (cdc.startswith("libx264") or cdc.startswith("libx265") or cdc in ("libvpx_vp9", "")):
                        cmd += ["-crf", str(crf)]
                    if bitrate:
                        cmd += ["-b:v", bitrate]
                    if cdc.startswith("libx264") or cdc.startswith("libx265"):
                        cmd += ["-preset", "slow"]
                    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        last_err = None
                        break
                    last_err = f"Codec '{cdc or 'auto'}' échec:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                if last_err:
                    raise RuntimeError("ffmpeg a échoué après tous les fallbacks:\n" + last_err)
            finally:
                self._plotter.close()
                self._plotter = None
                with contextlib.suppress(Exception):
                    shutil.rmtree(temp_dir)
            if original_size is not None and target_size is not None:
                with contextlib.suppress(Exception):
                    self._plotter.window_size = original_size
            return str(output_path)

        # 3) Erreur explicite
        raise RuntimeError(
            "Aucun backend vidéo fonctionnel.\n"
            "- Installez imageio-ffmpeg: pip install imageio-ffmpeg\n"
            "ou\n"
            "- Installez ffmpeg (ex: apt-get install ffmpeg)\n"
            "Sinon utilisez gif=True pour générer un GIF."
        )

    # ------------------------------
    # Utilitaires
    # ------------------------------
    def list_arrays(self) -> Dict[str, List[str]]:
        out = {}
        for ds, prm in self._datasets:
            out[prm.get("name")] = list(ds.array_names)
        return out

    @staticmethod
    def _choose_font_color(bg) -> str:
        """
        Retourne 'black' ou 'white' selon la luminance du background.
        Implémentation robuste sans dépendre de pyvista.parse_color (absent sur certaines versions).
        Accepte:
          - noms de couleurs (si matplotlib installé)
          - code hex (#RRGGBB ou #RGB)
          - tuple/list (r,g,b[,a]) avec composantes 0-1 ou 0-255
        """
        def _norm_rgb(rgb):
            if max(rgb) > 1.0:
                return [c / 255.0 for c in rgb]
            return rgb

        # Tentative matplotlib
        try:
            import matplotlib.colors as mcolors
            try:
                r, g, b = mcolors.to_rgb(bg)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                pass
        except Exception:
            pass

        # Tuples / listes
        if isinstance(bg, (tuple, list)) and 3 <= len(bg) <= 4:
            try:
                r, g, b = _norm_rgb(list(bg)[:3])
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return "black" if lum > 0.6 else "white"
            except Exception:
                return "black"

        # Hex manuel
        if isinstance(bg, str) and bg.startswith("#"):
            h = bg.lstrip("#")
            if len(h) == 3:
                h = "".join([c*2 for c in h])
            if len(h) == 6:
                try:
                    r = int(h[0:2], 16) / 255.0
                    g = int(h[2:4], 16) / 255.0
                    b = int(h[4:6], 16) / 255.0
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    return "black" if lum > 0.6 else "white"
                except Exception:
                    return "black"

        # Fallback par défaut
        return "black"


# -------------------------------------------------
# Exemple d'utilisation (protégé par __main__)
# -------------------------------------------------
if __name__ == "__main__":
    #Set TRACTVIEWER_NIFTI_COORD=LPS in env to have nifti in LPS

    vis = TractViewer(background="black", off_screen=False)
    vis.add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/long_central_line/summed_AF_left_centroids.vtk",
        {
            "display_array": None,
            "color": "red",
            "opacity": 1.0,
            "line_width": 10,
            "scalar_bar": True,  # mappé vers show_scalar_bar
            "name": "associations_enveloppe",
            "style": "surface",
        }
    ).add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-associations_model-MCM_space-HCP_tracto.vtk",
        {
            "display_array": "point_index",
            "cmap": "viridis",
            "opacity": 0.5,
            "scalar_bar": True,  # mappé vers show_scalar_bar
            "name": "associations",
            "style": "surface",
        }
    ).add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-centroids_model-MCM_space-subject_tracto.vtk",
        {
            "display_array": "point_index",
            "cmap": "viridis",
            "point_size": 20,
            "opacity": 1.0,
            "name": "centroids",
            "style": "points",
            "render_points_as_spheres": True,
        }
    ).add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/average_anat.nii.gz",
        {
            "display_array": "intensity",
            "cmap": "gray",
            "clim": (200, 800),
            "opacity": 0.3,
            "scalar_bar": False,
            "name": "anatomy",
            "ambient": 0.6,
            "specular": 0.1,
            "diffuse": 0.8,
            "style": "surface",
        }
    )
    # .add_dataset(
    #     "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_AF_left.tck",
    #     {
    #         "display_array": "length_mm",
    #         "cmap": "plasma",
    #         "clim": (50, 150),
    #         "opacity": 0.6,
    #         "scalar_bar": True,
    #         "name": "summed_AF",
    #         "style": "surface",
    #         "ambient": 0.3,
    #         "specular": 0.2,
    #         "diffuse": 0.5,
    #     }
    # )
    
    # vis.capture_screenshot("capture.png")
    # vis.record_rotation("rotation.mp4", n_frames=240, step=1.5, quality=10)
    vis.show()  # si rendu interactif possible
    # vis.show()  # si rendu interactif possible
