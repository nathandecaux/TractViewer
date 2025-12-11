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
from PIL import Image
from tractviewer.enveloppe import enveloppe_minimale  # ajout
# Ajout: imports debug
import sys
import traceback
import faulthandler  # ajout
import gc

try:
    import imageio.v2 as imageio  # GIF option
except Exception:  # pragma: no cover
    imageio = None


DataInput = Union[str, Path, pv.DataSet]
ParamDict = Dict[str, object]

os.environ["TRACTVIEWER_NIFTI_COORD"] = "LPS"
os.environ["TRACTVIEWER_NIFTI_ISO"] = "0.2"
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
      - points_as_spheres: bool (utile si style='points')
      - name: nom interne (sinon auto)
    Paramètres globaux (constructeur):
      - background: couleur du fond (default "white")
      - off_screen: bool (pour rendu sans interface, ex: serveur)
    """
    def __init__(self, background: str = "white", off_screen: bool = False,window_size: Optional[Tuple[int,int]] = None):
        self._datasets: List[Tuple[pv.DataSet, ParamDict]] = []
        self.background = background
        self.off_screen = off_screen
        self._debug: bool = True
        self._plotter: Optional[pv.Plotter] = None
        self._scalar_bar_added = False
        self._font_color = self._choose_font_color(self.background)
        self._reference_affine: Optional[np.ndarray] = None  # 4x4 issue du premier NIfTI
        self.window_size: Optional[Tuple[int,int]] = window_size
        # Backend VTK détecté (classe du RenderWindow)
        self._vtk_backend_class: Optional[str] = None
        # Mode debug (1 par défaut, mettre TRACTVIEWER_DEBUG=0 pour désactiver)
        self._debug: bool = os.environ.get("TRACTVIEWER_DEBUG", "1") != "0"
        # Installer faulthandler / excepthook en debug
        if self._debug:
            try:
                faulthandler.enable()
            except Exception:
                pass
            # Aide au debug du chargement des plugins Qt
            os.environ.setdefault("QT_DEBUG_PLUGINS", "1")
        # Auto bascule headless si pas de DISPLAY
        if not self.off_screen and not os.environ.get("DISPLAY"):
            warnings.warn("Aucun DISPLAY détecté -> passage en mode off_screen.")
            self.off_screen = True
        # Détection backend VTK (OSMesa/EGL => pas d'interactif)
        backend_name, backend_is_headless = self._detect_vtk_backend()
        self._vtk_backend_class = backend_name
        if not self.off_screen and backend_is_headless:
            warnings.warn(f"Backend VTK '{backend_name}' détecté (OSMesa/EGL) -> passage en mode off_screen (pas d'affichage interactif possible).")
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
          - .png : image PNG -> ImageData avec couleurs RGB comme scalaires
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
        if suffix == ".png":
            return TractViewer._load_png_as_vtk(path)
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

    @staticmethod
    def _write_gif_with_ffmpeg(
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: int = 10,
        scale_width = None,
        optimize: bool = True
    ) -> str:
        """
        Write frames to an infinite loop GIF using ffmpeg with palette optimization.
        
        Args:
            frames: List of numpy arrays (H, W, 3) representing RGB frames
            output_path: Output path for the GIF
            fps: Frames per second
            scale_width: Width for scaling (height computed to preserve aspect ratio)
            optimize: Use palette optimization for better quality/size
        
        Returns:
            str: Path to the created GIF
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found. Install it: apt-get install ffmpeg")
        print('USING FFMPEG BINARY:', ffmpeg_bin)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if scale_width is None:
            scale_width = frames[0].shape[1]

        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="tractviewer_gif_"))
        
        try:
            # Save frames as PNG
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame{i:05d}.png"
                if imageio is not None:
                    imageio.imwrite(frame_path, frame)
                else:
                    from PIL import Image
                    Image.fromarray(frame).save(frame_path)
            
            if optimize:
                # Two-pass approach with palette generation for better quality
                palette_path = temp_dir / "palette.png"
                
                # Generate palette
                palette_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-framerate", str(fps),
                    "-i", str(temp_dir / "frame%05d.png"),
                    "-vf", f"fps={fps},scale={scale_width}:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=diff",
                    str(palette_path)
                ]
                
                proc = subprocess.run(palette_cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"Palette generation failed:\n{proc.stderr}")
                
                # Create GIF with palette
                gif_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-framerate", str(fps),
                    "-i", str(temp_dir / "frame%05d.png"),
                    "-i", str(palette_path),
                    "-filter_complex", f"fps={fps},scale={scale_width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0",  # 0 = infinite loop
                    str(output_path)
                ]
            else:
                # Simple one-pass approach
                gif_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-framerate", str(fps),
                    "-i", str(temp_dir / "frame%05d.png"),
                    "-vf", f"fps={fps},scale={scale_width}:-1:flags=lanczos",
                    "-loop", "0",  # 0 = infinite loop
                    str(output_path)
                ]
            
            proc = subprocess.run(gif_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"GIF creation failed:\n{proc.stderr}")
            
            return str(output_path)
        
        finally:
            # Cleanup temporary directory
            with contextlib.suppress(Exception):
                shutil.rmtree(temp_dir)

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
            surf = grid.contour([iso_value], scalars="intensity",method="marching_cubes")
            print(f"Isosurface extraite (marching_cubes) à {iso_value:.3f} dans {path.name}")
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

    @staticmethod
    def _load_png_as_vtk(path: Path) -> pv.ImageData:
        """Charge une image PNG et la convertit en VTK ImageData.
        
        Les couleurs RGB sont stockées comme scalaires 'rgb_colors'.
        Crée aussi des arrays séparés 'red', 'green', 'blue' pour chaque canal.
        """
        try:
            from PIL import Image
        except ImportError:
            try:
                import imageio.v2 as imageio
            except ImportError:
                raise ImportError("PIL (Pillow) ou imageio requis pour charger des images PNG. 'pip install Pillow' ou 'pip install imageio'")
        
        # Charger l'image
        try:
            img = Image.open(str(path))
            # Garder le mode original pour détecter la transparence
            has_alpha = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            # Convertir en RGBA pour gérer la transparence, puis extraire RGB
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_array = np.array(img)
        except Exception:
            # Fallback avec imageio si PIL échoue
            img_array = imageio.imread(str(path))
            if img_array.ndim == 2:
                # Image en niveaux de gris -> convertir en RGBA
                img_array = np.stack([img_array] * 3 + [np.full_like(img_array, 255)], axis=-1)
                has_alpha = False
            elif img_array.shape[2] == 3:
                # RGB -> ajouter canal alpha
                alpha = np.full(img_array.shape[:2], 255, dtype=img_array.dtype)
                img_array = np.concatenate([img_array, alpha[..., None]], axis=-1)
                has_alpha = False
            else:
                has_alpha = True
        
        height, width = img_array.shape[:2]
        
        # Créer ImageData VTK
        grid = pv.ImageData()
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.dimensions = (width, height, 1)
        
        # Aplatir l'image en gardant l'ordre (x,y) compatible VTK
        # VTK utilise un ordre différent, on transpose et ravel
        img_transposed = np.transpose(img_array, (1, 0, 2))  # (width, height, 4)
        
        # Séparer RGBA
        rgba_flat = img_transposed.reshape(-1, 4)
        rgb_flat = rgba_flat[:, :3]
        alpha_flat = rgba_flat[:, 3]
        
        # Marquer les pixels transparents (alpha < seuil)
        transparency_threshold = 10  # pixels avec alpha < 10 considérés transparents
        is_transparent = alpha_flat < transparency_threshold
        
        # Couleurs RGB combinées (pour affichage principal)
        # Méthode 1: utiliser la luminance
        luminance = 0.299 * rgb_flat[:, 0] + 0.587 * rgb_flat[:, 1] + 0.114 * rgb_flat[:, 2]
        grid.point_data["luminance"] = luminance.astype(np.float32)
        
        # Méthode 2: encoder RGB en un seul entier pour préserver les couleurs
        rgb_encoded = (rgb_flat[:, 0].astype(np.uint32) << 16) + \
                      (rgb_flat[:, 1].astype(np.uint32) << 8) + \
                      rgb_flat[:, 2].astype(np.uint32)
        grid.point_data["rgb_encoded"] = rgb_encoded.astype(np.float32)
        
        # Canaux séparés
        grid.point_data["red"] = rgb_flat[:, 0].astype(np.float32)
        grid.point_data["green"] = rgb_flat[:, 1].astype(np.float32)
        grid.point_data["blue"] = rgb_flat[:, 2].astype(np.float32)
        grid.point_data["alpha"] = alpha_flat.astype(np.float32)
        
        # Marquer les pixels transparents
        grid.point_data["is_transparent"] = is_transparent.astype(np.uint8)
        
        # Métadonnées
        grid.field_data["image_width"] = np.array([width], dtype=np.int32)
        grid.field_data["image_height"] = np.array([height], dtype=np.int32)
        grid.field_data["has_transparency"] = np.array([has_alpha], dtype=np.bool_)
        grid.field_data["source_file"] = np.array([str(path)], dtype=object)
        
        return grid

    def add(self, data: DataInput, params: Optional[ParamDict] = None, **kwargs):
        ds = self._load(data)
        params = dict(params or {})
        params.update(kwargs)  # Merge params with additional kwargs
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

    def rm (self, name: str):
        """Supprime un dataset par son nom."""
        initial_count = len(self._datasets)
        
        removed_datasets = [ds for ds, prm in self._datasets if prm.get("name") == name]
        if not removed_datasets:
            return self
        
        # Filtrer la liste pour retirer les datasets correspondants
        self._datasets = [(ds, prm) for ds, prm in self._datasets if prm.get("name") != name]

        # Invalider le plotter si des datasets ont été supprimés
        if len(self._datasets) < initial_count:
            self._close_plotter()
        
        for ds in removed_datasets:
            self._release_dataset_resources(ds)
        
        gc.collect()
        return self

    @staticmethod
    def _release_dataset_resources(ds: pv.DataSet):
        """Libère explicitement les gros buffers PyVista/VTK associés à un dataset."""
        with contextlib.suppress(Exception):
            ds.clear_data()
        with contextlib.suppress(Exception):
            if hasattr(ds, "points"):
                ds.points = np.empty((0, 3), dtype=np.float32)
        for attr in ("lines", "faces", "polys", "strips"):
            with contextlib.suppress(Exception):
                if hasattr(ds, attr):
                    setattr(ds, attr, np.empty((0,), dtype=np.int32))
        with contextlib.suppress(Exception):
            ds.deep_clean()

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
            vis.add(p, prm)
        return vis

    # ------------------------------
    # Construction de la scène
    # ------------------------------
    def _close_plotter(self):
        """Ferme explicitement le plotter PyVista pour éviter les fuites mémoire."""
        if self._plotter is None:
            return
        with contextlib.suppress(Exception):
            self._plotter.clear()
        with contextlib.suppress(Exception):
            self._plotter.close()
        self._plotter = None

    def _ensure_plotter(self):
        # Reconstruit si plotter absent ou déjà fermé
        if self._plotter is not None and not getattr(self._plotter, "_closed", False):
            return
        # Si un ancien plotter fermé reste référencé, on l'oublie
        if self._plotter is not None and getattr(self._plotter, "_closed", False):
            self._plotter = None
        self._plotter = pv.Plotter(off_screen=self.off_screen)
        self._plotter.set_background(self.background)
        # Appliquer une taille de fenêtre si fournie
        if self.window_size:
            with contextlib.suppress(Exception):
                self._plotter.window_size = tuple(map(int, self.window_size))
        # Recalcule (au cas où background modifié avant nouvel ensure)
        self._font_color = self._choose_font_color(self.background)
        # Remplir la scène avec les datasets
        try:
            self._populate_plotter()
        except Exception:
            self._warn_debug("Erreur lors de la population du plotter.", exc=sys.exc_info(), extra=self._debug_env_info())
            raise
        # Ajuster caméra globale
        if self._datasets:
            self._plotter.camera_position = "xy"  # orientation initiale simple

    # Ajout: factorisation de l'ajout de meshes
    def _populate_plotter(self):
        self._scalar_bar_added = False
        for idx, (ds, prm) in enumerate(self._datasets):
            try:
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

                # Gestion spéciale pour les couleurs RGB encodées
                if display_array == "rgb_encoded" and "is_transparent" in mesh.array_names:
                    mesh = mesh.threshold(value=0.5, scalars="is_transparent", invert=True)

                if display_array:
                    with contextlib.suppress(Exception):
                        mesh.set_active_scalars(display_array)

                style = prm.get("style")
                if style == "enveloppe":
                    try:
                        mesh = enveloppe_minimale(mesh)
                        prm_style_forced = "surface"
                    except Exception as e:
                        self._warn_debug(f"Echec enveloppe (dataset {idx}, name={prm.get('name')}).", exc=sys.exc_info())
                        prm_style_forced = "surface"
                else:
                    prm_style_forced = style

                if display_array == "rgb_encoded":
                    rgb_values = mesh.point_data[display_array].astype(np.uint32)
                    red = (rgb_values >> 16) & 0xFF
                    green = (rgb_values >> 8) & 0xFF
                    blue = rgb_values & 0xFF
                    colors = np.column_stack([red, green, blue]).astype(np.uint8)
                    mesh.point_data["RGB"] = colors
                    add_kwargs = dict(
                        scalars="RGB",
                        rgb=True,
                        opacity=prm.get("opacity", 1.0),
                        show_edges=prm.get("show_edges", False),
                        show_scalar_bar=False,
                        name=prm.get("name"),
                        style=prm_style_forced,
                    )
                else:
                    add_kwargs = dict(
                        scalars=display_array,
                        cmap=prm.get("cmap"),
                        clim=prm.get("clim"),
                        opacity=prm.get("opacity", 1.0),
                        show_edges=prm.get("show_edges", False),
                        show_scalar_bar=bool(display_array) and prm.get("scalar_bar", True) and not self._scalar_bar_added,
                        name=prm.get("name"),
                        color=prm.get("color"),
                        style=prm_style_forced,
                    )

                if add_kwargs.get("style") == "points" and "point_size" not in prm:
                    add_kwargs["point_size"] = 5
                if "points_as_spheres" in prm:
                    add_kwargs["points_as_spheres"] = prm["points_as_spheres"]
                if add_kwargs.get("show_scalar_bar"):
                    sb_args = prm.get("scalar_bar_args") or {}
                    sb_defaults = {
                        "title": display_array or "",
                        "n_labels": 5,
                        "fmt": "%.2f",
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
            except Exception:
                self._warn_debug(
                    f"Échec d'ajout du dataset index={idx}, name={prm.get('name')}.",
                    exc=sys.exc_info(),
                    extra=f"display_array={prm.get('display_array')}, style={prm.get('style')}, arrays={list(ds.array_names)}"
                )
                # Continuer avec les autres datasets
                continue

    # Ajout: reconstruction rapide des acteurs (sans fermer la fenêtre)
    def _rebuild_actors(self):
        if self._plotter is None:
            return
        # Sauvegarde de la caméra
        try:
            cam = getattr(self._plotter, "camera", None)
            cam_state = None
            if cam is not None:
                cam_state = (tuple(cam.position), tuple(cam.focal_point), tuple(cam.view_up))
            with contextlib.suppress(Exception):
                self._plotter.clear()
            self._populate_plotter()
            if cam_state is not None:
                with contextlib.suppress(Exception):
                    self._plotter.camera.position = list(cam_state[0])
                    self._plotter.camera.focal_point = list(cam_state[1])
                    self._plotter.camera.view_up = list(cam_state[2])
            with contextlib.suppress(Exception):
                self._plotter.render()
        except Exception:
            self._warn_debug("Erreur _rebuild_actors.", exc=sys.exc_info())
        else:
            self._log_opengl_info(where="rebuild")

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

    def _rotate_camera_fallback(self, axis='z', angle_deg=0.0):
        """Rotation caméra fallback par manipulation directe de position."""
        if not angle_deg:
            return
        cam = self._plotter.camera
        pos = np.array(cam.position)
        focal = np.array(cam.focal_point)
        # Vecteur caméra -> focal
        vec = focal - pos
        # Matrice de rotation selon l'axe
        angle_rad = math.radians(angle_deg)
        if axis.lower() == 'x':
            rot_matrix = np.array([
                [1, 0, 0],
                [0, math.cos(angle_rad), -math.sin(angle_rad)],
                [0, math.sin(angle_rad), math.cos(angle_rad)]
            ])
        elif axis.lower() == 'y':
            rot_matrix = np.array([
                [math.cos(angle_rad), 0, math.sin(angle_rad)],
                [0, 1, 0],
                [-math.sin(angle_rad), 0, math.cos(angle_rad)]
            ])
        else:  # axis == 'z'
            rot_matrix = np.array([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        # Appliquer rotation
        new_vec = rot_matrix @ vec
        cam.position = focal - new_vec

    def rotate_mesh(self, rotation_x: float = 0.0, rotation_y: float = 0.0, rotation_z: float = 0.0):
        """
        Applique une rotation aux meshes de tous les datasets.
        
        Args:
            rotation_x: rotation autour de l'axe X en degrés
            rotation_y: rotation autour de l'axe Y en degrés  
            rotation_z: rotation autour de l'axe Z en degrés
        """
        if not any([rotation_x, rotation_y, rotation_z]):
            return
            
        for ds, prm in self._datasets:
            self._apply_mesh_rotation(ds, rotation_x, rotation_y, rotation_z)
        
        # Invalider le plotter pour forcer la reconstruction
        self._plotter = None

    def _apply_mesh_rotation(self, dataset: pv.DataSet, rx: float, ry: float, rz: float):
        """Applique les rotations successives X, Y, Z au dataset."""
        points = dataset.points.copy()
        
        # Rotation X
        if rx != 0.0:
            angle_rad = math.radians(rx)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot_x = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
            points = points @ rot_x.T
        
        # Rotation Y
        if ry != 0.0:
            angle_rad = math.radians(ry)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot_y = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ rot_y.T
        
        # Rotation Z
        if rz != 0.0:
            angle_rad = math.radians(rz)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot_z = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            points = points @ rot_z.T
        
        dataset.points = points

    def _backup_mesh_state(self) -> List[np.ndarray]:
        """Sauvegarde l'état actuel des points de tous les datasets."""
        return [ds.points.copy() for ds, _ in self._datasets]

    def _restore_mesh_state(self, backup: List[np.ndarray]):
        """Restaure l'état des points depuis une sauvegarde."""
        for (ds, _), points in zip(self._datasets, backup):
            ds.points = points

    def _apply_rotation(self, azimuth_deg: float, elevation_deg: float):
        """Rotation + rendu (indispensable off_screen pour que la caméra se propage aux captures)."""
        self._rotate_camera(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        self._plotter.render()
    # ------------------------------
    # Affichage interactif
    # ------------------------------
    def show(self, **show_kwargs):
        # Log de contexte
        if self._debug:
            self._warn_debug("Appel show()", extra=self._debug_env_info())
        # Si off_screen: avertissement et fallback inchangé
        if self.off_screen:
            try:
                self._ensure_plotter()
            except Exception:
                self._warn_debug("Échec _ensure_plotter() en off_screen.", exc=sys.exc_info())
                raise
            reason = ""
            if self._vtk_backend_class:
                reason = f" (backend: {self._vtk_backend_class})"
            warnings.warn("off_screen=True : aucune fenêtre interactive ne s'ouvrira."
                          f"{reason} Utilisez capture_screenshot() ou record_rotation(), ou installez un VTK avec support écran.")
            return self._plotter.show(**show_kwargs, interactive=True)

        # Utiliser pv.Plotter pour un affichage interactif dans le terminal
        try:
            self._warn_debug("Création Plotter interactif...")
            self._ensure_plotter()
            # Forcer le rendu avant l'affichage
            self._plotter.render()
            self._warn_debug("Plotter interactif prêt à être affiché.")
            self._plotter.show(**show_kwargs, interactive=True)
        except Exception:
            self._warn_debug("Échec création/initialisation Plotter interactif.", exc=sys.exc_info(), extra=self._debug_env_info())
            raise

    # ------------------------------
    # Capture d'écran
    # ------------------------------
    def capture_screenshot(self, 
                         output_path: Union[str, Path], 
                         transparent: bool = False,
                         rotation_x: float = 0.0,
                         rotation_y: float = 0.0, 
                         rotation_z: float = 0.0):
        """
        Capture une image avec rotation optionnelle du mesh.
        
        Args:
            output_path: chemin de sortie
            transparent: fond transparent
            rotation_x/y/z: rotations du mesh en degrés avant capture
        """
        # Sauvegarder l'état original
        backup = None
        if any([rotation_x, rotation_y, rotation_z]):
            backup = self._backup_mesh_state()
            self.rotate_mesh(rotation_x, rotation_y, rotation_z)
        
        try:
            self._ensure_plotter()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img = self._plotter.screenshot(filename=str(output_path), transparent_background=transparent)
            return img
        finally:
            # Restaurer l'état original
            if backup is not None:
                self._restore_mesh_state(backup)
                self._plotter = None  # Forcer reconstruction

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
        step: float = None,
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
        off_screen: Optional[bool] = True,
        rotation_x: float = 0.0,
        rotation_y: float = 0.0,
        rotation_z: float = 0.0,
    ):
        """
        Effectue une rotation azimutale de la caméra et enregistre avec rotation optionnelle du mesh.
        
        Args supplémentaires:
            rotation_x/y/z: rotations du mesh en degrés appliquées avant l'enregistrement
        """
        # Sauvegarder l'état original et appliquer rotation mesh si demandée
        backup = None
        if any([rotation_x, rotation_y, rotation_z]):
            backup = self._backup_mesh_state()
            self.rotate_mesh(rotation_x, rotation_y, rotation_z)

        try:
            if off_screen:
                self.off_screen = True

            self._ensure_plotter()
            output_path = Path(output_path)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".gif" if gif else ".mp4")
            else:
                gif = output_path.suffix.lower() == ".gif"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Gestion résolution / supersampling
            original_size = getattr(self._plotter, "window_size", None)
            target_size = None

            #If step is None, compute it to do a full 360 rotation
            if step is None:
                step = 360.0 / n_frames

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
                
                # Use ffmpeg for proper infinite loop GIF
                try:
                    self._write_gif_with_ffmpeg(
                        frames, 
                        output_path, 
                        fps=fps,
                        optimize=True
                    )
                    print('GIF created using ffmpeg.')
                except Exception as e:
                    print('Using imageio fallback for GIF creation.')
                    warnings.warn(f"ffmpeg GIF creation failed: {e}. Falling back to imageio.")
                    # Fallback to imageio (though loop parameter may not work)
                    imageio.mimsave(str(output_path), frames, format='GIF', duration=1000/fps, loop=0)
                
                if original_size is not None and target_size is not None:
                    with contextlib.suppress(Exception):
                        self._plotter.window_size = original_size
                self._close_plotter()
                return str(output_path)

            print("Not GIF, proceeding with video encoding.")
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
                    self._close_plotter()
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
                finally:
                    self._close_plotter()
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
        
        finally:
            # Restaurer l'état original des meshes
            if backup is not None:
                self._restore_mesh_state(backup)
            self._close_plotter()

    # ------------------------------
    # Utilitaires
    # ------------------------------
    def list_arrays(self) -> Dict[str, List[str]]:
        out = {}
        for ds, prm in self._datasets:
            out[prm.get("name")] = list(ds.array_names)
        return out

    @staticmethod
    def _detect_vtk_backend() -> Tuple[str, bool]:
        """
        Retourne (class_name, is_headless). is_headless=True pour OSMesa/EGL.
        """
        try:
            import vtk  # import local pour éviter coût si inutilisé
            rw = vtk.vtkRenderWindow()
            cls = (rw.GetClassName() or "").lower()
            # Exemples possibles:
            # - 'vtkxopenglrenderwindow' (X11)
            # - 'vtkwglopenglrenderwindow' (Windows)
            # - 'vtkcocoaopenglrenderwindow' (macOS)
            # - 'vtkosopenglrenderwindow' (OSMesa)
            # - 'vtkeglopenglrenderwindow' (EGL)
            if "osopengl" in cls or "osmesa" in cls:
                return (rw.GetClassName(), True)
            if "egl" in cls:
                return (rw.GetClassName(), True)
            # Backends avec affichage
            if any(k in cls for k in ("xopengl", "wglopengl", "cocoaopengl")):
                return (rw.GetClassName(), False)
            # Inconnu: ne pas forcer
            return (rw.GetClassName() or "unknown", False)
        except Exception:
            return ("unknown", False)

    @staticmethod
    def _choose_font_color(bg) -> str:
        """
        Retourne 'black' ou 'white' selon la luminance du background.
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