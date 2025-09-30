# tractviewer

Petit utilitaire pour visualiser (ou capturer hors écran) des tractographies / surfaces.

## Installation

```bash
pip install git+https://github.com/nathandecaux/TractViewer.git
```

ou en version editable (pour développement) :

```bash
git clone https://github.com/nathandecaux/TractViewer.git
cd TractViewer
pip install -e .
```

## Utilisation CLI

Simple capture d'écran :

```bash
tractviewer bundle.vtk autre.tck --background black --screenshot img.png
```

Rotation vidéo :

```bash
tractviewer bundle.vtk --rotate 180 --rotation-output rot.mp4 --step 2
```

Mode interactif:
```bash
tractviewer bundle.vtk --interactive
```

### Paramètres par fichier

Chaque entrée peut être enrichie de paramètres d'affichage via la syntaxe :

```
chemin.vtk:clé=val,clé=val
```

Clés supportées actuellement :

| Clé | Effet |
|-----|-------|
| `scalar` / `scalars` | Nom du scalaire à activer pour la coloration |
| `color` | Couleur fixe (nom matplotlib, hex) |
| `opacity` | Opacité globale (0-1) |
| `cmap` | Colormap (si scalaire actif) |
| `line_width` | Épaisseur des lignes (streamlines) |
| `points` / `as_points` | Force rendu de points (glyph natif, sphères implicites) |
| `spheres` | Alias de points (rendu en sphères si supporté) |
| `point_size` | Taille des points (pixels) si rendu points |
| `point_radius` | Rayon (en unités monde) pour glyph de sphères (convertit en maillage) |

Exemple :

```bash
tractviewer bundle.vtk:scalar=FA,cmap=viridis,opacity=0.5 another.vtk:color=red,line_width=3 \
	--rotate 180 --rotation-output rot.mp4
```

### Mode hors-écran

Pour forcer le rendu headless (cluster sans X) :

```bash
export PYVISTA_OFF_SCREEN=1
tractviewer bundle.vtk --screenshot cap.png
```

### Rotation

La rotation applique un pas d'azimut fixe à la caméra avant chaque capture. Ajuster la fluidité avec `--step` et `--rotate` (nombre total de frames). Pour un GIF :

```bash
tractviewer bundle.vtk --rotate 120 --rotation-output rot.gif --step 3 --gif
```

## API

```python
from tractviewer import TractViewer
vis = TractViewer(off_screen=True)
vis.add_dataset("bundle.vtk", {"name": "bundle"})
vis.capture_screenshot("cap.png")

# Exemple avec paramètres d'affichage
vis.add_dataset("bundle.vtk", {"scalar": "FA", "cmap": "magma", "opacity": 0.5})
vis.record_rotation("rotation.mp4", n_frames=180, step=2)

# Points / sphères
vis.add_dataset(
	"points.vtk",
	{"points": True, "point_size": 8, "color": "yellow"}
)
vis.add_dataset(
	"points.vtk",
	{"point_radius": 0.5, "color": "red", "opacity": 0.6}
)
```

### Exemples API avancés

```python
from pathlib import Path
import os
from tractviewer import TractViewer

# (Optionnel) Forcer coord LPS pour aligner avec certains pipelines neuro
os.environ["TRACTVIEWER_NIFTI_COORD"] = "LPS"
# (Optionnel) Valeur d'iso manuelle pour surfaces NIfTI
# os.environ["TRACTVIEWER_NIFTI_ISO"] = "700"

vis = TractViewer(background="black", off_screen=True)

# 1. Chargement d'une anatomie (surface iso) + réglages matériaux
vis.add_dataset(
    "anat.nii.gz",
    {
        "display_array": "intensity",
        "cmap": "gray",
        "clim": (200, 800),
        "opacity": 0.25,
        "ambient": 0.6,
        "diffuse": 0.8,
        "specular": 0.1,
        "show_scalar_bar": False,
        "name": "anatomy",
        "style": "surface",
    }
)

# 2. Tractographie (vtk / tck / trk) avec colormap sur longueur + threshold
vis.add_dataset(
    "bundle_AF_left.tck",
    {
        "display_array": "length_mm",
        "cmap": "plasma",
        "clim": (40, 160),
        "opacity": 0.6,
        "threshold": ("length_mm", (50, 200)),  # filtre
        "show_scalar_bar": True,
        "name": "AF_left",
        "line_width": 2.0,
    }
)

# 3. Centroides en points rendus comme sphères
vis.add_dataset(
    "centroids.vtk",
    {
        "style": "points",
        "points_as_spheres": True,
        "point_size": 18,          # si style points "legacy"
        "color": "yellow",
        "opacity": 1.0,
        "name": "centroids",
    }
)

# 4. Même fichier tract mais rendu en points (glyph implicites) sans scalaires
vis.add_dataset(
    "bundle_AF_left.tck",
    {
        "style": "points",
        "points_as_spheres": False,
        "point_size": 4,
        "color": "#00ffcc",
        "opacity": 0.35,
        "name": "AF_points_overlay",
        "show_scalar_bar": False,
    }
)

# 5. Liste des arrays disponibles par dataset (utile pour choisir display_array)
print(vis.list_arrays())

# 6. Capture d'écran
vis.capture_screenshot("out/capture.png")

# 7. Rotation vidéo (MP4) avec qualité élevée + CRF
vis.record_rotation(
    "out/rotation.mp4",
    n_frames=240,      # nombre d'images
    step=1.5,          # incrément azimut
    elevation=0.0,     # mettre p.ex 10 pour basculer en 1ère moitié puis -10
    fps=30,
    quality=9,         # (imageio) 0-10
    codec="libx264",
    crf=18,            # (si pas de bitrate)
    supersample=2,     # rendu interne 2x puis compressé (plus net)
    window_size=(960, 720),
)

# 8. GIF
vis.record_rotation(
    "out/rotation.gif",
    n_frames=120,
    step=3,
    gif=True,
    fps=20
)

# 9. Reconstruction d’une nouvelle instance via from_paths
paths = ["anat.nii.gz", "bundle_AF_left.tck"]
params = [
    {"display_array": "intensity", "cmap": "gray", "opacity": 0.2},
    {"display_array": "length_mm", "cmap": "viridis", "opacity": 0.7},
]
vis2 = TractViewer.from_paths(paths, params, background="white", off_screen=True)
vis2.capture_screenshot("out/quick.png")

# 10. Interaction (si DISPLAY dispo)
if os.environ.get("DISPLAY"):
    TractViewer(background="white").add_dataset(
        "bundle_AF_left.tck",
        {"display_array": "length_mm", "cmap": "magma", "show_scalar_bar": True}
    ).show()
```

### Paramètres principaux (API add_dataset)

| Clé | Description |
|-----|-------------|
| display_array | Nom d'un array (point/cell) à colorer |
| cmap | Colormap matplotlib ou liste de couleurs |
| clim | (min, max) pour l'échelle |
| threshold | (array, (min,max)) applique un filtre |
| opacity | float / séquence / str (ex: 'linear') |
| show_scalar_bar | bool (affiche une seule barre globale la 1ère fois) |
| line_width / point_size | Tailles primitives |
| style | surface | wireframe | points |
| points_as_spheres | Rendu sphères (points) |
| ambient / diffuse / specular | Matériau lumière |
| name | Identifiant interne |
| color | Couleur fixe (si pas de display_array) |

### record_rotation (paramètres utiles)

| Paramètre | Effet |
|-----------|-------|
| n_frames | Nombre d'images |
| step | Incrément azimut (°) par frame |
| elevation | Bascules (±) sur première/ seconde moitié |
| gif | True => GIF sinon vidéo |
| fps | Images/sec |
| codec | libx264, libvpx_vp9, mpeg4... (fallback auto) |
| crf | Qualité (H.264/265/VP9) si pas de bitrate |
| bitrate | Débit cible '8M' (override crf) |
| supersample | Sur-échantillonnage spatial |
| window_size | Taille finale (avant supersample) |
| quality | Param imageio (0-10) |
| auto_codec_fallback | Essayez codecs alternatifs |

### Variables d'environnement supportées

| Variable | Effet |
|----------|-------|
| PYVISTA_OFF_SCREEN=1 | Rendu sans fenêtre |
| TRACTVIEWER_NIFTI_COORD=LPS | Conversion points NIfTI/tract en LPS |
| TRACTVIEWER_NIFTI_ISO=val | Force valeur iso pour surface NIfTI |
| DISPLAY (unset) | Force off_screen automatique |

### Bonnes pratiques

- Toujours vérifier list_arrays() avant de choisir un scalaire.
- Utiliser supersample=2 pour vidéos nettes + CRF bas (18-22).
- Pour clusters sans X11: exporter PYVISTA_OFF_SCREEN=1.
- Pour forcer mode LPS cohérent entre surfaces et tracto: TRACTVIEWER_NIFTI_COORD=LPS.
