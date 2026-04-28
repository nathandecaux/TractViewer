# tractviewer

Utilitaire simple pour visualiser et capturer des tractographies, surfaces et volumes.

Formats pris en charge:
- tracto: `.tck`, `.trk`
- maillages/VTK: formats lus par PyVista/VTK (`.vtk`, `.vtp`, etc.)
- NIfTI: `.nii`, `.nii.gz`
- images: `.png`

![Exemple de rendu](https://raw.githubusercontent.com/nathandecaux/TractViewer/main/01_CC2.gif)

## Installation

```bash
pip install git+https://github.com/nathandecaux/TractViewer.git
```

Développement:

```bash
git clone https://github.com/nathandecaux/TractViewer.git
cd TractViewer
pip install -e .
```

## CLI

### Exemples rapides

Capture:

```bash
tractviewer bundle.tck --background black --screenshot out.png
```

Rotation vidéo:

```bash
tractviewer bundle.tck --rotate 180 --rotation-output rot.mp4 --step 2
```

Rotation GIF:

```bash
tractviewer bundle.tck --rotate 120 --rotation-output rot.gif --gif
```

Exemple VTK (comme le GIF):

```bash
tractviewer bundle.vtk:display_array=affected,cmap=viridis,line_width=2 --background black --screenshot out.png
```

Mode interactif:

```bash
tractviewer bundle.tck --interactive
```

### Paramètres par fichier

Syntaxe:

```text
chemin:cle=val,cle=val
```

Exemple:

```bash
tractviewer \
  anat.nii.gz:display_array=intensity,cmap=gray,clim=(200,800),opacity=0.25 \
  bundle.tck:display_array=length_mm,cmap=plasma,line_width=2,threshold=(length_mm,50,200)
```

Types auto supportés côté CLI:
- bool: `true/false`, `yes/no`, `on/off`
- `None`/`null`
- `int`, `float`
- tuples: `(a,b,c)`

Raccourcis utiles:
- `clim=(min,max)` ou `clim=min:max`
- `threshold=(array,min,max)` ou `threshold=array;min;max`
- cast de l'array affiché: `type=int` ou `type=float`

### Options CLI principales

- `--background` couleur de fond
- `--off-screen` force le rendu headless
- `--interactive` force l'affichage interactif
- `--screenshot` capture PNG
- `--rotate` nombre de frames de rotation
- `--rotation-output` sortie vidéo/GIF
- `--gif` force le format GIF
- `--step` pas d'azimut entre frames
- `--window-size` taille `WxH` (ex: `1280x720`)
- `--rotation-x`, `--rotation-y`, `--rotation-z` rotation des meshes avant capture/export
- `--no-marching-cubes` charge un NIfTI en volume brut (pas d'isosurface)
- `--smooth` lissage appliqué après marching cubes
- `--color` couleur fixe par défaut (si non définie par fichier)

## API

```python
from tractviewer import TractViewer

vis = TractViewer(background="black", off_screen=True, window_size=(1280, 720))

vis.add(
    "anat.nii.gz",
    {
        "display_array": "intensity",
        "cmap": "gray",
        "clim": (200, 800),
        "opacity": 0.2,
        "name": "anat",
    },
)

vis.add(
    "bundle.tck",
    {
        "display_array": "length_mm",
        "cmap": "viridis",
        "line_width": 2,
        "threshold": ("length_mm", (50, 200)),
        "name": "bundle",
    },
)

print(vis.list_arrays())
vis.capture_screenshot("capture.png")
vis.record_rotation("rotation.mp4", n_frames=180, step=2, fps=30)
```

Exemple API VTK (comme le GIF):

```python
from tractviewer import TractViewer

vis = TractViewer(background="black", off_screen=True)
vis.add("bundle.vtk", {"display_array": "affected", "cmap": "viridis", "line_width": 2})
vis.capture_screenshot("out.png")
```

### Paramètres importants de `add(...)`

- `display_array`, `cmap`, `clim`
- `threshold=(array, (min, max))`
- `opacity`, `color`
- `style`: `surface`, `wireframe`, `points`, `enveloppe`
- `points_as_spheres`, `point_size`, `line_width`
- `ambient`, `diffuse`, `specular`
- `contour` (`True` ou couleur) et `contour_width`
- `categorical_legend`, `categorical_max_values`
- `scalar_bar` (barre globale affichée une seule fois)
- `name`

### Méthodes utiles

- `add(...)`
- `edit(name, ...)`
- `rm(name)`
- `show()`
- `capture_screenshot(...)`
- `record_rotation(...)`
- `list_arrays()`
- `TractViewer.from_paths(...)`

## Variables d'environnement

- `PYVISTA_OFF_SCREEN=1`: rendu headless
- `TRACTVIEWER_NIFTI_COORD=LPS` ou `RAS`: système de coordonnées cible
- `TRACTVIEWER_NIFTI_ISO=<valeur>`: isovaleur NIfTI
- `TRACTVIEWER_DEBUG=0|1`: logs debug

## Notes pratiques

- Sans `DISPLAY`, le mode off-screen est activé automatiquement.
- Pour export vidéo, installer `ffmpeg` est recommandé.
- Pour choisir un scalaire valide, utiliser `list_arrays()`.