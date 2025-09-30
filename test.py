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
from TractViewer import TractViewer

def _print_env():
    print("=== ENV DEBUG ===")
    print("DISPLAY=", os.environ.get("DISPLAY"))
    print("PYVISTA_OFF_SCREEN=", os.environ.get("PYVISTA_OFF_SCREEN"))
    print("TRACTVIEWER_QT=", os.environ.get("TRACTVIEWER_QT"))
    print("TRACTVIEWER_DEBUG=", os.environ.get("TRACTVIEWER_DEBUG"))
    try:
        import pyvista as _pv
        print("pyvista version=", _pv.__version__, "OFF_SCREEN=", getattr(_pv, 'OFF_SCREEN', None))
    except Exception as e:
        print("pyvista import error:", e)
    print("==================")

if __name__ == "__main__":
    import pyvistaqt as pvqt  # type: ignore

    # Forcer debug par défaut
    os.environ.setdefault("TRACTVIEWER_DEBUG", "1")
    os.environ.setdefault("TRACTVIEWER_NIFTI_COORD", "LPS")
    _print_env()

    use_qt = os.environ.get("TRACTVIEWER_QT", "").lower() in {"1","true","yes"}
    vis = TractViewer(background="black", off_screen=False)

    # Chargement datasets
    assoc_path = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-associations_model-MCM_space-HCP_tracto.vtk"
    cent_path = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_24pts/sub-01001/tracto/sub-01001_bundle-AFleft_desc-centroids_model-MCM_space-subject_tracto.vtk"
    if not Path(assoc_path).exists():
        print("Fichier introuvable:", assoc_path)
    if not Path(cent_path).exists():
        print("Fichier introuvable:", cent_path)

    vis.add_dataset(
        assoc_path,
        {
            "display_array": "point_index",
            "cmap": "viridis",
            "threshold": ("point_index", (0, 24)),
            "opacity": 0.9,
            "scalar_bar": True,
            "name": "associations",
        }
    ).add_dataset(
        cent_path,
        {
            "display_array": None,
            "color": "red",
            "point_size": 20,
            "opacity": 1.0,
            "name": "centroids",
            "style": "points",
            "points_as_spheres": True,
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
    ).add_dataset(
        "/home/ndecaux/NAS_EMPENN/share/projects/HCP105_Zenodo_NewTrkFormat/inGroupe1Space/Atlas/summed_AF_left.trk",
        {
            "display_array": "length_mm",
            "cmap": "plasma",
            "clim": (50, 150),
            "opacity": 0.6,
            "scalar_bar": True,
            "name": "summed_AF",
            "style": "surface",
            "ambient": 0.3,
            "specular": 0.2,
            "diffuse": 0.5,
        }
    )

    vis.show()
    #vis.record_rotation('rotation.mp4')

    #Show with QT panel 
    # Fin
    # print("Fin script.")