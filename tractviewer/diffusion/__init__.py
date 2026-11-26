"""Diffusion models (DTI, ODF, Anima MCM): loading and glyph data for the 2D views.

  - models: files -> pv.ImageData (scalar map + model arrays in the world RAS frame)
  - glyphs: glyph payload sent to the vtk.js page
  - io: NRRD and Anima .mcm/.mcmx readers
  - sh: MRtrix3 spherical harmonic basis, icospheres
"""
from tractviewer.diffusion.models import load_diffusion_nifti, load_mcm, MODELS  # noqa: F401


def is_diffusion(ds) -> bool:
    return "diffusion_model" in ds.field_data
