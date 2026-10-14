import warnings

from ._deprecation import TractViewerDeprecationWarning
from .core import TractViewer

# TractViewer deprecations are shown by default (once per calling line).
warnings.filterwarnings("default", category=TractViewerDeprecationWarning)

__all__ = ["TractViewer", "TractViewerDeprecationWarning"]
