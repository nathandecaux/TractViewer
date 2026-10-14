"""Deprecated module: use tractviewer.envelope (English names)."""
from __future__ import annotations

from tractviewer._deprecation import deprecated
from tractviewer.envelope import minimal_envelope, tractogram_envelope

enveloppe_minimale = deprecated("tractviewer.envelope.minimal_envelope",
                                old="tractviewer.enveloppe.enveloppe_minimale")(minimal_envelope)
enveloppe_tractogram = deprecated("tractviewer.envelope.tractogram_envelope",
                                  old="tractviewer.enveloppe.enveloppe_tractogram")(tractogram_envelope)

__all__ = ["enveloppe_minimale", "enveloppe_tractogram"]
