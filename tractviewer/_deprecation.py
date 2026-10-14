"""TractViewer deprecation warnings.

`TractViewerDeprecationWarning` subclasses DeprecationWarning but is shown by
default (filter installed in `tractviewer/__init__.py`), so existing scripts
see the message without any extra configuration.
"""
from __future__ import annotations
import functools
import warnings
from typing import Optional


class TractViewerDeprecationWarning(DeprecationWarning):
    """Deprecated feature: still works, but should be migrated."""


def warn_deprecated(old: str, new: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"Deprecated: {old}. Use {new} instead.",
        TractViewerDeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(new: str, old: Optional[str] = None):
    """Decorator: the function keeps working but emits a warning."""
    def decorator(func):
        name = old or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(name, new)
            return func(*args, **kwargs)
        return wrapper
    return decorator
