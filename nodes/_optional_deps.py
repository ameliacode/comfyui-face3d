"""Optional-dependency probes — keep import errors out of node registration."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_PYTORCH3D_STATE = None


def try_import_pytorch3d():
    """Return the pytorch3d module, or None if unavailable."""
    global _PYTORCH3D_STATE
    if _PYTORCH3D_STATE is not None:
        return _PYTORCH3D_STATE if _PYTORCH3D_STATE is not False else None
    try:
        import pytorch3d  # noqa: F401
        _PYTORCH3D_STATE = pytorch3d
        return pytorch3d
    except Exception as e:
        log.info("pytorch3d unavailable (%s); FLAME render will use soft_torch fallback.", e)
        _PYTORCH3D_STATE = False
        return None
