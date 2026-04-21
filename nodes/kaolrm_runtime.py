"""Runtime helpers for resolving and importing KaoLRM."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
KAOLRM_ENV_VAR = "KAOLRM_ROOT"
KAOLRM_CANDIDATES = [REPO_ROOT / "third_party" / "kaolrm"]


def _is_kaolrm_root(root: Path) -> bool:
    return (root / "kaolrm").is_dir()


def _resolve_env_root() -> Path | None:
    root_str = os.environ.get(KAOLRM_ENV_VAR)
    if not root_str:
        return None
    root = Path(root_str).expanduser().resolve()
    return root if _is_kaolrm_root(root) else None


def _resolve_installed_root() -> Path | None:
    spec = importlib.util.find_spec("kaolrm")
    if spec is None:
        return None

    search_locations = list(spec.submodule_search_locations or [])
    for location in search_locations:
        package_dir = Path(location).resolve()
        root = package_dir.parent
        if _is_kaolrm_root(root):
            return root

    if spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        root = package_dir.parent
        if _is_kaolrm_root(root):
            return root
    return None


def resolve_kaolrm_root(*, required: bool = True) -> Path | None:
    env_root = _resolve_env_root()
    if env_root is not None:
        return env_root

    for root in KAOLRM_CANDIDATES:
        if _is_kaolrm_root(root):
            return root

    installed_root = _resolve_installed_root()
    if installed_root is not None:
        return installed_root

    if not required:
        return None

    roots = ", ".join(str(p) for p in KAOLRM_CANDIDATES)
    raise RuntimeError(
        "KaoLRM runtime is not available. Install the upstream 'kaolrm' package, "
        f"set {KAOLRM_ENV_VAR} to a KaoLRM checkout, or vendor it under one of: {roots}."
    )


def ensure_kaolrm_on_path(*, required: bool = True) -> Path | None:
    root = resolve_kaolrm_root(required=required)
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        log.info("Added KaoLRM source root to sys.path: %s", root)
    return root


def import_kaolrm_symbols() -> dict[str, object]:
    root = ensure_kaolrm_on_path()
    try:
        cam_utils = importlib.import_module("kaolrm.datasets.cam_utils")
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"KaoLRM import failed because dependency '{missing}' is missing. "
            "Install the upstream KaoLRM Python dependencies, then retry. "
            f"Source root: {root}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to import KaoLRM runtime from '{root}': {e}") from e

    return {
        "root": root,
        "build_camera_principle": getattr(cam_utils, "build_camera_principle"),
        "create_intrinsics": getattr(cam_utils, "create_intrinsics"),
    }
