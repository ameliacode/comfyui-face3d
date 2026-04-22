"""Runtime helpers for resolving and importing FreeUV.

Deliberate divergence from the SMIRK/KaoLRM `spec_from_file_location` pattern:
FreeUV's `detail_encoder/` package uses relative imports (`._clip`,
`.attention_processor`, `.resampler`), which only resolve when the vendor root
is on `sys.path`. We inject the path here instead of loading submodules in
isolation.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEUV_ENV_VAR = "FREEUV_ROOT"
FREEUV_CANDIDATES = [REPO_ROOT / "third_party" / "freeuv"]


def _is_freeuv_root(root: Path) -> bool:
    return (root / "detail_encoder" / "__init__.py").is_file()


def _resolve_env_root() -> Path | None:
    root_str = os.environ.get(FREEUV_ENV_VAR)
    if not root_str:
        return None
    root = Path(root_str).expanduser().resolve()
    return root if _is_freeuv_root(root) else None


def _resolve_installed_root() -> Path | None:
    spec = importlib.util.find_spec("detail_encoder")
    if spec is None:
        return None
    for location in list(spec.submodule_search_locations or []):
        package_dir = Path(location).resolve()
        root = package_dir.parent
        if _is_freeuv_root(root):
            return root
    return None


def resolve_freeuv_root(*, required: bool = True) -> Path | None:
    env_root = _resolve_env_root()
    if env_root is not None:
        return env_root

    for root in FREEUV_CANDIDATES:
        if _is_freeuv_root(root):
            return root

    installed_root = _resolve_installed_root()
    if installed_root is not None:
        return installed_root

    if not required:
        return None

    roots = ", ".join(str(p) for p in FREEUV_CANDIDATES)
    raise RuntimeError(
        "FreeUV runtime is not available. Install the upstream 'freeuv' package, "
        f"set {FREEUV_ENV_VAR} to a FreeUV checkout, or vendor it under one of: {roots}. "
        "Upstream: https://github.com/YangXingchao/FreeUV"
    )


def ensure_freeuv_on_path(*, required: bool = True) -> Path | None:
    """Inject the FreeUV vendor dir into sys.path.

    Required because `detail_encoder/` uses relative imports that break under
    isolated spec loading. Call lazily — never at module import.
    """
    root = resolve_freeuv_root(required=required)
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        log.info("Added FreeUV source root to sys.path: %s", root)
    return root
