"""Runtime helpers for resolving and importing SMIRK."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMIRK_ENV_VAR = "SMIRK_ROOT"
SMIRK_CANDIDATES = [REPO_ROOT / "third_party" / "smirk"]


def _is_smirk_root(root: Path) -> bool:
    return (root / "src" / "smirk_encoder.py").is_file()


def _resolve_env_root() -> Path | None:
    root_str = os.environ.get(SMIRK_ENV_VAR)
    if not root_str:
        return None
    root = Path(root_str).expanduser().resolve()
    return root if _is_smirk_root(root) else None


def _resolve_installed_root() -> Path | None:
    spec = importlib.util.find_spec("smirk")
    if spec is None:
        return None
    for location in list(spec.submodule_search_locations or []):
        package_dir = Path(location).resolve()
        root = package_dir.parent
        if _is_smirk_root(root):
            return root
    if spec.origin:
        root = Path(spec.origin).resolve().parent.parent
        if _is_smirk_root(root):
            return root
    return None


def resolve_smirk_root(*, required: bool = True) -> Path | None:
    env_root = _resolve_env_root()
    if env_root is not None:
        return env_root

    for root in SMIRK_CANDIDATES:
        if _is_smirk_root(root):
            return root

    installed_root = _resolve_installed_root()
    if installed_root is not None:
        return installed_root

    if not required:
        return None

    roots = ", ".join(str(p) for p in SMIRK_CANDIDATES)
    raise RuntimeError(
        "SMIRK runtime is not available. Install the upstream 'smirk' package, "
        f"set {SMIRK_ENV_VAR} to a SMIRK checkout, or vendor it under one of: {roots}. "
        "Upstream: https://github.com/georgeretsi/smirk"
    )


def ensure_smirk_on_path(*, required: bool = True) -> Path | None:
    root = resolve_smirk_root(required=required)
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        log.info("Added SMIRK source root to sys.path: %s", root)
    return root


def import_smirk_encoder() -> type:
    """Load SmirkEncoder via isolated spec loading to sidestep upstream demo imports."""
    root = ensure_smirk_on_path()
    encoder_path = root / "src" / "smirk_encoder.py"
    try:
        spec = importlib.util.spec_from_file_location("_smirk_encoder_isolated", encoder_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            f"SMIRK import failed because dependency '{missing}' is missing. "
            "Install SMIRK's upstream requirements (e.g., `pip install timm`), then retry. "
            f"Source root: {root}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to import SmirkEncoder from '{encoder_path}': {e}") from e

    encoder_cls = getattr(module, "SmirkEncoder", None)
    if encoder_cls is None:
        raise RuntimeError(
            f"'{encoder_path}' does not define SmirkEncoder. "
            "Upstream API may have changed — check https://github.com/georgeretsi/smirk."
        )
    return encoder_cls
