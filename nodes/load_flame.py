"""LoadFlameModel — locate or download a FLAME pickle and warm the in-proc cache.

The official FLAME 2023 weights live behind a license-acceptance form at
flame.is.tue.mpg.de. We don't automate that login. Instead we try a small
allowlist of HuggingFace mirrors that legitimately redistribute FLAME 2023 Open
under CC-BY-4.0; if none succeed we surface a clear instruction pointing the
user at the official URL with the expected drop path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .flame_core import get_flame_core

log = logging.getLogger(__name__)

FLAME_MODEL = io.Custom("FLAME_MODEL")

# Files placed manually by the user are accepted at any of these names.
LOCAL_FILENAME_CANDIDATES = {
    "generic": ["flame2023.pkl", "flame2023_Open.pkl", "flame2023_generic.pkl", "generic_model.pkl"],
    "female":  ["flame2023_female.pkl",  "female_model.pkl"],
    "male":    ["flame2023_male.pkl",    "male_model.pkl"],
}

# Best-effort download mirrors. Maintainers must verify each entry before
# publishing — do not add a mirror that lacks a CC-BY-4.0 redistribution notice.
HF_MIRRORS: list[dict] = [
    # {"repo_id": "<verified-mirror-id>", "filenames": {"generic": "flame2023.pkl", ...}}
]

MODEL_SUBDIR = "flame"


def _flame_dir() -> Path:
    return Path(folder_paths.models_dir) / MODEL_SUBDIR


def _resolve_local_pkl(gender: str) -> Path | None:
    base = _flame_dir()
    for name in LOCAL_FILENAME_CANDIDATES[gender]:
        p = base / name
        if p.is_file():
            return p
    return None


def _try_download(gender: str) -> Path | None:
    if not HF_MIRRORS:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning("huggingface_hub not installed; cannot auto-download FLAME.")
        return None

    dest = _flame_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for mirror in HF_MIRRORS:
        filename = mirror.get("filenames", {}).get(gender)
        if not filename:
            continue
        try:
            log.info("Trying FLAME mirror %s/%s", mirror["repo_id"], filename)
            path = hf_hub_download(repo_id=mirror["repo_id"], filename=filename, local_dir=str(dest))
            return Path(path)
        except Exception as e:
            log.warning("Mirror %s failed: %s", mirror["repo_id"], e)
    return None


def ensure_flame_assets(gender: str) -> Path:
    p = _resolve_local_pkl(gender)
    if p is not None:
        return p
    p = _try_download(gender)
    if p is not None:
        return p
    expected = _flame_dir() / LOCAL_FILENAME_CANDIDATES[gender][0]
    raise RuntimeError(
        f"FLAME {gender} model not found. Place it at:\n  {expected}\n"
        "Download from https://flame.is.tue.mpg.de (registration required) or "
        "configure a verified HuggingFace mirror in HF_MIRRORS."
    )


def _resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        return "cuda"
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class LoadFlameModel(io.ComfyNode):
    """Resolve a FLAME pickle and warm the CPU forward-pass cache."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadFlameModel",
            display_name="Load FLAME Model",
            category="FLAME",
            description="Locate or download a FLAME pickle and warm the forward-pass cache.",
            inputs=[
                io.Combo.Input(
                    "gender",
                    options=["generic", "female", "male"],
                    default="generic",
                    tooltip="FLAME gender variant.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    optional=True,
                    tooltip="Device for the rendering forward pass. The editor preview always runs on CPU.",
                ),
                io.Int.Input(
                    "shape_dim",
                    default=50, min=1, max=300,
                    tooltip="How many shape PCA components to expose downstream.",
                ),
                io.Int.Input(
                    "expr_dim",
                    default=50, min=1, max=100,
                    tooltip="How many expression PCA components to expose downstream.",
                ),
            ],
            outputs=[FLAME_MODEL.Output(display_name="flame_model")],
        )

    @classmethod
    def execute(cls, gender: str, device: str = "auto", shape_dim: int = 50, expr_dim: int = 50):
        pkl = ensure_flame_assets(gender)
        resolved_device = _resolve_device(device)

        # Always warm the CPU core — the editor route uses it.
        cpu_core = get_flame_core(gender, pkl, device="cpu")
        shape_dim = min(int(shape_dim), cpu_core.max_shape_dim)
        expr_dim = min(int(expr_dim), cpu_core.max_expr_dim)

        descriptor = {
            "gender": gender,
            "pkl_path": str(pkl),
            "device": resolved_device,
            "shape_dim": shape_dim,
            "expr_dim": expr_dim,
            "n_vertices": cpu_core.n_vertices,
            "n_faces": cpu_core.n_faces,
        }
        log.info("FLAME model loaded: %s", descriptor)
        return io.NodeOutput(descriptor)
