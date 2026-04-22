"""KaoLRM model descriptor loader and weight resolver."""
from __future__ import annotations

import logging
from pathlib import Path

import folder_paths
import torch
from comfy_api.latest import io

from .kaolrm_runtime import resolve_kaolrm_root

log = logging.getLogger(__name__)

KAOLRM_MODEL = io.Custom("KAOLRM_MODEL")

KAOLRM_SUBDIR = "kaolrm"
KAOLRM_FILENAMES = {
    "mono": "mono.safetensors",
    "multiview": "multiview.safetensors",
}
KAOLRM_CONFIG_FILENAMES = {
    "mono": "mono.config.json",
    "multiview": "multiview.config.json",
}
FLAME_FILENAME = "generic_model.pkl"
KAOLRM_RELEASE_URL = "https://github.com/CyberAgentAILab/KaoLRM/releases"
FLAME_URL = "https://flame.is.tue.mpg.de/"


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> str:
    if device == "cpu":
        return "fp32"
    if dtype == "auto":
        return "fp16"
    return dtype


def ensure_kaolrm_weights(variant: str) -> Path:
    path = Path(folder_paths.models_dir) / KAOLRM_SUBDIR / KAOLRM_FILENAMES[variant]
    if not path.exists():
        raise RuntimeError(
            f"Missing KaoLRM checkpoint '{path.name}'. Place it at '{path}'. "
            f"Upstream release: {KAOLRM_RELEASE_URL}"
        )
    return path


def ensure_kaolrm_config(variant: str) -> Path:
    path = Path(folder_paths.models_dir) / KAOLRM_SUBDIR / KAOLRM_CONFIG_FILENAMES[variant]
    if not path.exists():
        raise RuntimeError(
            f"Missing KaoLRM config '{path.name}'. Place it at '{path}'. "
            f"Upstream release: {KAOLRM_RELEASE_URL}"
        )
    return path


def ensure_generic_flame_pkl() -> Path:
    path = Path(folder_paths.models_dir) / "flame" / FLAME_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"Missing FLAME generic model '{path.name}'. Place it at '{path}'. "
            f"Register and download from {FLAME_URL}"
        )
    return path


class LoadKaoLRM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadKaoLRM",
            display_name="Load KaoLRM",
            category="KaoLRM",
            description=(
                "Resolve KaoLRM mesh-reconstruction assets and output a model descriptor. "
                "Non-commercial use only."
            ),
            inputs=[
                io.Combo.Input(
                    "variant",
                    options=["mono", "multiview"],
                    default="mono",
                    tooltip="Checkpoint variant. 'mono' for single-image, 'multiview' for multi-view inputs.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    optional=True,
                    tooltip="'auto' picks cuda when available, cpu otherwise.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["auto", "fp32", "fp16", "bf16"],
                    default="auto",
                    optional=True,
                    tooltip="'auto' uses fp16 on cuda, fp32 on cpu. FLAME head always runs in fp32 to avoid expression underflow.",
                ),
                io.Boolean.Input(
                    "i_understand_non_commercial",
                    default=False,
                    tooltip="KaoLRM (CC BY-NC 4.0) and FLAME (MPI non-commercial) restrict outputs to research use.",
                ),
            ],
            outputs=[KAOLRM_MODEL.Output(display_name="kaolrm_model")],
        )

    @classmethod
    def execute(
        cls,
        variant: str = "mono",
        device: str = "auto",
        dtype: str = "auto",
        i_understand_non_commercial: bool = False,
    ):
        if not i_understand_non_commercial:
            raise RuntimeError("Set 'i_understand_non_commercial' to True before using KaoLRM.")
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
        ckpt_path = ensure_kaolrm_weights(variant)
        config_path = ensure_kaolrm_config(variant)
        flame_pkl_path = ensure_generic_flame_pkl()
        kaolrm_root = resolve_kaolrm_root(required=False)
        config = {
            "variant": variant,
            "ckpt_path": str(ckpt_path),
            "config_path": str(config_path),
            "flame_pkl_path": str(flame_pkl_path),
            "device": resolved_device,
            "dtype": resolved_dtype,
            "kaolrm_root": str(kaolrm_root) if kaolrm_root is not None else None,
        }
        return io.NodeOutput(config)
