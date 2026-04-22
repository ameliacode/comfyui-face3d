"""SMIRK expression-encoder descriptor loader."""
from __future__ import annotations

import logging
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .kaolrm_load import resolve_device, resolve_dtype
from .smirk_runtime import resolve_smirk_root

log = logging.getLogger(__name__)

SMIRK_MODEL = io.Custom("SMIRK_MODEL")

SMIRK_SUBDIR = "smirk"
SMIRK_CKPT_FILENAME = "SMIRK_em1.pt"
SMIRK_CKPT_URL = "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view"


def ensure_smirk_weights() -> Path:
    path = Path(folder_paths.models_dir) / SMIRK_SUBDIR / SMIRK_CKPT_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"Missing SMIRK checkpoint '{path.name}'. Place it at '{path}'. "
            f"Download from {SMIRK_CKPT_URL}"
        )
    return path


class LoadSMIRK(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSMIRK",
            display_name="Load SMIRK",
            category="KaoLRM",
            description=(
                "Resolve SMIRK expression-encoder assets and output a model descriptor. "
                "SMIRK is MIT licensed, but it drives FLAME (MPI non-commercial) — "
                "the resulting pipeline is research-only."
            ),
            inputs=[
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
                    tooltip="'auto' uses fp16 on cuda, fp32 on cpu.",
                ),
                io.Boolean.Input(
                    "i_understand_non_commercial",
                    default=False,
                    tooltip=(
                        "SMIRK itself is MIT, but it runs on FLAME (MPI non-commercial). "
                        "Any mesh produced via this pipeline is research-only."
                    ),
                ),
            ],
            outputs=[SMIRK_MODEL.Output(display_name="smirk_model")],
        )

    @classmethod
    def execute(
        cls,
        device: str = "auto",
        dtype: str = "auto",
        i_understand_non_commercial: bool = False,
    ):
        if not i_understand_non_commercial:
            raise RuntimeError(
                "Set 'i_understand_non_commercial' to True before using SMIRK. "
                "FLAME downstream constrains the pipeline to research use."
            )
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
        ckpt_path = ensure_smirk_weights()
        smirk_root = resolve_smirk_root(required=False)
        descriptor = {
            "ckpt_path": str(ckpt_path),
            "device": resolved_device,
            "dtype": resolved_dtype,
            "smirk_root": str(smirk_root) if smirk_root is not None else None,
        }
        return io.NodeOutput(descriptor)
