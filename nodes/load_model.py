import logging
from pathlib import Path

import folder_paths
from comfy_api.latest import io

log = logging.getLogger(__name__)

# ── change these for your actual model ──────────────────────────────────────
HF_REPO_ID = "your-hf-org/your-model"
HF_FILENAME = "model.safetensors"
MODEL_SUBDIR = "your_model"          # inside ComfyUI/models/
# ────────────────────────────────────────────────────────────────────────────

MY_MODEL_CONFIG = io.Custom("MY_MODEL_CONFIG")


class LoadMyModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadMyModel",
            display_name="(Down)Load My Model",
            category="MyNodes",
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    optional=True,
                    tooltip="Model precision. 'auto' selects based on GPU.",
                ),
                io.Boolean.Input(
                    "compile",
                    default=False,
                    optional=True,
                    tooltip="Enable torch.compile for faster inference (first run is slower).",
                ),
            ],
            outputs=[
                MY_MODEL_CONFIG.Output(display_name="model_config"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", compile=False):
        import torch
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        checkpoint = Path(folder_paths.models_dir) / MODEL_SUBDIR / HF_FILENAME

        if not checkpoint.exists():
            log.info("Model not found — downloading from HuggingFace…")
            cls._download()

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]

        config = {
            "checkpoint_path": str(checkpoint),
            "precision": precision,
            "dtype": dtype_str,   # string so it's JSON-safe across process boundaries
            "compile": compile,
        }
        return io.NodeOutput(config)

    @staticmethod
    def _download():
        from huggingface_hub import hf_hub_download
        dest = Path(folder_paths.models_dir) / MODEL_SUBDIR
        dest.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=str(dest),
        )
