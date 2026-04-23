"""FreeUVGenerate — thin wrapper over FreeUV's SD1.5 + ControlNet + detail_encoder stack."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from PIL import Image

from .freeuv_assets import load_reference_uv
from .freeuv_load import FREEUV_MODEL
from .freeuv_runtime import ensure_freeuv_on_path, patch_huggingface_hub_compat

log = logging.getLogger(__name__)

_FREEUV_CACHE: dict[tuple[str, str, str, str, str, str, str | None], object] = {}


def _cache_key(freeuv_model: dict) -> tuple[str, str, str, str, str, str, str | None]:
    return (
        freeuv_model["device"],
        freeuv_model["dtype"],
        freeuv_model["sd15_root"],
        freeuv_model["clip_root"],
        freeuv_model["aligner_path"],
        freeuv_model["detail_path"],
        freeuv_model.get("freeuv_root"),
    )


def _load_freeuv_pipeline(freeuv_model: dict):
    """Build the FreeUV inference stack. First call wires sys.path and imports upstream.

    Returns an opaque handle `{"pipe": pipeline, "detail_extractor": detail_encoder_wrapper}`
    that `_run_generate` consumes. Mirrors upstream `inference.py`.
    """
    ensure_freeuv_on_path()
    patch_huggingface_hub_compat()

    # Upstream vendors pipeline_sd15.py at repo root and a detail_encoder/ package.
    # The class lives in detail_encoder.encoder_freeuv (upstream __init__.py is empty).
    from detail_encoder.encoder_freeuv import detail_encoder  # type: ignore[import-not-found]
    from diffusers import ControlNetModel, DDIMScheduler
    from pipeline_sd15 import (  # type: ignore[import-not-found]
        StableDiffusionControlNetPipeline,
    )
    from pipeline_sd15 import (
        UNet2DConditionModel as OriginalUNet2DConditionModel,
    )

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[freeuv_model["dtype"]]
    device = freeuv_model["device"]

    unet = OriginalUNet2DConditionModel.from_pretrained(
        freeuv_model["sd15_root"], subfolder="unet"
    ).to(device)
    uv_aligner = ControlNetModel.from_unet(unet)

    detail_extractor = detail_encoder(
        unet, freeuv_model["clip_root"], device, dtype=torch_dtype
    )

    aligner_state = torch.load(
        freeuv_model["aligner_path"], map_location="cpu", weights_only=True
    )
    uv_aligner.load_state_dict(aligner_state, strict=False)
    detail_state = torch.load(
        freeuv_model["detail_path"], map_location="cpu", weights_only=True
    )
    detail_extractor.load_state_dict(detail_state, strict=False)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        freeuv_model["sd15_root"],
        safety_checker=None,
        unet=unet,
        controlnet=uv_aligner,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return {"pipe": pipe, "detail_extractor": detail_extractor}


def _get_cached_pipeline(freeuv_model: dict):
    key = _cache_key(freeuv_model)
    handle = _FREEUV_CACHE.get(key)
    if handle is None:
        handle = _load_freeuv_pipeline(freeuv_model)
        _FREEUV_CACHE[key] = handle
    return handle


def _image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert an IMAGE tensor `[1, H, W, 3]` float32 in [0,1] to a 512×512 PIL RGB."""
    chw = image.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    if chw.shape[-2:] != (512, 512):
        chw = F.interpolate(chw, size=(512, 512), mode="bicubic", align_corners=False)
        chw = chw.clamp(0.0, 1.0)
    arr = (chw[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_to_image(pil: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to IMAGE `[1, 512, 512, 3]` float32 in [0,1]."""
    rgb = pil.convert("RGB")
    if rgb.size != (512, 512):
        rgb = rgb.resize((512, 512), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class FreeUVGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FreeUVGenerate",
            display_name="FreeUV Generate",
            category="KaoLRM",
            description=(
                "Generate a 512×512 albedo UV from a flawed UV input plus a reference UV. "
                "Thin wrapper over FreeUV's SD1.5 + ControlNet + detail-encoder stack. "
                "Non-commercial + CC BY-NC-SA 4.0 ShareAlike."
            ),
            inputs=[
                FREEUV_MODEL.Input("freeuv_model", tooltip="Descriptor from LoadFreeUV."),
                io.Image.Input(
                    "flaw_uv_image",
                    tooltip="512×512 UV image with imperfections. Batch must be 1.",
                ),
                io.Image.Input(
                    "reference_uv",
                    optional=True,
                    tooltip=(
                        "Clean reference UV template. Falls back to the bundled "
                        "`assets/freeuv_reference_uv.jpg` when not wired."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2**31 - 1,
                    optional=True,
                    tooltip="-1 → random each run; fix a value for reproducibility.",
                ),
                io.Float.Input(
                    "guidance_scale",
                    default=1.4,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    optional=True,
                    tooltip="Matches upstream default in FreeUV's inference.py.",
                ),
                io.Int.Input(
                    "num_inference_steps",
                    default=30,
                    min=1,
                    max=200,
                    optional=True,
                    tooltip="DDIM step count.",
                ),
            ],
            outputs=[io.Image.Output(display_name="albedo_uv")],
        )

    @classmethod
    def execute(
        cls,
        freeuv_model,
        flaw_uv_image,
        reference_uv=None,
        seed: int = -1,
        guidance_scale: float = 1.4,
        num_inference_steps: int = 30,
    ):
        if flaw_uv_image.ndim != 4 or flaw_uv_image.shape[-1] != 3:
            raise RuntimeError(
                f"flaw_uv_image must be [B, H, W, 3], got {tuple(flaw_uv_image.shape)}"
            )
        if flaw_uv_image.shape[0] != 1:
            raise RuntimeError(
                f"FreeUVGenerate currently supports batch size 1, got {flaw_uv_image.shape[0]}."
            )

        ref = reference_uv if reference_uv is not None else load_reference_uv()
        if ref.ndim != 4 or ref.shape[-1] != 3 or ref.shape[0] != 1:
            raise RuntimeError(
                f"reference_uv must be [1, H, W, 3], got {tuple(ref.shape)}"
            )

        if seed == -1:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        flaw_pil = _image_to_pil(flaw_uv_image)
        ref_pil = _image_to_pil(ref)

        handle = _get_cached_pipeline(freeuv_model)
        detail_extractor = handle["detail_extractor"]
        pipe = handle["pipe"]

        result_pil = detail_extractor.generate(
            uv_structure_image=ref_pil,
            flaw_uv_image=flaw_pil,
            pipe=pipe,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        if isinstance(result_pil, list):
            result_pil = result_pil[0]

        output = _pil_to_image(result_pil)
        return io.NodeOutput(output)
