"""SMIRK expression prediction node."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .flame_params_wire import FLAME_PARAMS
from .smirk_load import SMIRK_MODEL
from .smirk_runtime import import_smirk_encoder

log = logging.getLogger(__name__)

_SMIRK_CACHE: dict[tuple[str, str, str, str | None], torch.nn.Module] = {}


def _prepare_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"expected IMAGE tensor [B, H, W, 3], got {tuple(image.shape)}")
    chw = image.permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
    if chw.shape[-2:] != (224, 224):
        chw = F.interpolate(chw, size=(224, 224), mode="bicubic", align_corners=False)
    return chw


def _load_smirk_encoder(ckpt_path: str, device: str, dtype: str) -> torch.nn.Module:
    encoder_cls = import_smirk_encoder()
    encoder = encoder_cls(n_shape=300, n_exp=50)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Upstream checkpoints bundle smirk_encoder + smirk_generator weights under a flat
    # state_dict; we only want the encoder side. Mirrors demo.py:56 in georgeretsi/smirk.
    encoder_state = {
        k.replace("smirk_encoder.", "", 1): v
        for k, v in state.items()
        if k.startswith("smirk_encoder.")
    }
    if not encoder_state:
        # Older training dumps may already be encoder-only — fall through with the raw state.
        encoder_state = state
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"SMIRK checkpoint at '{ckpt_path}' is missing {len(missing)} required weights "
            f"(first: {preview}). Unexpected keys: {len(unexpected)}. "
            "Likely an encoder/config mismatch."
        )
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    encoder = encoder.to(device=device)
    if torch_dtype != torch.float32 and device != "cpu":
        encoder = encoder.to(dtype=torch_dtype)
    encoder.eval()
    return encoder


def _get_cached_smirk(smirk_model: dict) -> torch.nn.Module:
    key = (
        smirk_model["device"],
        smirk_model["dtype"],
        smirk_model["ckpt_path"],
        smirk_model.get("smirk_root"),
    )
    encoder = _SMIRK_CACHE.get(key)
    if encoder is None:
        encoder = _load_smirk_encoder(
            ckpt_path=smirk_model["ckpt_path"],
            device=smirk_model["device"],
            dtype=smirk_model["dtype"],
        )
        _SMIRK_CACHE[key] = encoder
    return encoder


class SMIRKPredict(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SMIRKPredict",
            display_name="SMIRK Predict",
            category="KaoLRM",
            description=(
                "Run the SMIRK expression encoder on a pre-cropped face image. "
                "Emits FLAME_PARAMS (expression + jaw_pose). Non-commercial use only."
            ),
            inputs=[
                SMIRK_MODEL.Input("smirk_model", tooltip="Descriptor from LoadSMIRK."),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Pre-cropped face at 224x224. SMIRK expects MediaPipe-style alignment; "
                        "auto-crop is v0.2. Batch must be 1."
                    ),
                ),
            ],
            outputs=[FLAME_PARAMS.Output(display_name="flame_params")],
        )

    @classmethod
    def execute(cls, smirk_model, image):
        prepared = _prepare_image(image)
        if prepared.shape[0] != 1:
            raise RuntimeError(
                f"SMIRKPredict currently supports batch size 1, got {prepared.shape[0]}."
            )
        encoder = _get_cached_smirk(smirk_model)
        model_param = next(encoder.parameters())
        device = model_param.device
        prepared = prepared.to(device=device, dtype=model_param.dtype)

        with torch.no_grad():
            out = encoder(prepared)

        # Cast everything to float32 on CPU; upstream emits [1, N] tensors.
        expression = out["expression_params"].detach().cpu().float()
        jaw = out["jaw_params"].detach().cpu().float()

        # SMIRK has no notion of identity/scale/translation; fill with zeros at [1, N]
        # so FLAMEParamsEdit can validate shape compatibility without ambiguity.
        # FLAMEParamsEdit's merge policy discards these and reads the primary input's values.
        shape = torch.zeros(1, 100, dtype=torch.float32)
        pose = torch.cat([torch.zeros(1, 3, dtype=torch.float32), jaw], dim=1)
        scale = torch.ones(1, 1, dtype=torch.float32)
        translation = torch.zeros(1, 3, dtype=torch.float32)

        flame_params = {
            "shape": shape,
            "expression": expression,
            "pose": pose,
            "scale": scale,
            "translation": translation,
            "fix_z_trans": False,
        }
        return io.NodeOutput(flame_params)
