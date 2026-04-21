"""Image preprocessing for KaoLRM (background removal via rembg Python API)."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

log = logging.getLogger(__name__)

_REMBG_SESSIONS: dict[str, object] = {}


def _resize_image(image: torch.Tensor, size: int = 224) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"expected IMAGE tensor [B, H, W, 3], got {tuple(image.shape)}")
    chw = image.permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
    if chw.shape[-2:] != (size, size):
        chw = F.interpolate(chw, size=(size, size), mode="bicubic", align_corners=False)
    return chw.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, size: int = 224) -> torch.Tensor:
    chw = mask.unsqueeze(1).float().clamp(0.0, 1.0)
    if chw.shape[-2:] != (size, size):
        chw = F.interpolate(chw, size=(size, size), mode="bilinear", align_corners=False)
    return chw.squeeze(1).clamp(0.0, 1.0)


def _parse_hex_color(color: str) -> tuple[float, float, float]:
    s = color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"background_color must be hex like '#RRGGBB', got '{color}'")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _get_rembg_session(model_name: str):
    session = _REMBG_SESSIONS.get(model_name)
    if session is None:
        from rembg import new_session
        session = new_session(model_name)
        _REMBG_SESSIONS[model_name] = session
    return session


def _remove_background_batch(
    image: torch.Tensor,
    rembg_model: str,
    bg_rgb: tuple[float, float, float],
    composite_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from PIL import Image
    from rembg import remove

    session = _get_rembg_session(rembg_model)
    bg = np.array(bg_rgb, dtype=np.float32)

    composites: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        pil_rgb = Image.fromarray(arr, mode="RGB")
        rgba_pil = remove(pil_rgb, session=session, post_process_mask=True)
        rgba = np.array(rgba_pil.convert("RGBA"))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "rembg item %d rgba=%s alpha zeros=%d fulls=%d",
                i, rgba.shape,
                int((rgba[..., 3] == 0).sum()),
                int((rgba[..., 3] == 255).sum()),
            )

        rgb = rgba[..., :3].astype(np.float32) / 255.0
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        effective = np.clip(alpha * float(composite_alpha), 0.0, 1.0)
        comp = rgb * effective[..., None] + bg[None, None, :] * (1.0 - effective[..., None])
        composites.append(comp)
        masks.append(effective)

    image_out = torch.from_numpy(np.stack(composites, axis=0).astype(np.float32))
    mask_out = torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))
    return image_out, mask_out


class KaoLRMPreprocess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KaoLRMPreprocess",
            display_name="KaoLRM Preprocess",
            category="KaoLRM",
            description=(
                "Resize and optionally background-strip a portrait for KaoLRM "
                "using the rembg Python API. Non-commercial use only."
            ),
            inputs=[
                io.Image.Input("image", tooltip="RGB portrait; any HxW — resized to 224x224 internally."),
                io.Boolean.Input(
                    "remove_background",
                    default=False,
                    tooltip="Run rembg before resize. First invocation downloads the rembg model.",
                ),
                io.Combo.Input(
                    "rembg_model",
                    options=[
                        "u2net",
                        "u2netp",
                        "u2net_human_seg",
                        "silueta",
                        "isnet-general-use",
                        "birefnet-portrait",
                    ],
                    default="u2net",
                    optional=True,
                    tooltip="rembg segmentation model. 'u2net_human_seg' and 'birefnet-portrait' are portrait-tuned.",
                ),
                io.String.Input(
                    "background_color",
                    default="#FFFFFF",
                    tooltip="Hex color composited behind the stripped foreground.",
                ),
                io.Float.Input(
                    "composite_alpha",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Multiplier on rembg alpha. <1.0 softens the cutout edge.",
                ),
            ],
            outputs=[io.Image.Output("IMAGE"), io.Mask.Output("MASK")],
        )

    @classmethod
    def execute(
        cls,
        image,
        remove_background: bool = False,
        rembg_model: str = "u2net",
        background_color: str = "#FFFFFF",
        composite_alpha: float = 1.0,
    ):
        log.debug(
            "KaoLRMPreprocess remove_background=%s model=%s input=%s",
            remove_background, rembg_model, tuple(image.shape),
        )
        if remove_background:
            bg_rgb = _parse_hex_color(background_color)
            stripped, mask_full = _remove_background_batch(
                image, rembg_model, bg_rgb, composite_alpha
            )
            resized_image = _resize_image(stripped, size=224)
            resized_mask = _resize_mask(mask_full, size=224)
            return io.NodeOutput(resized_image, resized_mask)

        resized = _resize_image(image, size=224)
        mask = torch.ones(
            resized.shape[0], resized.shape[1], resized.shape[2], dtype=resized.dtype
        )
        return io.NodeOutput(resized, mask)
