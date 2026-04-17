"""Shared utilities for MyNodes."""

import numpy as np
import torch
from PIL import Image as PILImage


def tensor_to_pil(tensor: torch.Tensor) -> PILImage.Image:
    """Convert a single image tensor [H,W,C] or [1,H,W,C] to PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(arr)


def pil_to_tensor(image: PILImage.Image) -> torch.Tensor:
    """Convert a PIL image to a batched tensor [1,H,W,C]."""
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def resize_tensor(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize an image tensor [B,H,W,C] using bilinear interpolation."""
    import torch.nn.functional as F
    x = image.permute(0, 3, 1, 2)          # [B,C,H,W]
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)           # [B,H,W,C]
