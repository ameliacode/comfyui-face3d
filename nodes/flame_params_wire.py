"""FLAME_PARAMS wire type shared by SMIRK, KaoLRM shim, merge, and mesh re-solve."""
from __future__ import annotations

from comfy_api.latest import io

FLAME_PARAMS = io.Custom("FLAME_PARAMS")

CANONICAL_SHAPES = {
    "shape": (100,),
    "expression": (50,),
    "pose": (6,),
    "scale": (1,),
    "translation": (3,),
}


def validate_flame_params(params: dict, *, source: str) -> None:
    """Assert [B, N] layout with matching canonical dims. Raises RuntimeError on mismatch."""
    for key, expected_tail in CANONICAL_SHAPES.items():
        if key not in params:
            raise RuntimeError(f"FLAME_PARAMS from {source} missing key '{key}'.")
        t = params[key]
        if t.ndim != 2 or tuple(t.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"FLAME_PARAMS from {source}: '{key}' must be [B, {expected_tail[0]}], "
                f"got {tuple(t.shape)}. Flat [N] tensors are not accepted — the "
                "canonical wire carries batched [B, N] tensors."
            )
    if "fix_z_trans" not in params:
        raise RuntimeError(f"FLAME_PARAMS from {source} missing 'fix_z_trans' bool.")
