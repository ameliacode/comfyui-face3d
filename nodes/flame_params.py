"""FLAME_PARAMS custom type and JSON-safe helpers.

Contract: a FLAME_PARAMS value flowing between nodes is a plain Python dict:
  {
    "shape": [float, ...],   # length == flame_model.shape_dim (<=300)
    "expr":  [float, ...],   # length == flame_model.expr_dim  (<=100)
    "pose":  [float, ...],   # length 15 (global, neck, jaw, eye_L, eye_R)x(rx,ry,rz)
    "trans": [float, float, float],
  }
"""
from __future__ import annotations

import json
from typing import Any

import torch
from comfy_api.latest import io

FLAME_PARAMS = io.Custom("FLAME_PARAMS")

POSE_DIM = 15
TRANS_DIM = 3
MAX_SHAPE_DIM = 300
MAX_EXPR_DIM = 100


def default_params_dict(shape_dim: int = 50, expr_dim: int = 50) -> dict:
    return {
        "shape": [0.0] * int(shape_dim),
        "expr":  [0.0] * int(expr_dim),
        "pose":  [0.0] * POSE_DIM,
        "trans": [0.0] * TRANS_DIM,
    }


def default_params_json(shape_dim: int = 50, expr_dim: int = 50) -> str:
    return json.dumps(default_params_dict(shape_dim, expr_dim))


def _coerce_list(x: Any, length: int) -> list[float]:
    if x is None:
        return [0.0] * length
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten().cpu().tolist()
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"expected list/tensor, got {type(x).__name__}")
    out = [float(v) for v in x[:length]]
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out


def validate_params_dict(p: dict, flame_model: dict) -> dict:
    """Truncate or zero-pad incoming params to the flame_model's declared dims."""
    if not isinstance(p, dict):
        raise ValueError(f"FLAME_PARAMS must be a dict, got {type(p).__name__}")
    shape_dim = int(flame_model.get("shape_dim", 50))
    expr_dim = int(flame_model.get("expr_dim", 50))
    return {
        "shape": _coerce_list(p.get("shape"), shape_dim),
        "expr":  _coerce_list(p.get("expr"),  expr_dim),
        "pose":  _coerce_list(p.get("pose"),  POSE_DIM),
        "trans": _coerce_list(p.get("trans"), TRANS_DIM),
    }


def params_dict_to_tensors(
    p: dict,
    flame_model: dict,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (shape[1,Ns], expr[1,Ne], pose[1,15], trans[1,3])."""
    p = validate_params_dict(p, flame_model)
    kw = {"device": device, "dtype": dtype}
    shape = torch.tensor(p["shape"], **kw).unsqueeze(0)
    expr  = torch.tensor(p["expr"],  **kw).unsqueeze(0)
    pose  = torch.tensor(p["pose"],  **kw).unsqueeze(0)
    trans = torch.tensor(p["trans"], **kw).unsqueeze(0)
    return shape, expr, pose, trans


def tensors_to_params_dict(
    shape: torch.Tensor, expr: torch.Tensor, pose: torch.Tensor, trans: torch.Tensor
) -> dict:
    def _flat(t):
        return t.detach().flatten().cpu().tolist()
    return {
        "shape": _flat(shape),
        "expr":  _flat(expr),
        "pose":  _flat(pose),
        "trans": _flat(trans),
    }


def parse_params_json(s: str) -> dict | None:
    if not s:
        return None
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    return d
