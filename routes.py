"""HTTP routes for the FLAME editor live preview."""
from __future__ import annotations

import asyncio
import base64
import logging

import numpy as np
import torch
from aiohttp import web
from server import PromptServer

try:
    from .nodes.flame_core import get_flame_core
    from .nodes.flame_params import default_params_dict, validate_params_dict
    from .nodes.load_flame import ensure_flame_assets
except ImportError:  # pragma: no cover - test/import fallback
    from nodes.flame_core import get_flame_core
    from nodes.flame_params import default_params_dict, validate_params_dict
    from nodes.load_flame import ensure_flame_assets

log = logging.getLogger(__name__)
routes = PromptServer.instance.routes


def _encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _topology_sync(gender: str) -> dict:
    resolved_path = ensure_flame_assets(gender)
    core = get_flame_core(gender, resolved_path, device="cpu")
    faces = core.faces.detach().cpu().numpy().astype(np.int32, copy=False)
    template = core.v_template.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "gender": gender,
        "pkl_path": str(resolved_path),
        "faces_b64": _encode_array(faces),
        "template_b64": _encode_array(template),
        "n_vertices": int(core.n_vertices),
        "n_faces": int(core.n_faces),
    }


def _forward_sync(payload: dict) -> dict:
    gender = str(payload.get("gender", "generic"))
    pkl_path = str(ensure_flame_assets(gender))
    flame_model = {
        "gender": gender,
        "pkl_path": pkl_path,
        "device": "cpu",
        "shape_dim": int(payload.get("shape_dim", 50)),
        "expr_dim": int(payload.get("expr_dim", 50)),
    }
    params = payload.get("params") or default_params_dict(flame_model["shape_dim"], flame_model["expr_dim"])
    params = validate_params_dict(params, flame_model)
    core = get_flame_core(gender, pkl_path, device="cpu")

    with torch.no_grad():
        shape = torch.tensor(params["shape"], dtype=torch.float32, device=core.device).unsqueeze(0)
        expr = torch.tensor(params["expr"], dtype=torch.float32, device=core.device).unsqueeze(0)
        pose = torch.tensor(params["pose"], dtype=torch.float32, device=core.device).unsqueeze(0)
        trans = torch.tensor(params["trans"], dtype=torch.float32, device=core.device).unsqueeze(0)
        verts = core.forward(shape, expr, pose, trans)[0].detach().cpu().numpy().astype(np.float32, copy=False)

    return {
        "gender": gender,
        "pkl_path": pkl_path,
        "params": params,
        "verts_b64": _encode_array(verts),
        "n_vertices": int(core.n_vertices),
    }


@routes.get("/flame/faces")
async def flame_faces(request: web.Request) -> web.Response:
    try:
        gender = str(request.query.get("gender", "generic"))
        result = await asyncio.to_thread(_topology_sync, gender)
        return web.json_response(result)
    except Exception as e:
        log.exception("FLAME topology request failed")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/flame/forward")
async def flame_forward(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        result = await asyncio.to_thread(_forward_sync, payload)
        return web.json_response(result)
    except Exception as e:
        log.exception("FLAME preview failed")
        return web.json_response({"error": str(e)}, status=500)
