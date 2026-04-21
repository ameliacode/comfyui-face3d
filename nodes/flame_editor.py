"""Single-node FLAME viewer/editor with realtime preview and render outputs."""
from __future__ import annotations

import json
import logging

import torch
from comfy_api.latest import io

from .flame_params import (
    FLAME_PARAMS,
    default_params_dict,
    default_params_json,
    parse_params_json,
    params_dict_to_tensors,
    validate_params_dict,
)
from .flame_core import get_flame_core
from .flame_render_util import hex_to_rgb, render_mesh
from .load_flame import ensure_flame_assets

log = logging.getLogger(__name__)


def _resolve_params(params_json: str, flame_params_in, flame_model: dict) -> dict:
    parsed = parse_params_json(params_json)
    if parsed is not None:
        return validate_params_dict(parsed, flame_model)
    if flame_params_in is not None:
        return validate_params_dict(flame_params_in, flame_model)
    return default_params_dict(flame_model.get("shape_dim", 50), flame_model.get("expr_dim", 50))


class FlameEditor(io.ComfyNode):
    """Edit FLAME parameters with a live 3D preview and rendered outputs."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlameEditor",
            display_name="FLAME Viewer",
            category="FLAME",
            description=(
                "Single-node FLAME viewer like the official interactive model viewer: "
                "pick a model, edit sliders live, preview in realtime, and output both "
                "edited parameters and rendered images."
            ),
            inputs=[
                io.Combo.Input(
                    "gender",
                    options=["generic", "female", "male"],
                    default="generic",
                    tooltip="FLAME gender variant.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    optional=True,
                    tooltip="Device used for render-time FLAME forward pass.",
                ),
                io.Int.Input(
                    "shape_dim",
                    default=50, min=1, max=300,
                    tooltip="How many shape PCA components to expose in the editor.",
                ),
                io.Int.Input(
                    "expr_dim",
                    default=50, min=1, max=100,
                    tooltip="How many expression PCA components to expose in the editor.",
                ),
                FLAME_PARAMS.Input("flame_params_in", optional=True,
                                   tooltip="Optional: seed the editor on connect."),
                io.Int.Input("width", default=512, min=64, max=2048),
                io.Int.Input("height", default=512, min=64, max=2048),
                io.Float.Input(
                    "camera_distance",
                    default=0.6,
                    min=0.2,
                    max=3.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Float.Input(
                    "fov_degrees",
                    default=30.0,
                    min=5.0,
                    max=90.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Float.Input(
                    "light_intensity",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.String.Input("background_color", default="#808080"),
                io.Combo.Input(
                    "renderer",
                    options=["pytorch3d", "soft_torch"],
                    default="pytorch3d",
                    optional=True,
                ),
                io.String.Input(
                    "params_json",
                    multiline=True,
                    default=default_params_json(),
                    tooltip="Internal: serialized slider state. The JS widget owns this.",
                ),
            ],
            outputs=[
                FLAME_PARAMS.Output(display_name="flame_params"),
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(
        cls,
        gender,
        device,
        shape_dim,
        expr_dim,
        params_json,
        flame_params_in=None,
        width=512,
        height=512,
        camera_distance=0.6,
        fov_degrees=30.0,
        light_intensity=1.0,
        background_color="#808080",
        renderer="pytorch3d",
    ):
        # Empty string is allowed (will fall back to flame_params_in or defaults).
        if params_json:
            try:
                json.loads(params_json)
            except (ValueError, TypeError) as e:
                return f"params_json is not valid JSON: {e}"
        return True

    @classmethod
    def execute(
        cls,
        gender: str,
        device: str = "auto",
        shape_dim: int = 50,
        expr_dim: int = 50,
        flame_params_in=None,
        width: int = 512,
        height: int = 512,
        camera_distance: float = 0.6,
        fov_degrees: float = 30.0,
        light_intensity: float = 1.0,
        background_color: str = "#808080",
        renderer: str = "pytorch3d",
        params_json: str = "",
    ):
        pkl = ensure_flame_assets(gender)
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_core = get_flame_core(gender, pkl, device="cpu")
        flame_model = {
            "gender": gender,
            "pkl_path": str(pkl),
            "device": resolved_device,
            "shape_dim": min(int(shape_dim), cpu_core.max_shape_dim),
            "expr_dim": min(int(expr_dim), cpu_core.max_expr_dim),
            "n_vertices": cpu_core.n_vertices,
            "n_faces": cpu_core.n_faces,
        }
        params = _resolve_params(params_json, flame_params_in, flame_model)

        core = get_flame_core(gender, pkl, device=resolved_device)
        shape, expr, pose, trans = params_dict_to_tensors(params, flame_model, device=core.device)
        with torch.no_grad():
            verts = core.forward(shape, expr, pose, trans)[0]
            normals = core.compute_vertex_normals(verts)
            rgb, mask = render_mesh(
                verts=verts,
                faces=core.faces,
                normals=normals,
                width=int(width),
                height=int(height),
                camera_distance=float(camera_distance),
                fov_deg=float(fov_degrees),
                light_intensity=float(light_intensity),
                bg=hex_to_rgb(background_color),
                backend=renderer,
            )
        return io.NodeOutput(params, rgb.unsqueeze(0).cpu(), mask.unsqueeze(0).cpu())
