"""FlameRender node — render a FLAME parameter dict to IMAGE + MASK."""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .flame_core import get_flame_core
from .flame_params import FLAME_PARAMS, params_dict_to_tensors, validate_params_dict
from .flame_render_util import hex_to_rgb, render_mesh
from .load_flame import FLAME_MODEL


class FlameRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlameRender",
            display_name="FLAME Render",
            category="FLAME",
            description="Render FLAME parameters to an image and silhouette mask.",
            inputs=[
                FLAME_MODEL.Input("flame_model"),
                FLAME_PARAMS.Input("flame_params"),
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
                io.String.Input(
                    "background_color",
                    default="#808080",
                    tooltip="Hex color used behind the rendered head.",
                ),
                io.Combo.Input(
                    "renderer",
                    options=["pytorch3d", "soft_torch"],
                    default="pytorch3d",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(
        cls,
        flame_model,
        flame_params,
        width=512,
        height=512,
        camera_distance=0.6,
        fov_degrees=30.0,
        light_intensity=1.0,
        background_color="#808080",
        renderer="pytorch3d",
    ):
        flame_params = validate_params_dict(flame_params, flame_model)
        device = flame_model.get("device", "cpu")
        core = get_flame_core(flame_model["gender"], flame_model["pkl_path"], device=device)

        shape, expr, pose, trans = params_dict_to_tensors(flame_params, flame_model, device=core.device)
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
        return io.NodeOutput(rgb.unsqueeze(0).cpu(), mask.unsqueeze(0).cpu())
