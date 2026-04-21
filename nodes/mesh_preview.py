"""Render a matte preview image from a mesh payload."""
from __future__ import annotations

from comfy_api.latest import io

from .flame_render_util import hex_to_rgb, render_mesh, render_points
from .mesh_types import compute_vertex_normals, coerce_mesh


class MeshPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MeshPreview",
            display_name="Mesh Preview",
            category="KaoLRM",
            description=(
                "Render a matte preview from a reconstructed FLAME mesh. "
                "Non-commercial use only when driven by KaoLRM assets."
            ),
            inputs=[
                io.Mesh.Input("mesh", tooltip="MESH payload from KaoLRMReconstruct. Sampled point clouds render as dots."),
                io.Int.Input("width", default=512, min=64, max=2048, tooltip="Output image width in pixels."),
                io.Int.Input("height", default=512, min=64, max=2048, tooltip="Output image height in pixels."),
                io.Float.Input(
                    "camera_distance",
                    default=0.6, min=0.2, max=3.0, step=0.01,
                    tooltip="World-space distance from the mesh centroid along +Z.",
                ),
                io.Float.Input(
                    "fov_degrees",
                    default=30.0, min=5.0, max=90.0, step=0.1,
                    tooltip="Vertical field of view in degrees.",
                ),
                io.Float.Input(
                    "light_intensity",
                    default=1.0, min=0.0, max=3.0, step=0.01,
                    tooltip="Directional diffuse gain. Ambient is fixed at 0.35.",
                ),
                io.String.Input(
                    "background_color",
                    default="#808080",
                    tooltip="Hex color for pixels not covered by the mesh.",
                ),
                io.Combo.Input(
                    "renderer",
                    options=["soft_torch", "pytorch3d"],
                    default="soft_torch",
                    optional=True,
                    tooltip="'soft_torch' is pure-torch and portable. 'pytorch3d' is faster but needs the pytorch3d install.",
                ),
            ],
            outputs=[io.Image.Output("IMAGE"), io.Mask.Output("MASK")],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        width: int = 512,
        height: int = 512,
        camera_distance: float = 0.6,
        fov_degrees: float = 30.0,
        light_intensity: float = 1.0,
        background_color: str = "#808080",
        renderer: str = "soft_torch",
    ):
        verts, faces = coerce_mesh(mesh)
        bg = hex_to_rgb(background_color)
        if faces.numel() == 0:
            rgb, mask = render_points(
                verts=verts,
                width=int(width),
                height=int(height),
                camera_distance=float(camera_distance),
                fov_deg=float(fov_degrees),
                bg=bg,
            )
            return io.NodeOutput(rgb.unsqueeze(0).cpu(), mask.unsqueeze(0).cpu())

        normals = compute_vertex_normals(verts, faces)
        rgb, mask = render_mesh(
            verts=verts,
            faces=faces,
            normals=normals,
            width=int(width),
            height=int(height),
            camera_distance=float(camera_distance),
            fov_deg=float(fov_degrees),
            light_intensity=float(light_intensity),
            bg=bg,
            backend=renderer,
        )
        return io.NodeOutput(rgb.unsqueeze(0).cpu(), mask.unsqueeze(0).cpu())
