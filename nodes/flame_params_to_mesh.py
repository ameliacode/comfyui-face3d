"""FLAMEParamsToMesh — re-solve a FLAME mesh from FLAME_PARAMS.

Uses the project's own `FlameCore` (nodes/flame_core.py) rather than KaoLRM's
vendored `flame.py` — decouples the SMIRK path from the KaoLRM source tree.
The FLAME pkl path is still resolved via `ensure_generic_flame_pkl()` (same
`models/flame/generic_model.pkl` KaoLRM uses).

KaoLRM's `pose[6]` layout is `[global(3)|jaw(3)]`. `FlameCore.forward` expects
`pose[15]` = `[global|neck|jaw|eye_L|eye_R]` in axis-angle. Expand with zeros
for neck + eyes, matching KaoLRM's flame.py which hardcodes those joints to
rest.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io
from comfy_api.latest._util import MESH as MeshPayload

from .flame_core import N_VERTICES, get_flame_core
from .flame_params_wire import FLAME_PARAMS, validate_flame_params
from .kaolrm_load import ensure_generic_flame_pkl


def _expand_pose_6_to_15(pose6: torch.Tensor) -> torch.Tensor:
    """[B, 6] = [global|jaw]  →  [B, 15] = [global|neck=0|jaw|eye_L=0|eye_R=0]."""
    if pose6.shape[-1] != 6:
        raise RuntimeError(f"Expected pose [B, 6], got {tuple(pose6.shape)}.")
    zeros = torch.zeros(pose6.shape[0], 3, dtype=pose6.dtype, device=pose6.device)
    return torch.cat([pose6[:, :3], zeros, pose6[:, 3:], zeros, zeros], dim=1)


class FLAMEParamsToMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FLAMEParamsToMesh",
            display_name="FLAME Params → Mesh",
            category="KaoLRM",
            description=(
                "Solve a FLAME mesh from FLAME_PARAMS via the project's FlameCore. "
                "Honors fix_z_trans so mono-origin KaoLRM params reproduce their "
                "original pose through the merge path."
            ),
            inputs=[
                FLAME_PARAMS.Input(
                    "flame_params",
                    tooltip="Merged FLAME_PARAMS dict. All tensors [B, N]; fix_z_trans required.",
                ),
            ],
            outputs=[io.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, flame_params):
        validate_flame_params(flame_params, source="flame_params")

        flame_pkl_path = ensure_generic_flame_pkl()
        core = get_flame_core(gender="generic", pkl_path=str(flame_pkl_path), device="cpu")

        shape = flame_params["shape"].to("cpu").float()
        expression = flame_params["expression"].to("cpu").float()
        pose6 = flame_params["pose"].to("cpu").float()
        scale = flame_params["scale"].to("cpu").float()
        translation = flame_params["translation"].to("cpu").float().clone()

        if bool(flame_params["fix_z_trans"]):
            translation[:, -1] = 0.0

        pose15 = _expand_pose_6_to_15(pose6)
        zero_trans = torch.zeros_like(translation)
        vertices = core.forward(shape, expression, pose15, zero_trans)
        vertices = vertices * scale.unsqueeze(2) + translation.unsqueeze(1)

        if vertices.shape[1] != N_VERTICES:
            raise RuntimeError(
                f"FlameCore returned {vertices.shape[1]} vertices; expected {N_VERTICES}."
            )

        faces = core.faces.detach().cpu().to(torch.int64).unsqueeze(0).repeat(vertices.shape[0], 1, 1)
        mesh = MeshPayload(
            vertices=vertices.detach().cpu().contiguous(),
            faces=faces.contiguous(),
        )
        mesh.base_vertices = vertices.detach().cpu().contiguous()
        mesh.base_faces = faces.contiguous()
        mesh.flame_params = {
            "shape": shape[0].contiguous(),
            "expression": expression[0].contiguous(),
            "pose": pose6[0].contiguous(),
            "scale": scale[0].contiguous(),
            "translation": translation[0].contiguous(),
        }
        mesh.fix_z_trans = bool(flame_params["fix_z_trans"])
        mesh.gender = "generic"
        mesh.source_resolution = 224
        mesh.topology = "mesh"
        mesh.num_sampling = N_VERTICES
        return io.NodeOutput(mesh)
