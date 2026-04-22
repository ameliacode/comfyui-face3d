"""KaoLRM mesh reconstruction node scaffold."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_api.latest import io
from comfy_api.latest._util import MESH as MeshPayload

from .flame_params_wire import FLAME_PARAMS
from .kaolrm_load import KAOLRM_MODEL
from .kaolrm_mesh_model import load_mesh_only_model
from .kaolrm_runtime import import_kaolrm_symbols, resolve_kaolrm_root

_KAOLRM_CACHE: dict[tuple[str, str, str, str, str, str, str], torch.nn.Module] = {}

FLAME_VERT_COUNT = 5023


def _prepare_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"expected IMAGE tensor [B, H, W, 3], got {tuple(image.shape)}")
    chw = image.permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
    if chw.shape[-2:] != (224, 224):
        chw = F.interpolate(chw, size=(224, 224), mode="bicubic", align_corners=False)
    return chw


def _get_cached_model(kaolrm_model: dict) -> torch.nn.Module:
    kaolrm_root = kaolrm_model.get("kaolrm_root")
    if not kaolrm_root:
        resolved_root = resolve_kaolrm_root()
        kaolrm_root = str(resolved_root)
        kaolrm_model["kaolrm_root"] = kaolrm_root
    key = (
        kaolrm_model["variant"],
        kaolrm_model["device"],
        kaolrm_model["dtype"],
        kaolrm_model["ckpt_path"],
        kaolrm_model["config_path"],
        kaolrm_model["flame_pkl_path"],
        kaolrm_root,
    )
    model = _KAOLRM_CACHE.get(key)
    if model is None:
        model = load_mesh_only_model(
            kaolrm_root=kaolrm_root,
            variant=kaolrm_model["variant"],
            ckpt_path=kaolrm_model["ckpt_path"],
            config_path=kaolrm_model["config_path"],
            flame_pkl_path=kaolrm_model["flame_pkl_path"],
            device=kaolrm_model["device"],
            dtype=kaolrm_model["dtype"],
        )
        _KAOLRM_CACHE[key] = model
    return model


def _build_source_camera(runtime: dict[str, object], dist_to_center: float, device: torch.device) -> torch.Tensor:
    extrinsics = torch.tensor(
        [[[1, 0, 0, 0], [0, 0, -1, -dist_to_center], [0, 1, 0, 0]]],
        dtype=torch.float32,
        device=device,
    )
    intrinsics = runtime["create_intrinsics"](f=0.75, c=0.5, device=device).unsqueeze(0)
    return runtime["build_camera_principle"](extrinsics, intrinsics)


def _params_to_cpu(decoded_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu()[0].contiguous() for k, v in decoded_params.items()}


_FLAME_PARAM_KEYS = ("shape", "expression", "pose", "scale", "translation")


def _build_flame_params_output(
    decoded_params_f32: dict[str, torch.Tensor], fix_z_trans: bool
) -> dict:
    out = {
        k: decoded_params_f32[k].detach().cpu().float().contiguous()
        for k in _FLAME_PARAM_KEYS
    }
    out["fix_z_trans"] = bool(fix_z_trans)
    return out


class KaoLRMReconstruct(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KaoLRMReconstruct",
            display_name="KaoLRM Reconstruct",
            category="KaoLRM",
            description=(
                "Run single-image FLAME mesh reconstruction through KaoLRM. "
                "Non-commercial use only."
            ),
            inputs=[
                KAOLRM_MODEL.Input(
                    "kaolrm_model",
                    tooltip="Descriptor from LoadKaoLRM. Resolves weights + FLAME pkl lazily on first use.",
                ),
                io.Image.Input(
                    "image",
                    tooltip="Single RGB image; auto-resized to 224x224. Batch must be 1.",
                ),
                io.Float.Input(
                    "source_cam_dist",
                    default=2.0, min=1.0, max=4.0, step=0.1,
                    tooltip="Canonical OpenLRM camera distance. 2.0 assumes the face fills most of the frame.",
                ),
                io.Int.Input(
                    "num_sampling",
                    default=FLAME_VERT_COUNT, min=1, max=200000,
                    tooltip=(
                        f"{FLAME_VERT_COUNT} emits the FLAME mesh. Any other value emits a "
                        "sampled point cloud; the base mesh is preserved in base_vertices/base_faces."
                    ),
                ),
            ],
            outputs=[
                io.Mesh.Output(),
                FLAME_PARAMS.Output(
                    display_name="flame_params",
                    tooltip=(
                        "Canonical FLAME_PARAMS with [1, N] tensors + fix_z_trans. "
                        "Feed directly into FLAMEParamsEdit or FLAMEParamsToMesh."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        kaolrm_model,
        image,
        source_cam_dist: float = 2.0,
        num_sampling: int = FLAME_VERT_COUNT,
    ):
        prepared = _prepare_image(image)
        if prepared.shape[0] != 1:
            raise RuntimeError(f"KaoLRMReconstruct currently supports batch size 1, got {prepared.shape[0]}.")
        runtime = import_kaolrm_symbols()
        model = _get_cached_model(kaolrm_model)
        model_param = next(model.parameters())
        device = model_param.device
        source_camera = _build_source_camera(runtime, float(source_cam_dist), device).to(dtype=model_param.dtype)
        prepared = prepared.to(device=device, dtype=model_param.dtype)

        with torch.no_grad():
            planes = model.forward_planes(prepared, source_camera)
            decoded_params = model.flame_decoder(planes)
            decoded_params_f32 = {k: v.float() for k, v in decoded_params.items()}
            fix_z = kaolrm_model.get("variant", "mono") == "mono"
            vertices, _, sampled_vertices = model.flame2mesh(decoded_params_f32, int(num_sampling), fix_z_trans=fix_z)
            base_faces = model.flame_model.faces_tensor.repeat(vertices.shape[0], 1, 1)

        sampled = int(num_sampling) != FLAME_VERT_COUNT
        mesh_vertices = sampled_vertices if sampled else vertices
        if sampled:
            mesh_faces = torch.empty((vertices.shape[0], 0, 3), dtype=base_faces.dtype, device=base_faces.device)
        else:
            mesh_faces = base_faces

        mesh = MeshPayload(
            vertices=mesh_vertices.detach().cpu().contiguous(),
            faces=mesh_faces.detach().cpu().to(torch.int64).contiguous(),
        )
        mesh.base_vertices = vertices.detach().cpu().contiguous()
        mesh.base_faces = base_faces.detach().cpu().to(torch.int64).contiguous()
        mesh.flame_params = _params_to_cpu(decoded_params_f32)
        mesh.fix_z_trans = fix_z
        mesh.gender = "generic"
        mesh.source_resolution = 224
        mesh.topology = "point_cloud" if sampled else "mesh"
        mesh.num_sampling = int(num_sampling)
        flame_params_out = _build_flame_params_output(decoded_params_f32, fix_z)
        return io.NodeOutput(mesh, flame_params_out)
