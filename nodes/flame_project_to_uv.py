"""FLAMEProjectToUV — mesh + source IMAGE → flaw UV + visibility MASK.

Bridges KaoLRM/SMIRK geometry into FreeUV's albedo branch. The flaw UV is
what `FreeUVGenerate.flaw_uv_image` expects; upstream FreeUV ships no script
to produce one from an arbitrary face image.

Algorithm: inverse-UV rasterization. For each UV-space triangle, find the
texels it covers, interpolate 3D vertex positions via barycentric coords,
project to screen via the KaoLRM camera, and sample the input IMAGE.
Back-face cull drops occluded texels; eyeball triangles (FLAME verts
3931-5022) are skipped so sclera pixels never leak into the albedo.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .flame_uv_template import load_uv_template

log = logging.getLogger(__name__)

KAOLRM_SOURCE_RES = 224
EYEBALL_VERT_LO = 3931
EYEBALL_VERT_HI = 5022
FREEUV_UV_RES = 512
DEFAULT_SOURCE_CAM_DIST = 2.0
FOCAL = 0.75
PRINCIPAL = 0.5


def _prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Return float32 `[1, 3, 224, 224]` in [0, 1]."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise RuntimeError(
            f"image must be [B, H, W, 3], got {tuple(image.shape)}"
        )
    if image.shape[0] != 1:
        raise RuntimeError(
            f"FLAMEProjectToUV currently supports batch size 1, got {image.shape[0]}."
        )
    chw = image.permute(0, 3, 1, 2).float().clamp(0.0, 1.0).contiguous()
    if chw.shape[-2:] != (KAOLRM_SOURCE_RES, KAOLRM_SOURCE_RES):
        log.info(
            "FLAMEProjectToUV: resizing input image from %s to %dx%d to match KaoLRM camera.",
            tuple(chw.shape[-2:]), KAOLRM_SOURCE_RES, KAOLRM_SOURCE_RES,
        )
        chw = F.interpolate(
            chw, size=(KAOLRM_SOURCE_RES, KAOLRM_SOURCE_RES),
            mode="bicubic", align_corners=False,
        ).clamp(0.0, 1.0)
    return chw


def _coerce_mesh_to_flame_topology(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(vertices [V, 3], faces [F, 3])` on CPU float32/int64.

    Handles the `num_sampling != 5023` point-cloud branch by falling back to
    `mesh.base_vertices` / `mesh.base_faces` (the original FLAME topology).
    """
    topology = getattr(mesh, "topology", "mesh")
    if topology == "point_cloud" or (
        hasattr(mesh, "faces") and getattr(mesh, "faces") is not None
        and getattr(mesh, "faces").numel() == 0
    ):
        base_verts = getattr(mesh, "base_vertices", None)
        base_faces = getattr(mesh, "base_faces", None)
        if base_verts is None or base_faces is None:
            raise RuntimeError(
                "FLAMEProjectToUV: mesh has no face topology. "
                "Set num_sampling=5023 in KaoLRMReconstruct, or ensure "
                "mesh.base_vertices and mesh.base_faces are populated."
            )
        verts_src = base_verts
        faces_src = base_faces
    else:
        verts_src = mesh.vertices
        faces_src = mesh.faces

    verts = verts_src.detach().cpu().float()
    faces = faces_src.detach().cpu().to(torch.int64)
    if verts.ndim == 3 and verts.shape[0] >= 1:
        verts = verts[0]
    if faces.ndim == 3 and faces.shape[0] >= 1:
        faces = faces[0]
    if verts.ndim != 2 or verts.shape[-1] != 3:
        raise RuntimeError(f"vertices must be [V, 3], got {tuple(verts.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise RuntimeError(f"faces must be [F, 3], got {tuple(faces.shape)}")
    return verts.contiguous(), faces.contiguous()


def _intrinsic_matrix(device: torch.device) -> torch.Tensor:
    """Standard 3x3 intrinsic for normalized [0, 1] image coords.

    Matches KaoLRM's `create_intrinsics(f=0.75, c=0.5)` values; applied after
    perspective divide to give pixel coords in [0, 1]. KaoLRM's own
    `create_intrinsics` returns a `(3, 2)` "principle" layout used to seed its
    camera embedding — that is NOT a standard projection matrix and is not used
    for per-vertex projection here.
    """
    return torch.tensor(
        [
            [FOCAL, 0.0, PRINCIPAL],
            [0.0, FOCAL, PRINCIPAL],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _camera_origin(source_cam_dist: float, device: torch.device) -> torch.Tensor:
    """World-space camera position: `(0, 0, -d)`. Camera looks down +Z."""
    return torch.tensor(
        [0.0, 0.0, -float(source_cam_dist)], dtype=torch.float32, device=device
    )


def _eyeball_triangle_mask(faces: torch.Tensor) -> torch.Tensor:
    """Return `[F]` boolean: True where all three verts are in the eyeball range."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    in_range = lambda t: (t >= EYEBALL_VERT_LO) & (t <= EYEBALL_VERT_HI)
    return in_range(a) & in_range(b) & in_range(c)


def _project_to_screen(
    world_points: torch.Tensor,
    source_cam_dist: float,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project `[N, 3]` world points to `[0, 1]` screen coords.

    Camera at world `(0, 0, -d)` looking down `+Z`, so `cam_space = world + (0, 0, d)`.
    Pinhole: `pixel = intrinsic @ cam_space / cam_z`. Returns
    `(screen_xy [N, 2], valid [N] bool)` where `valid=False` when `cam_z <= 0`
    (point behind camera).
    """
    cam_space = world_points.clone()
    cam_space[:, 2] = cam_space[:, 2] + float(source_cam_dist)
    z = cam_space[:, 2]
    valid = z > 1e-6
    z_safe = z.clamp(min=1e-6)
    pixel_h = cam_space @ intrinsic.T                       # [N, 3]
    screen = pixel_h[:, :2] / z_safe.unsqueeze(1)
    return screen, valid


def _bilinear_sample(image_chw: torch.Tensor, screen_xy: torch.Tensor) -> torch.Tensor:
    """Sample `image [1, 3, H, W]` at `[0, 1]` screen coords `[N, 2]` → `[N, 3]`.

    Uses `torch.nn.functional.grid_sample` with coords remapped to `[-1, 1]`.
    Out-of-bounds samples return zeros.
    """
    if screen_xy.shape[0] == 0:
        return torch.zeros(0, 3, dtype=image_chw.dtype, device=image_chw.device)
    grid = (screen_xy * 2.0 - 1.0).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image_chw, grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )
    return sampled.view(3, -1).T.contiguous()


def _project_to_uv(
    vertices: torch.Tensor,        # [V, 3] float32
    faces: torch.Tensor,           # [F, 3] int64
    vt: torch.Tensor,              # [V_uv, 2] float32, in [0, 1]
    ft: torch.Tensor,              # [F, 3] int64
    image_chw: torch.Tensor,       # [1, 3, 224, 224] float32
    uv_resolution: int,
    source_cam_dist: float,
    intrinsic: torch.Tensor,       # [3, 3] float32
    cam_origin: torch.Tensor,      # [3] float32
    eyeball_mask: torch.Tensor,    # [F] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-UV rasterization. Returns `(uv_rgb [R, R, 3], mask [R, R])`."""
    R = int(uv_resolution)
    device = vertices.device
    uv_out = torch.zeros(R, R, 3, dtype=torch.float32, device=device)
    uv_mask = torch.zeros(R, R, dtype=torch.float32, device=device)

    F_count = faces.shape[0]
    if ft.shape[0] != F_count:
        raise RuntimeError(
            f"ft row count ({ft.shape[0]}) does not match faces row count ({F_count}); "
            "the UV template does not correspond to this mesh topology."
        )

    vt_pix = vt * R

    for tri in range(F_count):
        if eyeball_mask[tri]:
            continue

        ia, ib, ic = int(faces[tri, 0]), int(faces[tri, 1]), int(faces[tri, 2])
        va, vb, vc = vertices[ia], vertices[ib], vertices[ic]

        n = torch.linalg.cross(vb - va, vc - va)
        ctr = (va + vb + vc) / 3.0
        view = cam_origin - ctr
        if torch.dot(view, n) <= 0.0:
            continue

        ua = vt_pix[int(ft[tri, 0])]
        ub = vt_pix[int(ft[tri, 1])]
        uc = vt_pix[int(ft[tri, 2])]

        u_min = torch.min(torch.min(ua[0], ub[0]), uc[0]).item()
        u_max = torch.max(torch.max(ua[0], ub[0]), uc[0]).item()
        v_min = torch.min(torch.min(ua[1], ub[1]), uc[1]).item()
        v_max = torch.max(torch.max(ua[1], ub[1]), uc[1]).item()

        px_lo = max(int(u_min), 0)
        px_hi = min(int(u_max) + 1, R - 1)
        py_lo = max(int(v_min), 0)
        py_hi = min(int(v_max) + 1, R - 1)
        if px_hi < px_lo or py_hi < py_lo:
            continue

        py, px = torch.meshgrid(
            torch.arange(py_lo, py_hi + 1, device=device),
            torch.arange(px_lo, px_hi + 1, device=device),
            indexing="ij",
        )
        u = px.float() + 0.5
        v = py.float() + 0.5

        denom = (ub[1] - uc[1]) * (ua[0] - uc[0]) + (uc[0] - ub[0]) * (ua[1] - uc[1])
        if denom.abs() < 1e-9:
            continue

        w_a = ((ub[1] - uc[1]) * (u - uc[0]) + (uc[0] - ub[0]) * (v - uc[1])) / denom
        w_b = ((uc[1] - ua[1]) * (u - uc[0]) + (ua[0] - uc[0]) * (v - uc[1])) / denom
        w_c = 1.0 - w_a - w_b
        inside = (w_a >= 0) & (w_b >= 0) & (w_c >= 0)
        if not inside.any():
            continue

        world = (w_a.unsqueeze(-1) * va
                 + w_b.unsqueeze(-1) * vb
                 + w_c.unsqueeze(-1) * vc)
        world_flat = world.reshape(-1, 3)
        screen, valid_proj = _project_to_screen(world_flat, source_cam_dist, intrinsic)
        inside_flat = inside.reshape(-1)
        take = inside_flat & valid_proj
        if not take.any():
            continue

        rgb = _bilinear_sample(image_chw, screen[take])
        take_grid = take.reshape(inside.shape)
        py_hit = py[take_grid]
        px_hit = px[take_grid]

        uv_out[py_hit, px_hit] = rgb
        uv_mask[py_hit, px_hit] = 1.0

    return uv_out, uv_mask


class FLAMEProjectToUV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FLAMEProjectToUV",
            display_name="FLAME Project To UV",
            category="KaoLRM",
            description=(
                "Project the source face image into FLAME UV space, producing the "
                "flaw_uv_image that FreeUVGenerate consumes. Uses the same camera "
                "KaoLRMReconstruct used (f=0.75, c=0.5). Back-face cull only; "
                "eyeball triangles are masked out. Non-commercial — consumes FLAME "
                "UV data under the MPI license."
            ),
            inputs=[
                io.Mesh.Input(
                    "mesh",
                    tooltip=(
                        "FLAME-topology mesh from KaoLRMReconstruct. If KaoLRM was "
                        "run with num_sampling != 5023, this node falls back to "
                        "mesh.base_vertices / mesh.base_faces silently."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Original face image fed to KaoLRMReconstruct. Batch must be 1. "
                        "Auto-resized to 224x224 (camera assumption)."
                    ),
                ),
                io.Float.Input(
                    "source_cam_dist",
                    default=DEFAULT_SOURCE_CAM_DIST, min=1.0, max=4.0, step=0.1,
                    tooltip=(
                        "Must match the value used in KaoLRMReconstruct. "
                        "2.0 is the KaoLRM canonical default."
                    ),
                ),
                io.Int.Input(
                    "uv_resolution",
                    default=FREEUV_UV_RES, min=64, max=2048,
                    tooltip=(
                        "Output UV grid side length. Keep at 512 for FreeUV "
                        "compatibility; other values are inspection-only."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="flaw_uv_image"),
                io.Mask.Output(display_name="visibility"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        image,
        source_cam_dist: float = DEFAULT_SOURCE_CAM_DIST,
        uv_resolution: int = FREEUV_UV_RES,
    ):
        vertices, faces = _coerce_mesh_to_flame_topology(mesh)
        image_chw = _prepare_image(image)

        vt, ft = load_uv_template()
        device = vertices.device

        intrinsic = _intrinsic_matrix(device)
        cam_origin = _camera_origin(float(source_cam_dist), device)
        eyeball_mask = _eyeball_triangle_mask(faces)

        uv_rgb, uv_mask = _project_to_uv(
            vertices=vertices.float(),
            faces=faces,
            vt=vt.to(device=device, dtype=torch.float32),
            ft=ft.to(device=device, dtype=torch.int64),
            image_chw=image_chw.to(device=device, dtype=torch.float32),
            uv_resolution=int(uv_resolution),
            source_cam_dist=float(source_cam_dist),
            intrinsic=intrinsic,
            cam_origin=cam_origin,
            eyeball_mask=eyeball_mask,
        )

        flaw_uv = uv_rgb.unsqueeze(0).cpu().contiguous()      # [1, R, R, 3]
        visibility = uv_mask.unsqueeze(0).cpu().contiguous()  # [1, R, R]
        return io.NodeOutput(flaw_uv, visibility)
