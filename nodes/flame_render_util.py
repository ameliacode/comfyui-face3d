"""Mesh rasterization backends for FLAME renders.

Two backends:
  - "pytorch3d" — preferred on Linux+CUDA; better shading quality.
  - "soft_torch" — pure-torch barycentric rasterizer. Works everywhere torch
    works. Default fallback when pytorch3d isn't importable.
"""
from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F

from ._optional_deps import try_import_pytorch3d

log = logging.getLogger(__name__)


def hex_to_rgb(s: str) -> tuple[float, float, float]:
    s = (s or "#808080").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0.5, 0.5, 0.5)
    try:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except ValueError:
        return (0.5, 0.5, 0.5)


def _auto_camera(verts: torch.Tensor, camera_distance: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Center the camera on the mesh bbox center, looking down -Z.

    Returns (eye, target) in world space. Both shape [3].
    """
    center = (verts.min(dim=0).values + verts.max(dim=0).values) * 0.5
    eye = center + torch.tensor([0.0, 0.0, float(camera_distance)], device=verts.device)
    return eye, center


def _look_at(eye: torch.Tensor, target: torch.Tensor, up=(0.0, 1.0, 0.0)) -> torch.Tensor:
    """Return a 3x3 world→camera rotation (row-major)."""
    up = torch.tensor(up, device=eye.device, dtype=eye.dtype)
    f = F.normalize((target - eye).unsqueeze(0), dim=1)[0]
    r = F.normalize(torch.cross(f, up, dim=0).unsqueeze(0), dim=1)[0]
    u = torch.cross(r, f, dim=0)
    R = torch.stack([r, u, -f], dim=0)   # camera axes in world coords, rows
    return R


def _project_ndc(verts: torch.Tensor, eye: torch.Tensor, target: torch.Tensor,
                 fov_deg: float, aspect: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (verts_ndc [V, 2] in [-1, 1], z_view [V])."""
    R = _look_at(eye, target)
    v_cam = (verts - eye) @ R.T   # [V, 3], +Z forward after our R
    # In our look_at, -f is the third row, so camera forward is -Z_cam. Flip z:
    v_cam = v_cam * torch.tensor([1.0, 1.0, -1.0], device=verts.device, dtype=verts.dtype)
    z = v_cam[:, 2].clamp(min=1e-4)
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    x_ndc = (v_cam[:, 0] * f / aspect) / z
    y_ndc = (v_cam[:, 1] * f) / z
    return torch.stack([x_ndc, y_ndc], dim=1), z


def _soft_torch_render(
    verts: torch.Tensor,     # [V, 3]
    faces: torch.Tensor,     # [F, 3] long
    normals: torch.Tensor,   # [V, 3]
    width: int, height: int,
    camera_distance: float, fov_deg: float,
    light_intensity: float,
    bg: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize [V, 3] → ([H, W, 3] RGB in [0, 1], [H, W] mask in [0, 1])."""
    device = verts.device
    aspect = width / height

    eye, target = _auto_camera(verts, camera_distance)
    verts_ndc, z_view = _project_ndc(verts, eye, target, fov_deg, aspect)

    # Pixel grid (NDC in [-1, 1]; y points up).
    xs = torch.linspace(-1, 1, width, device=device)
    ys = torch.linspace(1, -1, height, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    image = torch.full((height, width, 3), 0.0, device=device)
    mask = torch.zeros((height, width), device=device)
    z_buffer = torch.full((height, width), float("inf"), device=device)

    light_dir = F.normalize(torch.tensor([0.3, 0.5, 1.0], device=device), dim=0)
    ambient = 0.35
    diffuse_gain = float(light_intensity)
    base_color = torch.tensor([0.75, 0.75, 0.75], device=device)

    v_ndc = verts_ndc
    v_z = z_view
    v_n = F.normalize(normals, dim=1)

    # Per-face rasterization with a bounding-box prune. For the FLAME mesh
    # (~10k triangles at 512x512 this runs in <1s on GPU).
    for fi in range(faces.shape[0]):
        ia, ib, ic = faces[fi]
        a = v_ndc[ia]; b = v_ndc[ib]; c = v_ndc[ic]
        za = v_z[ia]; zb = v_z[ib]; zc = v_z[ic]

        xmin = torch.min(torch.min(a[0], b[0]), c[0])
        xmax = torch.max(torch.max(a[0], b[0]), c[0])
        ymin = torch.min(torch.min(a[1], b[1]), c[1])
        ymax = torch.max(torch.max(a[1], b[1]), c[1])

        if xmax < -1 or xmin > 1 or ymax < -1 or ymin > 1:
            continue

        def _ndc_to_px_x(x):
            return (((x + 1) * 0.5) * width).long().clamp(0, width - 1)
        def _ndc_to_px_y(y):
            return (((1 - y) * 0.5) * height).long().clamp(0, height - 1)

        x0 = _ndc_to_px_x(xmin); x1 = _ndc_to_px_x(xmax)
        y0 = _ndc_to_px_y(ymax); y1 = _ndc_to_px_y(ymin)
        if x1 <= x0 or y1 <= y0:
            continue

        sub_gx = gx[y0:y1 + 1, x0:x1 + 1]
        sub_gy = gy[y0:y1 + 1, x0:x1 + 1]

        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if denom.abs() < 1e-9:
            continue

        w_a = ((b[1] - c[1]) * (sub_gx - c[0]) + (c[0] - b[0]) * (sub_gy - c[1])) / denom
        w_b = ((c[1] - a[1]) * (sub_gx - c[0]) + (a[0] - c[0]) * (sub_gy - c[1])) / denom
        w_c = 1.0 - w_a - w_b

        inside = (w_a >= 0) & (w_b >= 0) & (w_c >= 0)
        if not inside.any():
            continue

        zz = w_a * za + w_b * zb + w_c * zc
        sub_zbuf = z_buffer[y0:y1 + 1, x0:x1 + 1]
        winning = inside & (zz < sub_zbuf)
        if not winning.any():
            continue

        n_interp = (w_a.unsqueeze(-1) * v_n[ia]
                    + w_b.unsqueeze(-1) * v_n[ib]
                    + w_c.unsqueeze(-1) * v_n[ic])
        n_interp = F.normalize(n_interp, dim=-1)
        ndl = (n_interp * light_dir).sum(dim=-1).clamp(min=0.0)
        shade = ambient + diffuse_gain * ndl
        color = base_color * shade.unsqueeze(-1)

        sub_img = image[y0:y1 + 1, x0:x1 + 1]
        sub_mask = mask[y0:y1 + 1, x0:x1 + 1]
        sub_img[winning] = color[winning]
        sub_mask[winning] = 1.0
        sub_zbuf[winning] = zz[winning]

    bg_t = torch.tensor(bg, device=device, dtype=image.dtype)
    rgb = torch.where(mask.unsqueeze(-1) > 0, image, bg_t.view(1, 1, 3))
    return rgb.clamp(0.0, 1.0), mask


def _pytorch3d_render(
    verts: torch.Tensor, faces: torch.Tensor, normals: torch.Tensor,
    width: int, height: int, camera_distance: float, fov_deg: float,
    light_intensity: float, bg: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    from pytorch3d.renderer import (
        PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, DirectionalLights, TexturesVertex,
    )
    from pytorch3d.structures import Meshes

    device = verts.device
    eye, target = _auto_camera(verts, camera_distance)
    # PyTorch3D uses OpenGL camera convention (+Y up, looking down -Z).
    R = _look_at(eye, target).unsqueeze(0)
    T = (-R @ eye.unsqueeze(-1)).squeeze(-1)  # [1, 3]

    focal = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[focal, focal]], device=device),
        principal_point=torch.zeros(1, 2, device=device),
        R=R, T=T.unsqueeze(0) if T.dim() == 1 else T,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(height, width), blur_radius=0.0, faces_per_pixel=1,
    )
    lights = DirectionalLights(
        direction=[[0.3, 0.5, 1.0]],
        ambient_color=[[0.35, 0.35, 0.35]],
        diffuse_color=[[light_intensity] * 3],
        specular_color=[[0.0, 0.0, 0.0]],
        device=device,
    )
    verts_rgb = torch.full_like(verts, 0.75).unsqueeze(0)  # [1, V, 3]
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    images = renderer(mesh)   # [1, H, W, 4]
    rgb = images[0, ..., :3].clamp(0.0, 1.0)
    alpha = images[0, ..., 3]
    mask = (alpha > 0.0).float()
    bg_t = torch.tensor(bg, device=device, dtype=rgb.dtype)
    rgb = torch.where(mask.unsqueeze(-1) > 0, rgb, bg_t.view(1, 1, 3))
    return rgb, mask


def render_mesh(
    verts: torch.Tensor,       # [V, 3]
    faces: torch.Tensor,       # [F, 3] long
    normals: torch.Tensor,     # [V, 3]
    width: int, height: int,
    camera_distance: float, fov_deg: float,
    light_intensity: float,
    bg: tuple[float, float, float],
    backend: Literal["pytorch3d", "soft_torch"] = "pytorch3d",
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "pytorch3d" and try_import_pytorch3d() is not None:
        try:
            return _pytorch3d_render(verts, faces, normals, width, height,
                                     camera_distance, fov_deg, light_intensity, bg)
        except Exception as e:
            log.warning("pytorch3d render failed (%s); falling back to soft_torch.", e)
    return _soft_torch_render(verts, faces, normals, width, height,
                              camera_distance, fov_deg, light_intensity, bg)


def render_points(
    verts: torch.Tensor,
    width: int,
    height: int,
    camera_distance: float,
    fov_deg: float,
    bg: tuple[float, float, float],
    *,
    point_radius_px: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = verts.device
    aspect = width / height
    eye, target = _auto_camera(verts, camera_distance)
    verts_ndc, z_view = _project_ndc(verts, eye, target, fov_deg, aspect)

    xs = ((verts_ndc[:, 0] + 1.0) * 0.5 * (width - 1)).round().long()
    ys = ((1.0 - verts_ndc[:, 1]) * 0.5 * (height - 1)).round().long()

    # Expand each point over a (2r+1)^2 stamp by adding all dx/dy offsets, then
    # z-reduce per pixel via scatter_reduce_(amin).
    r = int(point_radius_px)
    offsets = torch.arange(-r, r + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.reshape(-1)
    dx = dx.reshape(-1)
    stamp_x = xs.unsqueeze(1) + dx.unsqueeze(0)          # [V, S]
    stamp_y = ys.unsqueeze(1) + dy.unsqueeze(0)          # [V, S]
    stamp_z = z_view.unsqueeze(1).expand_as(stamp_x)     # [V, S]

    valid = (
        (stamp_x >= 0) & (stamp_x < width) &
        (stamp_y >= 0) & (stamp_y < height) &
        torch.isfinite(stamp_z)
    )
    flat_idx = (stamp_y * width + stamp_x).masked_fill(~valid, 0)
    flat_z = stamp_z.masked_fill(~valid, float("inf"))

    z_buffer = torch.full((height * width,), float("inf"), device=device)
    z_buffer.scatter_reduce_(0, flat_idx.reshape(-1), flat_z.reshape(-1), reduce="amin")
    z_buffer = z_buffer.view(height, width)

    mask = (z_buffer < float("inf")).float()
    base_color = torch.tensor([0.78, 0.78, 0.78], device=device)
    image = base_color.view(1, 1, 3).expand(height, width, 3) * mask.unsqueeze(-1)

    bg_t = torch.tensor(bg, device=device, dtype=image.dtype)
    rgb = torch.where(mask.unsqueeze(-1) > 0, image, bg_t.view(1, 1, 3))
    return rgb.clamp(0.0, 1.0), mask
