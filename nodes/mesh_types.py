"""Mesh payload coercion + normal computation helpers for the KaoLRM pipeline."""
from __future__ import annotations

from typing import Any

import torch


def _as_tensor(value: Any, *, dtype=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        return t.to(dtype=dtype) if dtype is not None else t
    return torch.as_tensor(value, dtype=dtype)


def coerce_mesh(mesh: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(mesh, dict):
        v_src, f_src = mesh.get("vertices"), mesh.get("faces")
    else:
        v_src, f_src = getattr(mesh, "vertices", None), getattr(mesh, "faces", None)
    if v_src is None or f_src is None:
        raise ValueError("mesh must expose 'vertices' and 'faces'")

    vertices = _as_tensor(v_src, dtype=torch.float32)
    faces = _as_tensor(f_src, dtype=torch.long)

    if vertices.ndim == 3 and vertices.shape[0] >= 1:
        vertices = vertices[0]
    if faces.ndim == 3 and faces.shape[0] >= 1:
        faces = faces[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape [V, 3], got {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {tuple(faces.shape)}")
    return vertices.contiguous(), faces.contiguous()


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    vertices = _as_tensor(vertices, dtype=torch.float32)
    faces = _as_tensor(faces, dtype=torch.long)
    if faces.numel() == 0:
        return torch.zeros_like(vertices)
    normals = torch.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    for i in range(3):
        normals.index_add_(0, faces[:, i], face_normals)
    return torch.nn.functional.normalize(normals, dim=1, eps=1e-8)
