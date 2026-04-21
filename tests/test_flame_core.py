from __future__ import annotations

import torch

from nodes.flame_core import FlameCore


def test_flame_core_forward_shapes(synthetic_flame_pkl):
    core = FlameCore(synthetic_flame_pkl, device="cpu")
    shape = torch.zeros(1, 300)
    expr = torch.zeros(1, 100)
    pose = torch.zeros(1, 15)
    trans = torch.zeros(1, 3)

    verts = core.forward(shape, expr, pose, trans)
    normals = core.compute_vertex_normals(verts[0])

    assert verts.shape == (1, core.n_vertices, 3)
    assert normals.shape == (core.n_vertices, 3)
    assert torch.isfinite(verts).all()
    assert torch.isfinite(normals).all()


def test_flame_core_translation_changes_vertices(synthetic_flame_pkl):
    core = FlameCore(synthetic_flame_pkl, device="cpu")
    shape = torch.zeros(1, 300)
    expr = torch.zeros(1, 100)
    pose = torch.zeros(1, 15)
    trans = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)

    verts = core.forward(shape, expr, pose, trans)
    baseline = core.forward(shape, expr, pose, torch.zeros_like(trans))

    assert not torch.allclose(verts, baseline)
