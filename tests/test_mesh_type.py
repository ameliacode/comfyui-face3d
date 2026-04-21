from __future__ import annotations

import pytest
import torch


def _tetra() -> tuple[torch.Tensor, torch.Tensor]:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    return verts, faces


def test_coerce_mesh_accepts_batched_tensors():
    from nodes.mesh_types import coerce_mesh

    verts, faces = _tetra()
    payload = {"vertices": verts.unsqueeze(0), "faces": faces.unsqueeze(0)}
    v, f = coerce_mesh(payload)
    assert v.shape == (4, 3)
    assert f.shape == (4, 3)
    assert v.dtype == torch.float32
    assert f.dtype == torch.long


def test_coerce_mesh_accepts_object_with_attrs():
    from types import SimpleNamespace

    from nodes.mesh_types import coerce_mesh

    verts, faces = _tetra()
    mesh_obj = SimpleNamespace(vertices=verts.unsqueeze(0), faces=faces.unsqueeze(0))
    v, f = coerce_mesh(mesh_obj)
    assert v.shape == (4, 3)
    assert f.shape == (4, 3)


def test_coerce_mesh_rejects_missing_keys():
    from nodes.mesh_types import coerce_mesh

    with pytest.raises(ValueError, match="vertices"):
        coerce_mesh({"faces": torch.zeros(1, 3, dtype=torch.long)})


def test_coerce_mesh_rejects_bad_shapes():
    from nodes.mesh_types import coerce_mesh

    with pytest.raises(ValueError, match=r"vertices must have shape"):
        coerce_mesh({"vertices": torch.zeros(4, 2), "faces": torch.zeros(1, 3, dtype=torch.long)})
    with pytest.raises(ValueError, match=r"faces must have shape"):
        coerce_mesh({"vertices": torch.zeros(4, 3), "faces": torch.zeros(1, 4, dtype=torch.long)})


def test_compute_vertex_normals_unit_length_and_outward():
    from nodes.mesh_types import compute_vertex_normals

    verts, faces = _tetra()
    normals = compute_vertex_normals(verts, faces)
    assert normals.shape == verts.shape
    norms = normals.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    centroid = verts.mean(dim=0)
    outward = (verts - centroid)
    dots = (normals * outward).sum(dim=1)
    assert (dots > 0).all(), f"normals should point outward from centroid, got {dots}"


def test_compute_vertex_normals_accepts_numpy_like():
    from nodes.mesh_types import compute_vertex_normals

    verts, faces = _tetra()
    normals = compute_vertex_normals(verts.tolist(), faces.tolist())
    assert normals.shape == (4, 3)
