from __future__ import annotations

import torch


def _base_params() -> dict:
    return {
        "shape": torch.zeros(1, 100),
        "expression": torch.zeros(1, 50),
        "pose": torch.zeros(1, 6),
        "scale": torch.ones(1, 1),
        "translation": torch.tensor([[0.0, 0.0, 1.5]]),
        "fix_z_trans": False,
    }


def test_expand_pose_fills_neck_and_eyes_with_zero():
    from nodes.flame_params_to_mesh import _expand_pose_6_to_15

    pose6 = torch.tensor([[0.1, 0.2, 0.3, 0.7, 0.8, 0.9]])
    pose15 = _expand_pose_6_to_15(pose6)
    assert pose15.shape == (1, 15)
    assert torch.allclose(pose15[0, :3], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(pose15[0, 3:6], torch.zeros(3))  # neck
    assert torch.allclose(pose15[0, 6:9], torch.tensor([0.7, 0.8, 0.9]))  # jaw
    assert torch.allclose(pose15[0, 9:15], torch.zeros(6))  # eye_L + eye_R


def test_flame_params_to_mesh_uses_synthetic_flame(monkeypatch, synthetic_flame_pkl):
    from nodes import flame_core
    from nodes.flame_params_to_mesh import FLAMEParamsToMesh

    flame_core._CACHE.clear()

    monkeypatch.setattr(
        "nodes.flame_params_to_mesh.ensure_generic_flame_pkl",
        lambda: synthetic_flame_pkl,
    )
    N_VERTS = 12
    N_FACES = 8
    monkeypatch.setattr("nodes.flame_params_to_mesh.N_VERTICES", N_VERTS)

    out = FLAMEParamsToMesh.execute(_base_params())
    mesh = out[0]
    assert mesh.vertices.shape == (1, N_VERTS, 3)
    assert mesh.faces.shape == (1, N_FACES, 3)
    assert torch.isfinite(mesh.vertices).all()
    assert mesh.topology == "mesh"
    assert mesh.fix_z_trans is False


def test_fix_z_trans_true_zeros_translation_z(monkeypatch, synthetic_flame_pkl):
    from nodes import flame_core
    from nodes.flame_params_to_mesh import FLAMEParamsToMesh

    flame_core._CACHE.clear()
    monkeypatch.setattr(
        "nodes.flame_params_to_mesh.ensure_generic_flame_pkl",
        lambda: synthetic_flame_pkl,
    )
    monkeypatch.setattr("nodes.flame_params_to_mesh.N_VERTICES", 12)

    params_fz_false = _base_params()
    params_fz_true = _base_params()
    params_fz_true["fix_z_trans"] = True

    verts_false = FLAMEParamsToMesh.execute(params_fz_false)[0].vertices
    flame_core._CACHE.clear()
    verts_true = FLAMEParamsToMesh.execute(params_fz_true)[0].vertices

    assert not torch.allclose(verts_false, verts_true)
    # z of vertices should differ by exactly translation[z] = 1.5 (scale=1 so trans applied outside).
    dz = (verts_false[..., 2] - verts_true[..., 2]).unique()
    assert torch.allclose(dz, torch.tensor([1.5]))
