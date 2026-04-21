from __future__ import annotations

from types import SimpleNamespace

import torch


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.flame_model = SimpleNamespace(
            faces_tensor=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
        )

    def forward_planes(self, image, source_camera):
        return torch.ones(1, 4)

    def flame_decoder(self, planes):
        return {
            "shape": torch.zeros(1, 100),
            "expression": torch.zeros(1, 50),
            "pose": torch.zeros(1, 15),
            "scale": torch.ones(1, 1),
            "translation": torch.zeros(1, 3),
        }

    def flame2mesh(self, decoded_params, num_sampling, fix_z_trans):
        vertices = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]],
            dtype=torch.float32,
        )
        sampled = torch.linspace(0.0, 1.0, num_sampling * 3, dtype=torch.float32).view(1, num_sampling, 3)
        return vertices, torch.zeros(1, 4, 3), sampled


def test_reconstruct_uses_sampled_vertices_when_requested(monkeypatch):
    from nodes.kaolrm_reconstruct import KaoLRMReconstruct

    model = _DummyModel()
    monkeypatch.setattr("nodes.kaolrm_reconstruct.import_kaolrm_symbols", lambda: {
        "create_intrinsics": lambda f, c, device: torch.eye(3, dtype=torch.float32, device=device),
        "build_camera_principle": lambda extrinsics, intrinsics: torch.zeros(1, 16, dtype=torch.float32, device=extrinsics.device),
    })
    monkeypatch.setattr("nodes.kaolrm_reconstruct._get_cached_model", lambda payload: model)

    image = torch.rand(1, 224, 224, 3)
    out = KaoLRMReconstruct.execute(
        {"variant": "mono", "device": "cpu", "dtype": "fp32", "ckpt_path": "x", "flame_pkl_path": "y"},
        image,
        num_sampling=7,
    )
    mesh = out[0]
    assert mesh.vertices.shape == (1, 7, 3)
    assert mesh.faces.shape == (1, 0, 3)
    assert mesh.topology == "point_cloud"
    assert mesh.base_vertices.shape == (1, 4, 3)
    assert mesh.base_faces.shape == (1, 2, 3)


def test_reconstruct_keeps_mesh_topology_at_flame_resolution(monkeypatch):
    from nodes.kaolrm_reconstruct import FLAME_VERT_COUNT, KaoLRMReconstruct

    model = _DummyModel()
    monkeypatch.setattr("nodes.kaolrm_reconstruct.import_kaolrm_symbols", lambda: {
        "create_intrinsics": lambda f, c, device: torch.eye(3, dtype=torch.float32, device=device),
        "build_camera_principle": lambda extrinsics, intrinsics: torch.zeros(1, 16, dtype=torch.float32, device=extrinsics.device),
    })
    monkeypatch.setattr("nodes.kaolrm_reconstruct._get_cached_model", lambda payload: model)

    image = torch.rand(1, 224, 224, 3)
    out = KaoLRMReconstruct.execute(
        {"variant": "mono", "device": "cpu", "dtype": "fp32", "ckpt_path": "x", "flame_pkl_path": "y"},
        image,
        num_sampling=FLAME_VERT_COUNT,
    )
    mesh = out[0]
    assert mesh.vertices.shape == (1, 4, 3)
    assert mesh.faces.shape == (1, 2, 3)
    assert mesh.topology == "mesh"
