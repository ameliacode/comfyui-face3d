from __future__ import annotations

import torch


class _DummySmirkEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image):
        B = image.shape[0]
        return {
            "shape_params": torch.zeros(B, 300),
            "expression_params": torch.linspace(-0.5, 0.5, steps=B * 50).view(B, 50),
            "pose_params": torch.zeros(B, 3),
            "jaw_params": torch.full((B, 3), 0.1),
            "eyelid_params": torch.zeros(B, 2),
            "cam": torch.zeros(B, 3),
        }


def test_predict_emits_flame_params_with_batched_tensors(monkeypatch):
    from nodes.smirk_predict import SMIRKPredict, _SMIRK_CACHE

    _SMIRK_CACHE.clear()
    monkeypatch.setattr(
        "nodes.smirk_predict._load_smirk_encoder",
        lambda ckpt_path, device, dtype: _DummySmirkEncoder(),
    )
    image = torch.rand(1, 224, 224, 3)
    out = SMIRKPredict.execute(
        {"device": "cpu", "dtype": "fp32", "ckpt_path": "x", "smirk_root": None},
        image,
    )
    params = out[0]
    assert params["shape"].shape == (1, 100)
    assert params["expression"].shape == (1, 50)
    assert params["pose"].shape == (1, 6)
    assert params["scale"].shape == (1, 1)
    assert params["translation"].shape == (1, 3)
    assert params["fix_z_trans"] is False
    assert torch.allclose(params["pose"][:, 3:], torch.full((1, 3), 0.1))
    assert torch.allclose(params["pose"][:, :3], torch.zeros(1, 3))


def test_predict_rejects_multi_batch(monkeypatch):
    import pytest
    from nodes.smirk_predict import SMIRKPredict, _SMIRK_CACHE

    _SMIRK_CACHE.clear()
    monkeypatch.setattr(
        "nodes.smirk_predict._load_smirk_encoder",
        lambda ckpt_path, device, dtype: _DummySmirkEncoder(),
    )
    image = torch.rand(2, 224, 224, 3)
    with pytest.raises(RuntimeError, match="batch size 1"):
        SMIRKPredict.execute(
            {"device": "cpu", "dtype": "fp32", "ckpt_path": "x", "smirk_root": None},
            image,
        )
