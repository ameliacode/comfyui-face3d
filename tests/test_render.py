from __future__ import annotations

from nodes.flame_render import FlameRender


def test_render_node_outputs_image_and_mask(monkeypatch, synthetic_flame_pkl):
    flame_model = {
        "gender": "generic",
        "pkl_path": str(synthetic_flame_pkl),
        "device": "cpu",
        "shape_dim": 50,
        "expr_dim": 50,
    }
    flame_params = {
        "shape": [0.0] * 50,
        "expr": [0.0] * 50,
        "pose": [0.0] * 15,
        "trans": [0.0, 0.0, 0.0],
    }
    out = FlameRender.execute(
        flame_model,
        flame_params,
        width=64,
        height=64,
        renderer="soft_torch",
    )
    image, mask = tuple(out)
    assert image.shape == (1, 64, 64, 3)
    assert mask.shape == (1, 64, 64)
