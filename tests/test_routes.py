from __future__ import annotations

from routes import _forward_sync, _topology_sync


def test_topology_sync_returns_faces_and_template(monkeypatch, synthetic_flame_pkl):
    monkeypatch.setattr("routes.ensure_flame_assets", lambda gender: synthetic_flame_pkl)
    data = _topology_sync("generic")
    assert data["n_vertices"] > 0
    assert data["n_faces"] > 0
    assert isinstance(data["faces_b64"], str)
    assert isinstance(data["template_b64"], str)


def test_forward_sync_returns_vertices(monkeypatch, synthetic_flame_pkl):
    monkeypatch.setattr("routes.ensure_flame_assets", lambda gender: synthetic_flame_pkl)
    data = _forward_sync({
        "gender": "generic",
        "shape_dim": 50,
        "expr_dim": 50,
        "params": {
            "shape": [0.0] * 50,
            "expr": [0.0] * 50,
            "pose": [0.0] * 15,
            "trans": [0.0, 0.0, 0.0],
        },
    })
    assert data["n_vertices"] > 0
    assert isinstance(data["verts_b64"], str)
