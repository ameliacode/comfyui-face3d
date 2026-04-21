from __future__ import annotations

from nodes.flame_params import default_params_dict, validate_params_dict


def test_default_params_lengths():
    params = default_params_dict(4, 6)
    assert len(params["shape"]) == 4
    assert len(params["expr"]) == 6
    assert len(params["pose"]) == 15
    assert len(params["trans"]) == 3


def test_validate_params_truncates_and_pads():
    flame_model = {"shape_dim": 3, "expr_dim": 2}
    params = validate_params_dict(
        {"shape": [1, 2, 3, 4], "expr": [5], "pose": [1], "trans": [2, 3, 4, 5]},
        flame_model,
    )
    assert params["shape"] == [1.0, 2.0, 3.0]
    assert params["expr"] == [5.0, 0.0]
    assert params["pose"][:3] == [1.0, 0.0, 0.0]
    assert params["trans"] == [2.0, 3.0, 4.0]
