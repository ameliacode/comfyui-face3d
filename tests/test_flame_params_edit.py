from __future__ import annotations

import pytest
import torch


def _kaolrm_params(*, fix_z: bool = True) -> dict:
    return {
        "shape": torch.full((1, 100), 0.1),
        "expression": torch.full((1, 50), 0.2),
        "pose": torch.cat([torch.full((1, 3), 0.3), torch.full((1, 3), 0.4)], dim=1),
        "scale": torch.full((1, 1), 1.25),
        "translation": torch.tensor([[0.1, 0.2, 0.3]]),
        "fix_z_trans": fix_z,
    }


def _smirk_params() -> dict:
    return {
        "shape": torch.zeros(1, 100),
        "expression": torch.full((1, 50), 0.7),
        "pose": torch.cat([torch.zeros(1, 3), torch.full((1, 3), 0.9)], dim=1),
        "scale": torch.ones(1, 1),
        "translation": torch.zeros(1, 3),
        "fix_z_trans": False,
    }


def test_edit_passthrough_without_override_or_edits():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    out = FLAMEParamsEdit.execute(k)
    edited = out[0]
    assert torch.allclose(edited["shape"], k["shape"])
    assert torch.allclose(edited["expression"], k["expression"])
    assert torch.allclose(edited["pose"], k["pose"])
    assert torch.allclose(edited["scale"], k["scale"])
    assert torch.allclose(edited["translation"], k["translation"])
    assert edited["fix_z_trans"] is True


def test_edit_applies_merge_policy_when_override_present():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    s = _smirk_params()
    out = FLAMEParamsEdit.execute(k, s)
    merged = out[0]
    assert torch.allclose(merged["shape"], k["shape"])
    assert torch.allclose(merged["expression"], s["expression"])
    assert torch.allclose(merged["pose"][:, :3], k["pose"][:, :3])
    assert torch.allclose(merged["pose"][:, 3:], s["pose"][:, 3:])
    assert torch.allclose(merged["scale"], k["scale"])
    assert torch.allclose(merged["translation"], k["translation"])
    assert merged["fix_z_trans"] is True


def test_edit_strength_sliders_scale_coefficients():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    out = FLAMEParamsEdit.execute(
        k,
        shape_strength=0.5,
        expression_strength=0.0,
        jaw_strength=0.25,
    )
    edited = out[0]
    assert torch.allclose(edited["shape"], k["shape"] * 0.5)
    assert torch.allclose(edited["expression"], torch.zeros_like(k["expression"]))
    assert torch.allclose(edited["pose"][:, :3], k["pose"][:, :3])
    assert torch.allclose(edited["pose"][:, 3:], k["pose"][:, 3:] * 0.25)


def test_edit_offsets_are_additive():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    out = FLAMEParamsEdit.execute(
        k,
        global_pose_offset_x=0.1,
        global_pose_offset_y=-0.2,
        global_pose_offset_z=0.05,
        translation_offset_x=0.5,
        translation_offset_y=-0.5,
        translation_offset_z=0.25,
        scale_multiplier=2.0,
    )
    edited = out[0]
    expected_global = k["pose"][:, :3] + torch.tensor([[0.1, -0.2, 0.05]])
    assert torch.allclose(edited["pose"][:, :3], expected_global)
    assert torch.allclose(edited["pose"][:, 3:], k["pose"][:, 3:])
    assert torch.allclose(edited["translation"], k["translation"] + torch.tensor([[0.5, -0.5, 0.25]]))
    assert torch.allclose(edited["scale"], k["scale"] * 2.0)


def test_edit_fix_z_trans_override_force_true_and_false():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params(fix_z=False)
    assert FLAMEParamsEdit.execute(k)[0]["fix_z_trans"] is False
    assert FLAMEParamsEdit.execute(k, fix_z_trans_override="force_true")[0]["fix_z_trans"] is True
    assert FLAMEParamsEdit.execute(k, fix_z_trans_override="force_false")[0]["fix_z_trans"] is False
    assert FLAMEParamsEdit.execute(k, fix_z_trans_override="inherit")[0]["fix_z_trans"] is False


def test_edit_rejects_flat_inputs():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    k["shape"] = torch.zeros(100)
    with pytest.raises(RuntimeError, match=r"\[B, 100\]"):
        FLAMEParamsEdit.execute(k)


def test_edit_rejects_missing_fix_z_trans():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    del k["fix_z_trans"]
    with pytest.raises(RuntimeError, match="fix_z_trans"):
        FLAMEParamsEdit.execute(k)


def test_edit_rejects_batch_mismatch():
    from nodes.flame_params_edit import FLAMEParamsEdit

    k = _kaolrm_params()
    s = _smirk_params()
    s = {key: (v.repeat(2, 1) if isinstance(v, torch.Tensor) else v) for key, v in s.items()}
    with pytest.raises(RuntimeError, match="batch mismatch"):
        FLAMEParamsEdit.execute(k, s)


def test_edit_rejects_unknown_fix_z_trans_override():
    from nodes.flame_params_edit import FLAMEParamsEdit

    with pytest.raises(RuntimeError, match="fix_z_trans_override"):
        FLAMEParamsEdit.execute(_kaolrm_params(), fix_z_trans_override="bogus")
