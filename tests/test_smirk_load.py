from __future__ import annotations

import pytest


def test_execute_requires_non_commercial_acknowledgement():
    from nodes.smirk_load import LoadSMIRK

    with pytest.raises(RuntimeError, match="non_commercial"):
        LoadSMIRK.execute(i_understand_non_commercial=False)


def test_ensure_smirk_weights_missing_raises_with_path(tmp_path, monkeypatch):
    import folder_paths
    import nodes.smirk_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        mod.ensure_smirk_weights()
    msg = str(exc.value)
    assert "SMIRK_em1.pt" in msg
    assert str(tmp_path) in msg
    assert "drive.google.com" in msg


def test_execute_returns_descriptor_when_weights_present(tmp_path, monkeypatch):
    import folder_paths
    import nodes.smirk_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    weights = tmp_path / "smirk" / "SMIRK_em1.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"stub")
    monkeypatch.setattr(mod, "resolve_smirk_root", lambda required=False: None)

    out = mod.LoadSMIRK.execute(device="cpu", dtype="fp32", i_understand_non_commercial=True)
    descriptor = out[0]
    assert descriptor["ckpt_path"] == str(weights)
    assert descriptor["device"] == "cpu"
    assert descriptor["dtype"] == "fp32"
    assert descriptor["smirk_root"] is None
