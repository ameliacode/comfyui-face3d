from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_resolve_device_auto_matches_cuda_availability():
    from nodes.kaolrm_load import resolve_device

    got = resolve_device("auto")
    assert got == ("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_dtype_cpu_forces_fp32():
    from nodes.kaolrm_load import resolve_dtype

    assert resolve_dtype("fp16", "cpu") == "fp32"
    assert resolve_dtype("bf16", "cpu") == "fp32"
    assert resolve_dtype("auto", "cpu") == "fp32"


def test_resolve_dtype_cuda_defaults_fp16_on_auto():
    from nodes.kaolrm_load import resolve_dtype

    assert resolve_dtype("auto", "cuda") == "fp16"
    assert resolve_dtype("bf16", "cuda") == "bf16"
    assert resolve_dtype("fp32", "cuda") == "fp32"


def test_ensure_kaolrm_weights_missing_raises_with_path(tmp_path, monkeypatch):
    import folder_paths
    import nodes.kaolrm_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        mod.ensure_kaolrm_weights("mono")
    msg = str(exc.value)
    assert "mono.safetensors" in msg
    assert str(tmp_path) in msg


def test_ensure_kaolrm_weights_returns_path_when_present(tmp_path, monkeypatch):
    import folder_paths
    import nodes.kaolrm_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    weights = tmp_path / "kaolrm" / "mono.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"stub")

    got = mod.ensure_kaolrm_weights("mono")
    assert got == weights


def test_ensure_generic_flame_pkl_missing_raises(tmp_path, monkeypatch):
    import folder_paths
    import nodes.kaolrm_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError, match="generic_model.pkl"):
        mod.ensure_generic_flame_pkl()


def test_execute_requires_non_commercial_acknowledgement():
    from nodes.kaolrm_load import LoadKaoLRM

    with pytest.raises(RuntimeError, match="non_commercial"):
        LoadKaoLRM.execute(variant="mono", i_understand_non_commercial=False)


def test_execute_does_not_require_runtime_checkout(tmp_path, monkeypatch):
    import folder_paths
    import nodes.kaolrm_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    weights = tmp_path / "kaolrm" / "mono.safetensors"
    flame = tmp_path / "flame" / "generic_model.pkl"
    weights.parent.mkdir(parents=True)
    flame.parent.mkdir(parents=True)
    weights.write_bytes(b"stub")
    flame.write_bytes(b"stub")
    monkeypatch.setattr(mod, "resolve_kaolrm_root", lambda required=False: None)

    out = mod.LoadKaoLRM.execute(variant="mono", i_understand_non_commercial=True)
    descriptor = out[0]
    assert descriptor["ckpt_path"] == str(weights)
    assert descriptor["flame_pkl_path"] == str(flame)
    assert descriptor["kaolrm_root"] is None
