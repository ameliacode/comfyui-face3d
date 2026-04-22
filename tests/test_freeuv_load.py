from __future__ import annotations

from pathlib import Path

import pytest


def _populate_freeuv_weights(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "freeuv"
    root.mkdir(parents=True)
    sd15 = root / "sd15"
    sd15.mkdir()
    (sd15 / "model_index.json").write_text("{}")
    clip = root / "image_encoder_l"
    clip.mkdir()
    (clip / "config.json").write_text("{}")
    aligner = root / "uv_structure_aligner.bin"
    aligner.write_bytes(b"stub")
    detail = root / "flaw_tolerant_facial_detail_extractor.bin"
    detail.write_bytes(b"stub")
    return {"sd15": sd15, "clip": clip, "aligner": aligner, "detail": detail}


def test_execute_requires_non_commercial_acknowledgement():
    from nodes.freeuv_load import LoadFreeUV

    with pytest.raises(RuntimeError, match="non_commercial"):
        LoadFreeUV.execute(i_understand_non_commercial=False)


def test_ensure_sd15_missing_lists_expected_subdirs(tmp_path, monkeypatch):
    import folder_paths
    import nodes.freeuv_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        mod._ensure_sd15_snapshot()
    msg = str(exc.value)
    assert "stable-diffusion-v1-5/stable-diffusion-v1-5" in msg
    assert str(tmp_path / "freeuv" / "sd15") in msg
    for subdir in ("unet/", "vae/", "text_encoder/", "tokenizer/", "scheduler/",
                   "feature_extractor/", "safety_checker/", "model_index.json"):
        assert subdir in msg


def test_ensure_clip_missing_names_repo(tmp_path, monkeypatch):
    import folder_paths
    import nodes.freeuv_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        mod._ensure_clip_snapshot()
    msg = str(exc.value)
    assert "openai/clip-vit-large-patch14" in msg
    assert str(tmp_path / "freeuv" / "image_encoder_l") in msg


def test_ensure_aligner_missing_names_path_and_url(tmp_path, monkeypatch):
    import folder_paths
    import nodes.freeuv_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        mod._ensure_freeuv_weight("uv_structure_aligner.bin")
    msg = str(exc.value)
    assert "uv_structure_aligner.bin" in msg
    assert "github.com/YangXingchao/FreeUV/releases" in msg


def test_execute_returns_descriptor_when_all_weights_present(tmp_path, monkeypatch):
    import folder_paths
    import nodes.freeuv_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    paths = _populate_freeuv_weights(tmp_path)
    monkeypatch.setattr(mod, "resolve_freeuv_root", lambda required=False: None)

    out = mod.LoadFreeUV.execute(device="cpu", dtype="fp32", i_understand_non_commercial=True)
    descriptor = out[0]
    assert descriptor["sd15_root"] == str(paths["sd15"])
    assert descriptor["clip_root"] == str(paths["clip"])
    assert descriptor["aligner_path"] == str(paths["aligner"])
    assert descriptor["detail_path"] == str(paths["detail"])
    assert descriptor["device"] == "cpu"
    assert descriptor["dtype"] == "fp32"
    assert descriptor["freeuv_root"] is None


def test_execute_forces_fp32_on_cpu(tmp_path, monkeypatch):
    import folder_paths
    import nodes.freeuv_load as mod

    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    _populate_freeuv_weights(tmp_path)
    monkeypatch.setattr(mod, "resolve_freeuv_root", lambda required=False: None)

    out = mod.LoadFreeUV.execute(device="cpu", dtype="fp16", i_understand_non_commercial=True)
    assert out[0]["dtype"] == "fp32"
