from __future__ import annotations

from pathlib import Path

import pytest


def _make_smirk_root(tmp_path: Path) -> Path:
    root = tmp_path / "smirk_checkout"
    (root / "src").mkdir(parents=True)
    (root / "src" / "smirk_encoder.py").write_text("class SmirkEncoder:\n    pass\n")
    return root


def test_resolve_smirk_root_env_var_wins(tmp_path, monkeypatch):
    from nodes.smirk_runtime import SMIRK_ENV_VAR, resolve_smirk_root

    root = _make_smirk_root(tmp_path)
    monkeypatch.setenv(SMIRK_ENV_VAR, str(root))

    assert resolve_smirk_root() == root


def test_resolve_smirk_root_required_raises_with_url(tmp_path, monkeypatch):
    from nodes import smirk_runtime

    monkeypatch.delenv(smirk_runtime.SMIRK_ENV_VAR, raising=False)
    monkeypatch.setattr(smirk_runtime, "SMIRK_CANDIDATES", [tmp_path / "nope"])
    monkeypatch.setattr(smirk_runtime, "_resolve_installed_root", lambda: None)

    with pytest.raises(RuntimeError) as exc:
        smirk_runtime.resolve_smirk_root()
    msg = str(exc.value)
    assert "SMIRK runtime is not available" in msg
    assert "github.com/georgeretsi/smirk" in msg


def test_resolve_smirk_root_optional_returns_none(tmp_path, monkeypatch):
    from nodes import smirk_runtime

    monkeypatch.delenv(smirk_runtime.SMIRK_ENV_VAR, raising=False)
    monkeypatch.setattr(smirk_runtime, "SMIRK_CANDIDATES", [tmp_path / "nope"])
    monkeypatch.setattr(smirk_runtime, "_resolve_installed_root", lambda: None)

    assert smirk_runtime.resolve_smirk_root(required=False) is None


def test_is_smirk_root_checks_encoder_file(tmp_path):
    from nodes.smirk_runtime import _is_smirk_root

    assert _is_smirk_root(_make_smirk_root(tmp_path)) is True
    assert _is_smirk_root(tmp_path / "empty") is False
