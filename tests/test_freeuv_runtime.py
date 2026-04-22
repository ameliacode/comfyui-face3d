from __future__ import annotations

from pathlib import Path

import pytest


def _make_freeuv_root(tmp_path: Path) -> Path:
    root = tmp_path / "freeuv_checkout"
    (root / "detail_encoder").mkdir(parents=True)
    (root / "detail_encoder" / "__init__.py").write_text("class detail_encoder: pass\n")
    return root


def test_resolve_freeuv_root_env_var_wins(tmp_path, monkeypatch):
    from nodes.freeuv_runtime import FREEUV_ENV_VAR, resolve_freeuv_root

    root = _make_freeuv_root(tmp_path)
    monkeypatch.setenv(FREEUV_ENV_VAR, str(root))

    assert resolve_freeuv_root() == root


def test_resolve_freeuv_root_required_raises_with_url(tmp_path, monkeypatch):
    from nodes import freeuv_runtime

    monkeypatch.delenv(freeuv_runtime.FREEUV_ENV_VAR, raising=False)
    monkeypatch.setattr(freeuv_runtime, "FREEUV_CANDIDATES", [tmp_path / "nope"])
    monkeypatch.setattr(freeuv_runtime, "_resolve_installed_root", lambda: None)

    with pytest.raises(RuntimeError) as exc:
        freeuv_runtime.resolve_freeuv_root()
    msg = str(exc.value)
    assert "FreeUV runtime is not available" in msg
    assert "github.com/YangXingchao/FreeUV" in msg
    assert "FREEUV_ROOT" in msg


def test_resolve_freeuv_root_optional_returns_none(tmp_path, monkeypatch):
    from nodes import freeuv_runtime

    monkeypatch.delenv(freeuv_runtime.FREEUV_ENV_VAR, raising=False)
    monkeypatch.setattr(freeuv_runtime, "FREEUV_CANDIDATES", [tmp_path / "nope"])
    monkeypatch.setattr(freeuv_runtime, "_resolve_installed_root", lambda: None)

    assert freeuv_runtime.resolve_freeuv_root(required=False) is None


def test_is_freeuv_root_checks_detail_encoder_package(tmp_path):
    from nodes.freeuv_runtime import _is_freeuv_root

    assert _is_freeuv_root(_make_freeuv_root(tmp_path)) is True
    assert _is_freeuv_root(tmp_path / "empty") is False


def test_ensure_freeuv_on_path_injects_vendor_dir(tmp_path, monkeypatch):
    import sys

    from nodes import freeuv_runtime

    root = _make_freeuv_root(tmp_path)
    monkeypatch.setenv(freeuv_runtime.FREEUV_ENV_VAR, str(root))
    if str(root) in sys.path:
        sys.path.remove(str(root))

    returned = freeuv_runtime.ensure_freeuv_on_path()

    assert returned == root
    assert str(root) in sys.path
    # cleanup
    sys.path.remove(str(root))
