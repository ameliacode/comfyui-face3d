from __future__ import annotations

from pathlib import Path


def test_resolve_kaolrm_root_accepts_installed_package(monkeypatch, tmp_path):
    import importlib.machinery
    import nodes.kaolrm_runtime as mod

    root = tmp_path / "site-packages"
    package_dir = root / "kaolrm"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")

    spec = importlib.machinery.ModuleSpec("kaolrm", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]

    monkeypatch.delenv(mod.KAOLRM_ENV_VAR, raising=False)
    monkeypatch.setattr(mod, "KAOLRM_CANDIDATES", [])
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: spec if name == "kaolrm" else None)

    assert mod.resolve_kaolrm_root() == root


def test_resolve_kaolrm_root_rejects_missing_runtime(monkeypatch):
    import nodes.kaolrm_runtime as mod

    monkeypatch.delenv(mod.KAOLRM_ENV_VAR, raising=False)
    monkeypatch.setattr(mod, "KAOLRM_CANDIDATES", [])
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)

    assert mod.resolve_kaolrm_root(required=False) is None
