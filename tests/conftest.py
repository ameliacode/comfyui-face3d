from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_comfy_stubs(tmp_path: Path) -> None:
    if "comfy_api.latest" in sys.modules:
        return

    class _Port:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _CustomType:
        def __init__(self, name):
            self.name = name

        def Input(self, *args, **kwargs):
            return _Port(*args, **kwargs)

        def Output(self, *args, **kwargs):
            return _Port(*args, **kwargs)

    class _NodeOutput(tuple):
        def __new__(cls, *args, **kwargs):
            obj = super().__new__(cls, args)
            obj.ui = kwargs.get("ui")
            return obj

    class _Schema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _InputFactory:
        def __call__(self, *args, **kwargs):
            return _Port(*args, **kwargs)

    io = types.SimpleNamespace(
        Custom=lambda name: _CustomType(name),
        ComfyNode=type("ComfyNode", (), {}),
        NodeOutput=_NodeOutput,
        Schema=_Schema,
        String=types.SimpleNamespace(Input=_InputFactory()),
        Int=types.SimpleNamespace(Input=_InputFactory()),
        Float=types.SimpleNamespace(Input=_InputFactory()),
        Combo=types.SimpleNamespace(Input=_InputFactory()),
        Image=types.SimpleNamespace(Input=_InputFactory(), Output=_InputFactory()),
        Mesh=types.SimpleNamespace(Input=_InputFactory(), Output=_InputFactory()),
        Mask=types.SimpleNamespace(Input=_InputFactory(), Output=_InputFactory()),
        Boolean=types.SimpleNamespace(Input=_InputFactory()),
        Hidden=types.SimpleNamespace(unique_id="unique_id", prompt="prompt", extra_pnginfo="extra_pnginfo"),
        NumberDisplay=types.SimpleNamespace(slider="slider"),
    )
    latest = types.SimpleNamespace(io=io, ComfyExtension=type("ComfyExtension", (), {}))
    comfy_api = types.SimpleNamespace(latest=latest)
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest._util"] = types.SimpleNamespace(
        MESH=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    folder_paths = types.SimpleNamespace(models_dir=str(tmp_path / "models"))
    sys.modules["folder_paths"] = folder_paths

    server = types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=types.SimpleNamespace(
            get=lambda *args, **kwargs: (lambda fn: fn),
            post=lambda *args, **kwargs: (lambda fn: fn),
        )))
    )
    sys.modules["server"] = server

    aiohttp_web = types.SimpleNamespace(Response=object)
    aiohttp = types.SimpleNamespace(web=aiohttp_web)
    sys.modules["aiohttp"] = aiohttp
    sys.modules["aiohttp.web"] = aiohttp_web


@pytest.fixture(autouse=True)
def comfy_stubs(tmp_path):
    _install_comfy_stubs(tmp_path)


_install_comfy_stubs(ROOT / ".pytest-models")


@pytest.fixture
def synthetic_flame_pkl(tmp_path: Path) -> Path:
    pkl_path = tmp_path / "synthetic_flame.pkl"
    v = 12
    f = 8
    shape_dim = 400
    pose_dim = 36
    joints = 5

    rng = np.random.default_rng(1234)
    v_template = rng.normal(0.0, 0.03, size=(v, 3)).astype(np.float32)
    shapedirs = rng.normal(0.0, 0.01, size=(v, 3, shape_dim)).astype(np.float32)
    posedirs = rng.normal(0.0, 0.01, size=(v, 3, pose_dim)).astype(np.float32)
    J_regressor = np.zeros((joints, v), dtype=np.float32)
    for j in range(joints):
        J_regressor[j, j % v] = 1.0
    weights = np.abs(rng.normal(size=(v, joints))).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    kintree_table = np.array([
        [-1, 0, 1, 1, 1],
        [0, 1, 2, 3, 4],
    ], dtype=np.int64)
    faces = np.array([
        [0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4],
        [0, 4, 1], [1, 4, 5], [2, 6, 3], [3, 6, 7],
    ], dtype=np.int64)

    with open(pkl_path, "wb") as handle:
        pickle.dump({
            "v_template": v_template,
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "J_regressor": J_regressor,
            "weights": weights,
            "kintree_table": kintree_table,
            "f": faces,
        }, handle)
    return pkl_path
