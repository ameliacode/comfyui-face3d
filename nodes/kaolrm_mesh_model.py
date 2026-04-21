"""Mesh-only KaoLRM model wrapper without renderer dependencies."""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .flame_core import _install_chumpy_shims
from .kaolrm_runtime import ensure_kaolrm_on_path

log = logging.getLogger(__name__)

VENDOR_PACKAGE = "kaolrm_mesh_vendor"


def _ensure_vendor_packages(kaolrm_root: str) -> Path:
    root = Path(kaolrm_root)
    models_dir = root / "kaolrm" / "models"
    encoders_dir = models_dir / "encoders"
    if VENDOR_PACKAGE not in sys.modules:
        pkg = types.ModuleType(VENDOR_PACKAGE)
        pkg.__path__ = [str(models_dir)]
        sys.modules[VENDOR_PACKAGE] = pkg
    enc_name = f"{VENDOR_PACKAGE}.encoders"
    if enc_name not in sys.modules:
        enc_pkg = types.ModuleType(enc_name)
        enc_pkg.__path__ = [str(encoders_dir)]
        sys.modules[enc_name] = enc_pkg
    return models_dir


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_module(kaolrm_root: str, relative_path: str, module_suffix: str):
    models_dir = _ensure_vendor_packages(kaolrm_root)
    return _load_module(f"{VENDOR_PACKAGE}.{module_suffix}", models_dir / relative_path)


def _encoder_fn(encoder_type: str):
    encoder_type = encoder_type.lower()
    if encoder_type == "dino":
        return importlib.import_module(f"{VENDOR_PACKAGE}.encoders.dino_wrapper").DinoWrapper
    if encoder_type == "dinov2":
        return importlib.import_module(f"{VENDOR_PACKAGE}.encoders.dinov2_wrapper").Dinov2Wrapper
    raise ValueError(f"Unsupported KaoLRM encoder type: {encoder_type}")


class KaoLRMMesh(nn.Module):
    def __init__(self, config: dict, *, flame_pkl_path: str, kaolrm_root: str):
        super().__init__()
        ensure_kaolrm_on_path()
        _ensure_vendor_packages(kaolrm_root)
        _load_source_module(kaolrm_root, "block.py", "block")
        embedder_mod = _load_source_module(kaolrm_root, "embedder.py", "embedder")
        transformer_mod = _load_source_module(kaolrm_root, "transformer.py", "transformer")
        flame_decoder_mod = _load_source_module(kaolrm_root, "flame_decoder.py", "flame_decoder")
        flame_mod = _load_source_module(kaolrm_root, "flame.py", "flame")
        _load_module(
            f"{VENDOR_PACKAGE}.encoders.dino_wrapper",
            Path(kaolrm_root) / "kaolrm" / "models" / "encoders" / "dino_wrapper.py",
        )
        _load_module(
            f"{VENDOR_PACKAGE}.encoders.dinov2_wrapper",
            Path(kaolrm_root) / "kaolrm" / "models" / "encoders" / "dinov2_wrapper.py",
        )

        self.encoder_feat_dim = int(config["encoder_feat_dim"])
        self.camera_embed_dim = int(config["camera_embed_dim"])
        self.triplane_low_res = int(config["triplane_low_res"])
        self.triplane_high_res = int(config["triplane_high_res"])
        self.triplane_dim = int(config["triplane_dim"])

        self.encoder = _encoder_fn(config["encoder_type"])(
            model_name=config["encoder_model_name"],
            freeze=bool(config.get("encoder_freeze", False)),
        )
        self.camera_embedder = embedder_mod.CameraEmbedder(raw_dim=12 + 4, embed_dim=self.camera_embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 3 * self.triplane_low_res**2, int(config["transformer_dim"]))
            * (1.0 / int(config["transformer_dim"])) ** 0.5
        )
        self.transformer = transformer_mod.TransformerDecoder(
            block_type="cond_mod",
            num_layers=int(config["transformer_layers"]),
            num_heads=int(config["transformer_heads"]),
            inner_dim=int(config["transformer_dim"]),
            cond_dim=self.encoder_feat_dim,
            mod_dim=self.camera_embed_dim,
        )
        self.upsampler = nn.ConvTranspose2d(
            int(config["transformer_dim"]),
            self.triplane_dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.flame_decoder = flame_decoder_mod.FLAMEDecoder(self.triplane_dim, self.triplane_high_res)
        _install_chumpy_shims()
        flame_cfg = SimpleNamespace(
            flame_model_path=str(flame_pkl_path),
            flame_lmk_embedding_path=str(Path(kaolrm_root) / "data" / "landmark_embedding.npy"),
            n_shape=100,
            n_exp=50,
        )
        self.flame_model = flame_mod.FLAME(flame_cfg)
        self.sample_points_from_meshes = flame_mod.sample_points_from_meshes

    def forward_transformer(self, image_feats: torch.Tensor, camera_embeddings: torch.Tensor) -> torch.Tensor:
        n = image_feats.shape[0]
        x = self.pos_embed.repeat(n, 1, 1)
        return self.transformer(x, cond=image_feats, mod=camera_embeddings)

    def reshape_upsample(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        h = w = self.triplane_low_res
        x = tokens.view(n, 3, h, w, -1)
        x = torch.einsum("nihwd->indhw", x)
        x = x.contiguous().view(3 * n, -1, h, w)
        x = self.upsampler(x)
        x = x.view(3, n, *x.shape[-3:])
        x = torch.einsum("indhw->nidhw", x)
        return x.contiguous()

    def forward_planes(self, image: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        image_feats = self.encoder(image)
        camera_embeddings = self.camera_embedder(camera)
        tokens = self.forward_transformer(image_feats, camera_embeddings)
        return self.reshape_upsample(tokens)

    def flame2mesh(self, decoded_params: dict[str, torch.Tensor], num_sampling: int, fix_z_trans: bool):
        shape_params = decoded_params["shape"]
        expression_params = decoded_params["expression"]
        pose_params = decoded_params["pose"]
        scale = decoded_params["scale"]
        translation = decoded_params["translation"]

        if fix_z_trans:
            translation = translation.clone()
            translation[:, -1] = 0.0

        vertices, _, lmk3d = self.flame_model(shape_params, expression_params, pose_params)
        vertices = vertices * scale.unsqueeze(2) + translation.unsqueeze(1)
        lmk3d = lmk3d * scale.unsqueeze(2) + translation.unsqueeze(1)
        if num_sampling != 5023:
            faces = self.flame_model.faces_tensor.repeat(vertices.shape[0], 1, 1)
            vertices_render = self.sample_points_from_meshes(vertices, faces, num_sampling)
        else:
            vertices_render = vertices
        return vertices.float(), lmk3d, vertices_render


def load_kaolrm_release_config(kaolrm_root: str, variant: str) -> dict:
    config_path = Path(kaolrm_root) / "releases" / variant / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing KaoLRM config.json at '{config_path}'.")
    return json.loads(config_path.read_text())


def load_mesh_only_model(
    *,
    kaolrm_root: str,
    variant: str,
    ckpt_path: str,
    flame_pkl_path: str,
    device: str,
    dtype: str,
) -> KaoLRMMesh:
    try:
        from accelerate import PartialState

        PartialState()
    except ImportError:
        log.debug("accelerate.PartialState unavailable; skipping distributed init.")
    config = load_kaolrm_release_config(kaolrm_root, variant)
    model = KaoLRMMesh(config, flame_pkl_path=flame_pkl_path, kaolrm_root=kaolrm_root)
    state = load_file(str(ckpt_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"KaoLRM variant '{variant}' checkpoint at '{ckpt_path}' is missing "
            f"{len(missing)} required weights (first: {preview}). "
            f"Unexpected keys: {len(unexpected)}. "
            "Likely a variant/config mismatch."
        )
    del unexpected

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    model = model.to(device=device)
    if torch_dtype != torch.float32 and device != "cpu":
        model = model.to(dtype=torch_dtype)
        model.flame_model.to(dtype=torch.float32)
    model.eval()
    return model
