"""FreeUV descriptor loader — LoadFreeUV node + weight discovery."""
from __future__ import annotations

import logging
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .freeuv_runtime import resolve_freeuv_root
from .kaolrm_load import resolve_device

log = logging.getLogger(__name__)

FREEUV_MODEL = io.Custom("FREEUV_MODEL")

FREEUV_SUBDIR = "freeuv"
SD15_SUBDIR = "freeuv/sd15"
CLIP_SUBDIR = "freeuv/image_encoder_l"
ALIGNER_FILENAME = "uv_structure_aligner.bin"
DETAIL_FILENAME = "flaw_tolerant_facial_detail_extractor.bin"
SD15_HF_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_HF_REPO = "openai/clip-vit-large-patch14"
FREEUV_RELEASE_URL = "https://github.com/YangXingchao/FreeUV/releases"
FREEUV_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1GkpZF9Ruzdvr0oX0J7__nkEr0bO5Jotj"
)


def _download_hf_snapshot(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("FreeUV: downloading HF snapshot '%s' → '%s'", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def _download_gdrive_folder(url: str, out_dir: Path) -> None:
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "Auto-download of FreeUV checkpoints requires 'gdown'. "
            "Install with: pip install 'gdown>=5.0'"
        ) from e
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("FreeUV: downloading Google Drive folder → '%s'", out_dir)
    gdown.download_folder(url=url, output=str(out_dir), quiet=False, use_cookies=False)


def _ensure_sd15_snapshot() -> Path:
    path = Path(folder_paths.models_dir) / SD15_SUBDIR
    if not path.is_dir() or not (path / "model_index.json").is_file():
        log.info("FreeUV SD1.5 snapshot missing at '%s' — auto-downloading.", path)
        _download_hf_snapshot(SD15_HF_REPO, path)
        if not (path / "model_index.json").is_file():
            raise RuntimeError(
                f"SD1.5 auto-download to '{path}' did not produce 'model_index.json'. "
                f"Expected snapshot of '{SD15_HF_REPO}' with: unet/, vae/, text_encoder/, "
                "tokenizer/, scheduler/, feature_extractor/, safety_checker/, model_index.json"
            )
    return path


def _ensure_clip_snapshot() -> Path:
    path = Path(folder_paths.models_dir) / CLIP_SUBDIR
    if not path.is_dir() or not (path / "config.json").is_file():
        log.info("FreeUV CLIP snapshot missing at '%s' — auto-downloading.", path)
        _download_hf_snapshot(CLIP_HF_REPO, path)
        if not (path / "config.json").is_file():
            raise RuntimeError(
                f"CLIP-ViT-L/14 auto-download to '{path}' did not produce 'config.json'. "
                f"Expected snapshot of '{CLIP_HF_REPO}'."
            )
    return path


def _ensure_freeuv_weight(filename: str) -> Path:
    path = Path(folder_paths.models_dir) / FREEUV_SUBDIR / filename
    if not path.is_file():
        log.info(
            "FreeUV checkpoint '%s' missing at '%s' — auto-downloading from Google Drive.",
            filename,
            path,
        )
        _download_gdrive_folder(FREEUV_GDRIVE_FOLDER, path.parent)
        if not path.is_file():
            raise RuntimeError(
                f"Missing FreeUV checkpoint '{filename}' at '{path}' after auto-download. "
                f"Source: {FREEUV_GDRIVE_FOLDER} (releases: {FREEUV_RELEASE_URL})"
            )
    return path


def ensure_freeuv_weights() -> dict[str, Path]:
    return {
        "sd15_root": _ensure_sd15_snapshot(),
        "clip_root": _ensure_clip_snapshot(),
        "aligner_path": _ensure_freeuv_weight(ALIGNER_FILENAME),
        "detail_path": _ensure_freeuv_weight(DETAIL_FILENAME),
    }


class LoadFreeUV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadFreeUV",
            display_name="Load FreeUV",
            category="KaoLRM",
            description=(
                "Resolve FreeUV albedo-UV generation assets and output a model descriptor. "
                "FreeUV is CC BY-NC-SA 4.0 — outputs are research-only AND carry a "
                "ShareAlike redistribution obligation."
            ),
            inputs=[
                io.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    optional=True,
                    tooltip="'auto' picks cuda when available, cpu otherwise.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["auto", "fp32", "fp16", "bf16"],
                    default="auto",
                    optional=True,
                    tooltip=(
                        "'auto' → fp32 everywhere. fp16 on ControlNet + detail_encoder "
                        "is untested and may underflow."
                    ),
                ),
                io.Boolean.Input(
                    "i_understand_non_commercial",
                    default=False,
                    tooltip=(
                        "FreeUV is CC BY-NC-SA 4.0. Combined with KaoLRM (CC BY-NC 4.0) "
                        "and FLAME (MPI non-commercial), ALL outputs are research-only. "
                        "The ShareAlike clause ALSO requires any redistribution of UV outputs "
                        "to carry the same CC BY-NC-SA 4.0 license."
                    ),
                ),
            ],
            outputs=[FREEUV_MODEL.Output(display_name="freeuv_model")],
        )

    @classmethod
    def execute(
        cls,
        device: str = "auto",
        dtype: str = "auto",
        i_understand_non_commercial: bool = False,
    ):
        if not i_understand_non_commercial:
            raise RuntimeError(
                "Set 'i_understand_non_commercial' to True before using FreeUV. "
                "CC BY-NC-SA 4.0 plus FLAME/KaoLRM non-commercial constrain the pipeline."
            )
        resolved_device = resolve_device(device)
        resolved_dtype = "fp32" if dtype == "auto" else dtype
        if resolved_device == "cpu":
            resolved_dtype = "fp32"
        weights = ensure_freeuv_weights()
        freeuv_root = resolve_freeuv_root(required=False)
        descriptor = {
            "sd15_root": str(weights["sd15_root"]),
            "clip_root": str(weights["clip_root"]),
            "aligner_path": str(weights["aligner_path"]),
            "detail_path": str(weights["detail_path"]),
            "device": resolved_device,
            "dtype": resolved_dtype,
            "freeuv_root": str(freeuv_root) if freeuv_root is not None else None,
        }
        return io.NodeOutput(descriptor)
