# FreeUV Node Suite — Layer 1 Albedo UV Generation (v0.1)

## Scope

Thin ComfyUI wrapper over FreeUV's inference stack (SD1.5 UNet + ControlNet UV structure aligner + detail encoder + DDIM). Two nodes: `LoadFreeUV` returns a descriptor; `FreeUVGenerate` takes one required IMAGE (`flaw_uv_image`) plus an optional reference UV IMAGE and emits a 512×512 albedo UV IMAGE. Weight discovery covers SD1.5 base, CLIP-ViT-L/14 image encoder, and the two FreeUV `.bin` checkpoints. License: CC BY-NC-SA 4.0 on top of the existing non-commercial stack.

**Out of scope (v0.2+):** `FLAMEProjectToUV` (render KaoLRM mesh + photo → flaw UV), scheduler choice, batch size > 1, negative prompt, separate dtype per submodel.

---

## 1. Node Boundaries and Wire Types

**Resolution (Q8):** Node is named `FreeUVGenerate` — mirrors the `SMIRKPredict` / `KaoLRMReconstruct` verb-noun pattern.

Two nodes only:

- `LoadFreeUV` (`nodes/freeuv_load.py`) — resolves all four weight directories, returns `FREEUV_MODEL` descriptor. Heavy pipeline construction deferred to `FreeUVGenerate`.
- `FreeUVGenerate` (`nodes/freeuv_generate.py`) — consumes `FREEUV_MODEL` + `flaw_uv_image` (required IMAGE) + `reference_uv` (optional IMAGE, falls back to bundled asset when not wired). Emits `io.Image`.

**Resolution (Q2):** `reference_uv` is an optional input. When the wire is not connected, `load_reference_uv()` in `nodes/freeuv_assets.py` supplies the bundled `assets/freeuv_reference_uv.jpg` (sourced from `data-process/resources/uv.jpg` in the upstream FreeUV repo, redistributed under CC BY-NC-SA 4.0 with attribution).

No new custom wire types beyond `FREEUV_MODEL`. Both IMAGE inputs and the IMAGE output use the built-in `io.Image`.

### `FREEUV_MODEL` descriptor

```python
{
    "sd15_root":    str,       # abs path to ComfyUI/models/freeuv/sd15/
    "clip_root":    str,       # abs path to ComfyUI/models/freeuv/image_encoder_l/
    "aligner_path": str,       # abs path to uv_structure_aligner.bin
    "detail_path":  str,       # abs path to flaw_tolerant_facial_detail_extractor.bin
    "device":       str,       # "cuda" | "cpu"
    "dtype":        str,       # "fp32" | "fp16" | "bf16"
    "freeuv_root":  str | None # resolved by freeuv_runtime.py
}
```

---

## 2. Runtime Resolution — `nodes/freeuv_runtime.py`

**Resolution (Q3):** FreeUV's `detail_encoder/` package uses relative imports (`._clip`, `.attention_processor`, `.resampler`). Isolated `importlib.util.spec_from_file_location` loading — the SMIRK/KaoLRM house pattern — breaks relative imports. This is a **deliberate house-pattern divergence**: `freeuv_runtime.py` injects the vendor dir into `sys.path` instead so the entire `detail_encoder/` package resolves normally.

Three-option resolution mirrors KaoLRM/SMIRK:

```python
FREEUV_ENV_VAR = "FREEUV_ROOT"
FREEUV_CANDIDATES = [REPO_ROOT / "third_party" / "freeuv"]

def _is_freeuv_root(root: Path) -> bool:
    return (root / "detail_encoder" / "__init__.py").is_file()

def resolve_freeuv_root(*, required: bool = True) -> Path | None: ...

def ensure_freeuv_on_path(*, required: bool = True) -> Path | None:
    """Inject vendor dir into sys.path so relative imports inside detail_encoder/ resolve.
    Diverges from SMIRK/KaoLRM spec_from_file_location pattern — justified by relative imports.
    """
    root = resolve_freeuv_root(required=required)
    if root is None:
        return None
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        log.info("Added FreeUV source root to sys.path: %s", root)
    return root
```

- **Option A (preferred):** `third_party/freeuv/` vendored submodule.
- **Option B:** `FREEUV_ROOT=/abs/path/to/FreeUV` env var.
- **Option C:** installed `freeuv` package in the active environment.

`ensure_freeuv_on_path()` is called lazily inside `FreeUVGenerate.execute()`, never at module import.

<!-- BLOCKER: Pin the upstream YangXingchao/FreeUV commit hash in freeuv_runtime.py before v0.1 release. Owner: first integrator. -->

---

## 3. Weight and Config Discovery — `nodes/freeuv_load.py`

**Resolution (Q1):** SD1.5 lives in a self-contained `ComfyUI/models/freeuv/sd15/` directory in HF snapshot format. The pipeline's `from_pretrained(model_id, subfolder="unet")` call expects a directory tree, not a single safetensors file. Sharing with an existing ComfyUI SD1.5 checkpoint is not supported in v0.1 to avoid cross-install config drift.

**Resolution (Q5):** CLIP weights live at `ComfyUI/models/freeuv/image_encoder_l/` — an HF snapshot of `openai/clip-vit-large-patch14`. A dedicated subdir avoids collisions with ComfyUI's `clip_vision/` directory.

```python
FREEUV_SUBDIR      = "freeuv"
SD15_SUBDIR        = "freeuv/sd15"
CLIP_SUBDIR        = "freeuv/image_encoder_l"
ALIGNER_FILENAME   = "uv_structure_aligner.bin"
DETAIL_FILENAME    = "flaw_tolerant_facial_detail_extractor.bin"
SD15_HF_REPO       = "stable-diffusion-v1-5/stable-diffusion-v1-5"   # not runwayml/ — deprecated
CLIP_HF_REPO       = "openai/clip-vit-large-patch14"
FREEUV_RELEASE_URL = "https://github.com/YangXingchao/FreeUV/releases"
```

<!-- BLOCKER: SHA256 hash for uv_structure_aligner.bin not confirmed. Pin in docs/model_hashes.md before v0.1 release. Owner: first integrator. -->
<!-- BLOCKER: SHA256 hash for flaw_tolerant_facial_detail_extractor.bin not confirmed. Pin in docs/model_hashes.md before v0.1 release. Owner: first integrator. -->

`ensure_freeuv_weights()` checks all four directories and raises `RuntimeError` on the first missing one, naming the exact path, filename, and HF repo URL — matching the pattern in `ensure_smirk_weights()` and `ensure_generic_flame_pkl()`.

The missing-SD15 error message must list the expected subdirectory tree so users know what to populate under `models/freeuv/sd15/`:

```
RuntimeError: SD1.5 snapshot not found at '{sd15_path}'.
Download the HF snapshot of 'stable-diffusion-v1-5/stable-diffusion-v1-5' and place it there.
Expected structure: unet/, vae/, text_encoder/, tokenizer/, scheduler/,
                    feature_extractor/, safety_checker/, model_index.json
```

---

## 4. `LoadFreeUV` Node — `nodes/freeuv_load.py`

```python
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
                io.Combo.Input("device", options=["auto", "cpu", "cuda"], default="auto",
                    optional=True,
                    tooltip="'auto' picks cuda when available, cpu otherwise."),
                io.Combo.Input("dtype", options=["auto", "fp32", "fp16", "bf16"], default="auto",
                    optional=True,
                    tooltip=(
                        "'auto' → fp32 everywhere. fp16 on ControlNet + detail_encoder "
                        "is untested and may underflow (analog of Brutal Review #11)."
                    )),
                io.Boolean.Input("i_understand_non_commercial", default=False,
                    tooltip=(
                        "FreeUV is CC BY-NC-SA 4.0. Combined with KaoLRM (CC BY-NC 4.0) "
                        "and FLAME (MPI non-commercial), ALL outputs are research-only. "
                        "The ShareAlike clause ALSO requires any redistribution of UV outputs "
                        "to carry the same CC BY-NC-SA 4.0 license."
                    )),
            ],
            outputs=[FREEUV_MODEL.Output(display_name="freeuv_model")],
        )
```

**Resolution (Q4):** `dtype` default is `auto` → `fp32` everywhere in v0.1. One knob covers the full pipeline — no separate dtype per submodel. fp16 on CUDA is exposed but flagged in the tooltip as untested.

`execute()` raises `RuntimeError` if the gate is False, resolves device/dtype via the shared `resolve_device` / `resolve_dtype` helpers from `kaolrm_load.py`, calls `ensure_freeuv_weights()`, and returns the descriptor dict. Heavy pipeline construction is deferred to `FreeUVGenerate`.

---

## 5. `FreeUVGenerate` Node — `nodes/freeuv_generate.py`

### Cache

```python
_FREEUV_CACHE: dict[tuple[str, str, str, str, str, str, str | None], object] = {}
# key = (device, dtype, sd15_root, clip_root, aligner_path, detail_path, freeuv_root)
```

One entry covers the full assembled pipeline (UNet + ControlNet + detail_encoder + CLIP encoder together). `freeuv_root` is in the key because a different source path implies different code.

### Schema

<!-- interpreted R#3 as: drop the Required column to match house convention (io.Int.Input with a default is effectively optional; SMIRK/KaoLRM nodes don't label required) -->

| Name | Type | Default | Notes |
|---|---|---|---|
| `freeuv_model` | `FREEUV_MODEL` | — | Descriptor from `LoadFreeUV`. |
| `flaw_uv_image` | `io.Image` | — | 512×512 UV image with imperfections. Batch must be 1. |
| `reference_uv` | `io.Image` | bundled asset | Optional. Clean reference UV template. Falls back to `load_reference_uv()` when not wired. |
| `seed` | `io.Int` | -1 | -1 → random each run. Fixed seed for reproducibility. |
| `guidance_scale` | `io.Float` | 1.4 | Matches upstream default in `inference.py`. |
| `num_inference_steps` | `io.Int` | 30 | DDIM step count. |

**Resolution (Q6):** Batch size 1 enforced. `RuntimeError` if `flaw_uv_image.shape[0] != 1`.

**Resolution (Q10):** DDIM scheduler locked in v0.1, swapped in after pipeline construction as upstream does. No exposed scheduler input. Scheduler choice is a v0.2 candidate.

Output: `io.Image` tensor `[1, 512, 512, 3]` float32 in [0, 1].

### Inference flow

1. Validate batch size 1. Resize both inputs to 512×512 (bicubic) if not already.
2. If `reference_uv` is None (not wired), call `load_reference_uv()` from `nodes/freeuv_assets.py`.
3. Load/cache the pipeline via `_get_cached_pipeline(freeuv_model)`. First call: `ensure_freeuv_on_path()`, construct `StableDiffusionControlNetPipeline` from SD1.5 HF snapshot dir, load aligner + detail encoder from `.bin` paths, swap in DDIM scheduler.
4. Convert IMAGE tensors `[1, H, W, 3]` → PIL RGB (upstream `inference.py` expects PIL).
5. Resolve seed: `-1` → `torch.randint(0, 2**32, (1,)).item()`.
6. Call the detail encoder's `generate` method with the verified upstream signature (source: `YangXingchao/FreeUV/blob/main/detail_encoder/encoder_freeuv.py`):

   ```python
   result_pil = detail_extractor.generate(
       uv_structure_image=ref_pil,
       flaw_uv_image=flaw_pil,
       pipe=pipe,
       seed=seed,
       guidance_scale=guidance_scale,
       num_inference_steps=num_inference_steps,
   )
   ```

   <!-- BLOCKER: confirm detail_encoder.generate signature against pinned commit before v0.1 release. The `pipe` kwarg and positional order above match `main` at time of spec authoring but must be re-verified at pin time. -->

   Generator is created with `torch.Generator(device=resolved_device).manual_seed(seed)` if the upstream `generate` method accepts a `generator` kwarg; otherwise seed is passed directly as shown. The `resolved_device` comes from the descriptor.

7. Convert PIL → `[1, 512, 512, 3]` float32 in [0, 1].
8. Return `io.NodeOutput(output_image)`. All tensors on CPU.

---

## 6. Bundled Reference UV Asset — `nodes/freeuv_assets.py`

**Resolution (Q2):** Upstream `data-process/resources/uv.jpg` is bundled as `assets/freeuv_reference_uv.jpg`. CC BY-NC-SA 4.0 permits redistribution with attribution and same-license conditions met.

```python
_REFERENCE_UV_CACHE: torch.Tensor | None = None

def load_reference_uv() -> torch.Tensor:
    """Load bundled reference UV once. Returns [1, H, W, 3] float32 in [0, 1]."""
    global _REFERENCE_UV_CACHE
    if _REFERENCE_UV_CACHE is None:
        asset_path = Path(__file__).resolve().parents[1] / "assets" / "freeuv_reference_uv.jpg"
        if not asset_path.exists():
            raise RuntimeError(
                f"Bundled reference UV not found at '{asset_path}'. "
                "This file ships with the repo — check your install."
            )
        img = Image.open(asset_path).convert("RGB")
        t = torch.from_numpy(np.array(img)).float() / 255.0
        _REFERENCE_UV_CACHE = t.unsqueeze(0)  # [1, H, W, 3]
    return _REFERENCE_UV_CACHE
```

One module-level function, one cache variable. No class hierarchy.

---

## 7. Known Corrections

Both corrections must be applied in the same PR that lands the FreeUV nodes.

**`docs/pipeline-roadmap.md` line 75 — "Native 1K" is incorrect.** FreeUV's SD1.5 pipeline outputs 512×512. The line reads: "Native 1K; upscale to 2K via ESRGAN-skin or similar only at export." Correct to: "Native 512×512; upscale to 1K/2K at export via ESRGAN-skin or similar."

**`runwayml/stable-diffusion-v1-5` is deprecated.** The HuggingFace repo was transferred to `stable-diffusion-v1-5/stable-diffusion-v1-5`. Every error message and doc that references the old org name must be updated. The constant in `nodes/freeuv_load.py` is `SD15_HF_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"` — this is the canonical reference.

---

## 8. File Layout and Registration

New files:

```
nodes/freeuv_runtime.py          # resolver + sys.path injection (deliberate divergence from SMIRK)
nodes/freeuv_load.py             # LoadFreeUV node + FREEUV_MODEL wire + weight helpers
nodes/freeuv_generate.py         # FreeUVGenerate node + _FREEUV_CACHE
nodes/freeuv_assets.py           # load_reference_uv() helper
assets/freeuv_reference_uv.jpg   # bundled from upstream data-process/resources/uv.jpg
assets/FREEUV_LICENSE.txt        # CC BY-NC-SA 4.0 license text from YangXingchao/FreeUV
```

`nodes/__init__.py` diff:

```python
# Before (current state in nodes/__init__.py):
from .flame_params_edit import FLAMEParamsEdit
from .flame_params_to_mesh import FLAMEParamsToMesh
from .kaolrm_load import LoadKaoLRM
from .kaolrm_preprocess import KaoLRMPreprocess
from .kaolrm_reconstruct import KaoLRMReconstruct
from .mesh_preview import MeshPreview
from .smirk_load import LoadSMIRK
from .smirk_predict import SMIRKPredict

NODE_CLASSES = [
    LoadKaoLRM,
    KaoLRMPreprocess,
    KaoLRMReconstruct,
    MeshPreview,
    LoadSMIRK,
    SMIRKPredict,
    FLAMEParamsEdit,
    FLAMEParamsToMesh,
]

# After (add these two imports and extend NODE_CLASSES):
from .freeuv_load import LoadFreeUV
from .freeuv_generate import FreeUVGenerate

NODE_CLASSES = [
    LoadKaoLRM,
    KaoLRMPreprocess,
    KaoLRMReconstruct,
    MeshPreview,
    LoadSMIRK,
    SMIRKPredict,
    FLAMEParamsEdit,
    FLAMEParamsToMesh,
    LoadFreeUV,        # new
    FreeUVGenerate,    # new
]
```

`__init__.py` (repo root) is unchanged — `ComfyuiFlameExtension.get_node_list()` reads `NODE_CLASSES` dynamically.

**Category:** Keep `category="KaoLRM"` for v0.1. Migrate all nodes to `WYSIWYG/Face/...` in the v0.2 multi-suite reorganization noted in CLAUDE.md.

`requirements.txt` additions: `diffusers`, `transformers`, `accelerate` (required by the SD1.5 diffusers pipeline). Keep `xformers` out of required deps — only needed for the Gaussian path.

---

## 9. Testing Strategy

All tests pass with no real weights present. Slow end-to-end (real weights) marked `@pytest.mark.slow`.

| Test file | Mocked | Asserted |
|---|---|---|
| `tests/test_freeuv_runtime.py` | `_is_freeuv_root` sentinel; env var | `resolve_freeuv_root`: env-var path wins over candidate dirs; candidate dir detected by `detail_encoder/__init__.py`; installed-package fallback; `required=True` raises naming `FREEUV_ROOT`. |
| `tests/test_freeuv_load.py` | `folder_paths.models_dir` (monkeypatch); dummy dir stubs | Gate False → `RuntimeError`; missing `sd15/` dir → `RuntimeError` naming path + `stable-diffusion-v1-5/stable-diffusion-v1-5` + expected subdirs; missing `uv_structure_aligner.bin` → `RuntimeError` naming exact path + URL; all present → descriptor has keys `sd15_root`, `clip_root`, `aligner_path`, `detail_path`, `device`, `dtype`, `freeuv_root`. |
| `tests/test_freeuv_generate.py` | `_load_freeuv_pipeline` monkeypatched to return a dummy `detail_extractor` whose `generate()` returns a fixed PIL image; `load_reference_uv` patched to return a `[1, 32, 32, 3]` tensor | Output shape `(1, 512, 512, 3)`, dtype float32, values in [0, 1]. Cache key tuple has 7 elements in correct order. Batch > 1 raises `RuntimeError`. Seed -1 does not raise. `reference_uv=None` triggers `load_reference_uv()` path (assert called once). When `reference_uv` is provided (non-None), `load_reference_uv()` is NOT called and the provided tensor flows into the `generate()` call (spy on `load_reference_uv`, assert `call_count == 0`). |
| `tests/test_freeuv_assets.py` | Monkeypatched asset path pointing at a synthetic 64×64 JPEG | `load_reference_uv()` returns `[1, 64, 64, 3]` float32; second call returns the same tensor object (cache hit, `is` identity check); missing asset path raises `RuntimeError` naming the path. |

---

## 10. Non-commercial Gate and ShareAlike Licensing

**This section is mandatory reading before shipping this node. Read it entirely.**

The existing pipeline — KaoLRM (CC BY-NC 4.0) + FLAME (MPI non-commercial) — already constrains all outputs to **research use only**. FreeUV adds an escalation that the existing stack does not impose.

### What escalates

FreeUV is licensed under **CC BY-NC-SA 4.0**. The two relevant clauses:

- **NC (Non-Commercial):** no commercial use — same as the existing stack.
- **SA (ShareAlike):** any redistribution of adapted material — including generated UV maps and downstream assets derived from them — must carry the **same CC BY-NC-SA 4.0 license**. This is a viral, downstream obligation.

The combined constraint is: **all outputs are research-only, and any UV maps or textures produced via the FreeUV node may only be redistributed under CC BY-NC-SA 4.0**. A researcher who publishes a dataset of generated UV maps must apply CC BY-NC-SA 4.0 to that dataset.

### Surface points

1. **`i_understand_non_commercial` tooltip on `LoadFreeUV`** (text in §4) names both NC and SA explicitly. A user who reads only the checkbox label cannot claim ignorance of SA.
2. **`assets/FREEUV_LICENSE.txt`** — full CC BY-NC-SA 4.0 license text from `YangXingchao/FreeUV`.
3. **`assets/README.md`** (create if missing) — add a paragraph: "UV maps produced via the FreeUV node inherit CC BY-NC-SA 4.0. If you redistribute such maps or downstream assets derived from them, you must apply the same license. This is in addition to the non-commercial-only restriction from KaoLRM and FLAME."
4. Do **not** change the repo-level `LICENSE` — that is a legal decision outside this spec's scope.

**Resolution (Q7):** The ShareAlike viral implication for exports is surfaced in the tooltip wording and in `assets/README.md`. The repo license is not touched.

---

## References

- FreeUV (CVPR 2025) — `YangXingchao/FreeUV`. Ground-truth-free UV recovery via SD1.5 + Cross-Assembly inference.
- Stable Diffusion v1.5 — `stable-diffusion-v1-5/stable-diffusion-v1-5` on HuggingFace (formerly `runwayml/stable-diffusion-v1-5`, now deprecated).
- CLIP ViT-L/14 — `openai/clip-vit-large-patch14` on HuggingFace.
- Pipeline roadmap: `docs/pipeline-roadmap.md` (Layer 1 — albedo).
- SMIRK integration spec (section-structure template): `.claude/plan/smirk-final-plan.md`.
