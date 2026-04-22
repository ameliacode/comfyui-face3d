# CLAUDE.md

Project guidance for Claude Code working on the WYSIWYG Studio ComfyUI face reconstruction pipeline.

**Current state.** A KaoLRM-based node suite that produces a FLAME-topology mesh from a single image, plus a SMIRK expression-refinement branch that merges SMIRK's per-frame expression + jaw pose into KaoLRM's identity params and re-solves the mesh via the canonical FLAME head. A single `FLAMEParamsEdit` node folds the merge policy together with user-facing sliders (strengths + pose/translation offsets + `fix_z_trans` override) — the seed for the FLAME Param Optimizer roadmap entry. FreeUV albedo suite (`LoadFreeUV` + `FreeUVGenerate`) is scaffolded but not yet smoke-tested end-to-end — requires manual weights + vendor checkout. Mesh only on the KaoLRM/SMIRK path; no texture wired into `MeshPreview` yet, no relighting maps. Scaffolded in `nodes/kaolrm_*.py`, `nodes/smirk_*.py`, `nodes/flame_params_*.py`, and `nodes/freeuv_*.py`; registered via the V3 `ComfyExtension` pattern in `__init__.py`.

**Target state.** A 5-layer WYSIWYG Studio pipeline — KaoLRM geometry → FreeUV albedo → MoSAR intrinsic maps → DECA expression detail → pore composite → export — producing fully-textured, relightable FLAME assets. Full roadmap in [`docs/pipeline-roadmap.md`](docs/pipeline-roadmap.md). Approved integration spec for the SMIRK branch: [`plan/smirk-final-plan.md`](plan/smirk-final-plan.md). Approved spec for the next layer (FreeUV albedo): [`plan/final-plan.md`](plan/final-plan.md).

**Rule of thumb.** Work forward from the KaoLRM mesh path. The old FLAME editor/viewer nodes are not the active product path, but `nodes/flame_core.py` and `nodes/flame_render_util.py` are still active dependencies of the KaoLRM scaffold and must not be treated as dead code.

---

## When the user gives a planning topic

For requests like "plan a spec for X" / "draft a design for Y", switch to orchestrator mode and run the multi-agent markdown planning loop. Read [`.claude/planning-workflow.md`](planning-workflow.md) before acting.

---

## Current implementation — KaoLRM mesh suite + SMIRK expression branch

### Node pipeline

```
LoadKaoLRM ─► KaoLRMPreprocess (optional) ─► KaoLRMReconstruct ─► MESH ─► MeshPreview
                                                         │
                                                         └─► FLAME_PARAMS ──────┐
                                                                                ▼
                                                                      FLAMEParamsEdit ─► FLAME_PARAMS ─► FLAMEParamsToMesh ─► MESH ─► MeshPreview
                                                                                ▲
LoadSMIRK ─► SMIRKPredict ─► FLAME_PARAMS ──────────────────────────────────────┘
```

`KaoLRMPreprocess` does resize and optional `rembg` background removal (opt-in — `rembg` ships its own heavy weights). The straight-line KaoLRM path (top row) still works standalone — `KaoLRMReconstruct` emits both the mesh and a canonical `FLAME_PARAMS` dict, so a params-refinement workflow can start from KaoLRM alone or fuse in the SMIRK branch via `FLAMEParamsEdit`'s optional secondary input. `FLAMEParamsEdit` applies the fixed merge policy when the override is wired (`shape ← params`, `expression ← override`, `pose = [global ← params, jaw ← override]`, `scale/translation ← params`, `fix_z_trans ← params`), then applies the user sliders (strengths + offsets + `fix_z_trans` override) on top. `FLAMEParamsToMesh` re-solves the FLAME vertices using `nodes/flame_core.FlameCore`. `MeshPreview` rasterizes either mesh for visual verification.

### Wire types

- `KAOLRM_MODEL` — `io.Custom("KAOLRM_MODEL")`, a plain dict descriptor `{variant, ckpt_path, flame_pkl_path, config_path, device, dtype, kaolrm_root}`. The heavy model is cached in `_KAOLRM_CACHE` inside `nodes/kaolrm_reconstruct.py`, keyed by `(variant, device, dtype, ckpt_path, flame_pkl_path, config_path, kaolrm_root)`.
- `SMIRK_MODEL` — `io.Custom("SMIRK_MODEL")`, a plain dict descriptor `{ckpt_path, device, dtype, smirk_root}`. The `SmirkEncoder` is cached in `_SMIRK_CACHE` inside `nodes/smirk_predict.py`, keyed by `(device, dtype, ckpt_path, smirk_root)`.
- `FLAME_PARAMS` — `io.Custom("FLAME_PARAMS")` declared in `nodes/flame_params_wire.py`. A plain dict with canonical batched shapes `{shape[B,100], expression[B,50], pose[B,6], scale[B,1], translation[B,3], fix_z_trans: bool}`. `pose` is laid out as `[global(3) | jaw(3)]` to match KaoLRM. `B=1` for v0.1. `validate_flame_params(params, *, source)` is the gatekeeper every node calls on entry.
- `MESH` on the wire is the **built-in** `io.Mesh` (`MeshPayload` from `comfy_api.latest._util`), carrying `{vertices[B,V,3], faces[B,F,3]}` plus ad-hoc attrs `flame_params` (flat `[N]` tensors for debugging only — the canonical batched params travel on the second output wire), `gender="generic"`, `source_resolution=224`, `topology`, `num_sampling`, `fix_z_trans`, and the original FLAME topology in `base_vertices`/`base_faces` when a sampled point cloud is emitted. `nodes/mesh_types.py` declares an unused `io.Custom("MESH")` alongside live helpers (`coerce_mesh`, `compute_vertex_normals`) — the custom type is vestigial and can be deleted at cleanup.

### Nodes

- **`LoadKaoLRM`** (`nodes/kaolrm_load.py`) — resolves `model.safetensors` + FLAME pkl and returns the descriptor. Heavy build is lazy in `KaoLRMReconstruct`, not here. Inputs: `variant` (`mono`|`multiview`, default `mono`), `device` (`auto`|`cpu`|`cuda`), `dtype` (`auto`|`fp32`|`fp16`|`bf16`; `auto` → `fp16` on CUDA, forced `fp32` on CPU), `i_understand_non_commercial` boolean gate (must be `True` — see Brutal Review #1).
- **`KaoLRMPreprocess`** (`nodes/kaolrm_preprocess.py`) — resize to 224×224, then optional `rembg` via cached session (`_get_rembg_session`), composite against `background_color` with `composite_alpha`, resize mask in lockstep. Default `remove_background=False`. Outputs `IMAGE`, `MASK`.
- **`KaoLRMReconstruct`** (`nodes/kaolrm_reconstruct.py`) — the mesh predictor. Inputs include `source_cam_dist` float (default `2.0`) and `num_sampling` int (default `5023`, the FLAME vertex count). Background removal / rembg model choice live on `KaoLRMPreprocess`, not here. Outputs: built-in `MESH` with attrs **and** `FLAME_PARAMS` (canonical batched `[1, N]` tensors + `fix_z_trans`). All tensors on CPU. When `num_sampling != 5023`, the mesh output switches to a sampled point cloud and preserves the original FLAME topology in `base_vertices`/`base_faces`; the `FLAME_PARAMS` output is unaffected.
- **`MeshPreview`** (`nodes/mesh_preview.py`) — rasterizes `MESH` to IMAGE+MASK. Default renderer `soft_torch`; `pytorch3d` opt-in when installed. Uses `render_mesh()` for triangle meshes and `render_points()` for sampled point clouds. Depends on `nodes/flame_render_util.py`, which is active shared code.
- **`LoadSMIRK`** (`nodes/smirk_load.py`) — resolves `SMIRK_em1.pt` and returns the `SMIRK_MODEL` descriptor. Inputs: `device` (`auto`|`cpu`|`cuda`), `dtype` (`auto`|`fp32`|`fp16`|`bf16`), `i_understand_non_commercial` gate (SMIRK code is MIT but FLAME topology is not). Heavy encoder build is lazy in `SMIRKPredict`.
- **`SMIRKPredict`** (`nodes/smirk_predict.py`) — runs `SmirkEncoder` on a 224×224 face crop and emits `FLAME_PARAMS`. SMIRK returns `{shape[1,300], expression[1,50], pose[1,3], jaw[1,3], eyelid[1,2], cam[1,3]}`; we build `pose = [zeros(1,3) | jaw]` and zero-fill `shape/scale/translation` (the merge policy inside `FLAMEParamsEdit` discards them). Enforces batch size 1. Cache: `_SMIRK_CACHE` keyed by `(device, dtype, ckpt_path, smirk_root)`.
- **`FLAMEParamsEdit`** (`nodes/flame_params_edit.py`) — one-stop node for merging + user refinement. Takes one required `params` (normally KaoLRM) and one optional `params_override` (normally SMIRK). When `params_override` is wired, the fixed merge policy applies first (`shape ← params`, `expression ← override`, `pose = [global ← params, jaw ← override]`, `scale/translation/fix_z_trans ← params`). Then the slider stage applies: `shape_strength`, `expression_strength`, `jaw_strength` (multiplicative, 0.0–1.5), `scale_multiplier` (0.1–3.0), `global_pose_offset_{x,y,z}` + `translation_offset_{x,y,z}` (additive, −1.0 to 1.0), and `fix_z_trans_override` combo (`inherit`/`force_true`/`force_false`). Seed for the FLAME Param Optimizer roadmap entry — landmark-fit / photometric losses land later inside this same node. Validates via `validate_flame_params`; rejects batch mismatch, missing `fix_z_trans`, and flat tensors with an error naming the canonical `[B, 100]`-style shape.
- **`FLAMEParamsToMesh`** (`nodes/flame_params_to_mesh.py`) — re-solves a FLAME mesh from edited params using `nodes.flame_core.get_flame_core` (not the KaoLRM-vendored `flame.py` — cleaner layering, and the cache is shared with other future params-driven nodes). Pose `[6]` is expanded to `[15]` via `_expand_pose_6_to_15` (neck + both eyes zero-filled). Scale and translation are applied outside `FlameCore.forward` so `fix_z_trans=True` zeros translation z *before* the per-vertex offset is applied.

### Inference flow inside `KaoLRMReconstruct.execute()`

1. Resize input IMAGE to 224×224, bicubic, clamp to [0,1]. **No ImageNet normalization** — `LRMInferrer` does not normalize. (Background removal already happened upstream in `KaoLRMPreprocess` if wired in.)
2. Pull cached model via `_get_cached_model(kaolrm_model)`; first call hits `load_mesh_only_model(...)` in `kaolrm_mesh_model.py`.
3. Build canonical camera via `_build_source_camera(runtime, source_cam_dist, device)` — extrinsics from `_default_source_camera(dist_to_center)`, intrinsics from `create_intrinsics(f=0.75, c=0.5)`, composed via `build_camera_principle`.
4. `planes = model.forward_planes(image, source_camera)`.
5. `decoded_params = model.flame_decoder(planes)` → `{shape[100], expression[50], pose[6], scale[1], translation[3]}`. Cast to float32 before the FLAME head (fp16 underflows expression coefficients — Brutal Review #11).
6. `vertices, _, sampled_vertices = model.flame2mesh(decoded_params, num_sampling, fix_z_trans=(variant=="mono"))` — matches the `skip_video=True` branch in `LRMInferrer.infer_results`. The `fix_z_trans` gate mirrors upstream's mono/multiview distinction.
7. `base_faces = model.flame_model.faces_tensor.repeat(B, 1, 1)`. If `num_sampling == 5023`, emit the FLAME mesh. Otherwise emit the sampled point cloud with empty `faces` and preserve the base mesh in `base_vertices` / `base_faces`.
8. Return `MeshPayload(...)` with attrs; all tensors on CPU so downstream device selection is free.

### Upstream reality checks

- KaoLRM inference entry: `LRMInferrer` at `/home/wswg3/github/KaoLRM/kaolrm/runners/infer/lrm.py:92`. `skip_video=True` returns vertices/faces/lmks/params without touching the Gaussian splat decoder.
- Canonical camera in that path: `_default_source_camera(dist_to_center=2.0)` → extrinsics `[[1,0,0,0],[0,0,-1,-d],[0,1,0,0]]`, intrinsics `f=0.75, c=0.5`, input `[1,3,224,224]` float32 in [0,1].
- Checkpoints: `releases/{mono,multiview}/model.safetensors`, 1.1 GB each, **CC BY-NC 4.0**. Plus **FLAME 2020 `generic_model.pkl`** (MPI non-commercial, gated registration at flame.is.tue.mpg.de).
- The full video path pulls `diff-surfel-rasterization`, `pytorch3d`, and `xformers` — all CUDA-compile. The mesh-only path needs none of them.

---

## Files

### Active (KaoLRM path)

- `nodes/kaolrm_load.py`, `nodes/kaolrm_preprocess.py`, `nodes/kaolrm_reconstruct.py`, `nodes/kaolrm_runtime.py`, `nodes/kaolrm_mesh_model.py` — node suite + runtime (KaoLRM runtime resolution, optional path injection, `import_kaolrm_symbols`, mesh-only `KaoLRMMesh` wrapper, `load_mesh_only_model(..., config_path=...)`).
- `nodes/mesh_types.py` — `coerce_mesh`, `compute_vertex_normals` helpers.
- `nodes/mesh_preview.py` — `MeshPreview` node.

### Active (SMIRK branch)

- `nodes/smirk_runtime.py` — mirrors `kaolrm_runtime.py`: `SMIRK_ENV_VAR = "SMIRK_ROOT"`, `third_party/smirk/` discovery, `import_smirk_encoder()` loads `src/smirk_encoder.py` in isolation via `importlib.util.spec_from_file_location`.
- `nodes/smirk_load.py` — `LoadSMIRK`, `SMIRK_MODEL` wire, `ensure_smirk_weights()`.
- `nodes/smirk_predict.py` — `SMIRKPredict`, `_SMIRK_CACHE`, `_load_smirk_encoder()` (monkey-patched by tests).
- `nodes/flame_params_wire.py` — `FLAME_PARAMS` custom type + `validate_flame_params()` + canonical shape map.
- `nodes/flame_params_edit.py` — `FLAMEParamsEdit` node + `_apply_merge_policy()` + `_apply_edits()` helpers + `FIX_Z_OPTIONS` constant.
- `nodes/flame_params_to_mesh.py` — `FLAMEParamsToMesh` node, `_expand_pose_6_to_15()` helper, `N_VERTICES` constant (monkey-patched by tests).

### Active (FreeUV albedo branch)

- `nodes/freeuv_runtime.py` — `FREEUV_ENV_VAR = "FREEUV_ROOT"`, `third_party/freeuv/` discovery (via `detail_encoder/__init__.py`), `ensure_freeuv_on_path()` uses **`sys.path` injection** instead of `spec_from_file_location` because FreeUV's `detail_encoder/` uses relative imports.
- `nodes/freeuv_assets.py` — `REFERENCE_UV_PATH`, `load_reference_uv()`, `_REFERENCE_UV_CACHE` (bundled 512×512 neutral reference used as the default `reference_uv` when one is not wired into `FreeUVGenerate`).
- `nodes/freeuv_load.py` — `LoadFreeUV`, `FREEUV_MODEL` wire, `_ensure_sd15_snapshot`, `_ensure_clip_snapshot`, `_ensure_freeuv_weight`, `ensure_freeuv_weights`. Weight layout under `ComfyUI/models/freeuv/{sd15/, image_encoder_l/, uv_structure_aligner.bin, flaw_tolerant_facial_detail_extractor.bin}`.
- `nodes/freeuv_generate.py` — `FreeUVGenerate`, `_FREEUV_CACHE` keyed by `(device, dtype, sd15_root, clip_root, aligner_path, detail_path, freeuv_root)`, `_load_freeuv_pipeline` (imports `detail_encoder`, diffusers `ControlNetModel` + `DDIMScheduler`, vendor `pipeline_sd15.StableDiffusionControlNetPipeline/UNet2DConditionModel`). Upstream entry: `detail_extractor.generate(uv_structure_image=..., flaw_uv_image=..., pipe=..., seed=..., guidance_scale=1.4, num_inference_steps=30)`; enforces batch size 1 and samples a random seed when `seed == -1`.

### Active shared helpers

- `nodes/flame_render_util.py` — `render_mesh()`, `render_points()`, `hex_to_rgb()`. Used by `MeshPreview`.
- `nodes/flame_core.py` — `_install_chumpy_shims()`. Called from `kaolrm_mesh_model.py` before FLAME pkl load.
- `nodes/_optional_deps.py` — `try_import_pytorch3d()`. Imported by `flame_render_util.py` for the optional pytorch3d backend.

### Parked (do not touch without explicit user request)

`nodes/flame_params.py`, `nodes/load_flame.py`, `nodes/flame_editor.py`, `nodes/flame_render.py`, `routes.py`, `js/extension.js`. Slated for repurposing as an optimizer that refines KaoLRM's predicted params.

---

## Development conventions

### Python

- Python 3.11+, type hints on all public functions.
- Use `torch` directly, not Lightning. ComfyUI is the orchestration layer.
- Avoid adding dependencies — every new package is a deployment risk. If numpy/torch/PIL can do it, do it there.
- `ruff format`, `ruff check`. Line length 100.

### ComfyUI nodes

- Register via V3 `ComfyExtension` (`io.ComfyNode` + `io.Schema`). The legacy `NODE_CLASS_MAPPINGS` dict is not used in this repo.
- Current category is `KaoLRM`. When the second node suite lands, migrate to per-suite dirs under `custom_nodes/<name>/nodes.py` and rename categories to `WYSIWYG/Face/...` (see roadmap).
- Inputs: explicit types, sensible defaults, tooltips on every field.
- Outputs: `io.NodeOutput`; for mesh use built-in `io.Mesh`, not a bare tensor.
- Model loading is lazy, cached at module scope (pattern: `_KAOLRM_CACHE` in `nodes/kaolrm_reconstruct.py`). Never load at module import.

### Weights and external models

- Per-model subdirs under `ComfyUI/models/`: `kaolrm/`, `flame/`, future `mosar/`, `freeuv/`, etc. No single umbrella dir — `folder_paths` works best one model-type per folder.
- Each node checks for its weights at first use and raises `RuntimeError` naming the exact path and upstream URL. No auto-download — artists want explicit provenance, and FLAME (gated) + KaoLRM/MoSAR (non-commercial) make automation a legal risk.
- SHA256 pins go in `docs/model_hashes.md` (file not yet authored; create when the first hash lands).

### Fixed asset paths

- `ComfyUI/models/kaolrm/mono.safetensors`
- `ComfyUI/models/kaolrm/multiview.safetensors`
- `ComfyUI/models/kaolrm/mono.config.json`
- `ComfyUI/models/kaolrm/multiview.config.json`
- `ComfyUI/models/flame/generic_model.pkl`
- `ComfyUI/models/smirk/SMIRK_em1.pt`
- `ComfyUI/models/freeuv/sd15/` (SD1.5 snapshot — `stable-diffusion-v1-5/stable-diffusion-v1-5`)
- `ComfyUI/models/freeuv/image_encoder_l/` (CLIP-ViT-L/14 snapshot — `openai/clip-vit-large-patch14`)
- `ComfyUI/models/freeuv/uv_structure_aligner.bin`
- `ComfyUI/models/freeuv/flaw_tolerant_facial_detail_extractor.bin`

### Required framework deps (`requirements.txt`)

`huggingface_hub`, `safetensors`, `einops`, `rembg`, `chumpy`, `timm>=0.9.16` (for SMIRK's `MobileNetV3`-based encoder), `diffusers>=0.27.0` (for the FreeUV pipeline). `transformers` + `accelerate` are bundled by ComfyUI. Keep `pytorch3d`, `xformers`, `diff-surfel-rasterization` **out** of required deps — they're only needed for the Gaussian splat video path we don't use.

### Dependency integration strategy for KaoLRM itself

Supported runtime resolution order:

- **Option A (preferred)** — vendor a pinned checkout under `third_party/kaolrm/`.
- **Option B** — install upstream `kaolrm` into the active Python environment.
- **Option C** — set `KAOLRM_ROOT=/abs/path/to/KaoLRM` for local development.

`LoadKaoLRM` no longer requires the runtime checkout up front; the actual import/build path is resolved lazily in `KaoLRMReconstruct`.

### Dependency integration strategy for SMIRK

Same three-option pattern as KaoLRM. `nodes/smirk_runtime.py` resolves the source tree in this order:

- **Option A (preferred)** — vendor a pinned checkout under `third_party/smirk/` (detected by the presence of `src/smirk_encoder.py`).
- **Option B** — install upstream `smirk` into the active Python environment.
- **Option C** — set `SMIRK_ROOT=/abs/path/to/smirk` for local development.

`SmirkEncoder` is imported in isolation via `importlib.util.spec_from_file_location`, sidestepping upstream's package-level `__init__.py`. Upstream repo: `https://github.com/georgeretsi/smirk`. Commit pin: **BLOCKER — to be resolved before merge.**

### Dependency integration strategy for FreeUV

Same three-option pattern as KaoLRM/SMIRK. `nodes/freeuv_runtime.py` resolves the source tree in this order:

- **Option A (preferred)** — vendor a pinned checkout under `third_party/freeuv/` (detected by the presence of `detail_encoder/__init__.py`).
- **Option B** — install upstream `freeuv` into the active Python environment.
- **Option C** — set `FREEUV_ROOT=/abs/path/to/FreeUV` for local development.

Deliberate divergence: unlike KaoLRM/SMIRK, FreeUV is loaded via **`sys.path` injection** (`ensure_freeuv_on_path`), not `importlib.util.spec_from_file_location`. FreeUV's `detail_encoder/` package uses relative imports (`._clip`, `.attention_processor`, `.resampler`), which only resolve when the vendor root is on `sys.path`. Upstream repo: `https://github.com/YangXingchao/FreeUV`. Commit pin: **BLOCKER — to be resolved before merge.**

### Testing

- Per-node unit tests: fixed input → expected output shape/range. KaoLRM suite: `test_kaolrm_load.py`, `test_kaolrm_preprocess.py`, `test_kaolrm_reconstruct.py` (mocked model; asserts the FLAME_PARAMS second output + fix_z_trans variant gating), `test_kaolrm_runtime.py`, `test_mesh_type.py`. SMIRK + params suite: `test_smirk_runtime.py`, `test_smirk_load.py`, `test_smirk_predict.py` (monkey-patches `_load_smirk_encoder`), `test_flame_params_edit.py` (passthrough, merge policy, strength sliders, offsets, `fix_z_trans_override`, validation errors), `test_flame_params_to_mesh.py` (uses `synthetic_flame_pkl` fixture from `conftest.py` to build a zeroed FLAME pkl with a configurable vertex count — set via `monkeypatch.setattr("nodes.flame_params_to_mesh.N_VERTICES", N)`). Legacy FLAME helper tests: `test_flame_core.py`, `test_flame_params.py`, `test_render.py`, `test_routes.py`.
- FreeUV suite: `test_freeuv_runtime.py` (env var + vendor detection + `sys.path` injection), `test_freeuv_load.py` (NC gate, missing-weight error messages, descriptor shape, CPU-forces-fp32), `test_freeuv_assets.py` (bundled reference UV loader + cache), `test_freeuv_generate.py` (fallback + explicit reference UV, batch>1 rejection, seed=-1 sampling, cache-key ordering; pipeline itself is monkey-patched).
- Run with `~/github/ComfyUI/venv311/bin/python -m pytest tests/` — all 74 should pass with no real weights present.
- End-to-end tests require the real weights — mark `@pytest.mark.slow`, run locally before release, never required in CI.

---

## Constraints

- **Non-commercial only.** KaoLRM weights (CC BY-NC 4.0) + NVIDIA EG3D-derived code in `gaussian_decoder.py` + FLAME 2020 (MPI non-commercial) make the whole stack research-only. Anything produced through these nodes is research-only. Surfaced via the `i_understand_non_commercial` gate on `LoadKaoLRM` and in node tooltips.
- **Mesh only.** No texture, no Gaussian splat rendering, no UV pipeline. Texture arrives at M2 (MoSAR + FreeUV).
- **Do not automate the FLAME download.** It's a gated, email-verified registration. Error messages include the exact URL, target path, and filename — that's the support contract.
- **Gen-Anima is excluded from this project.** Any reference in issues, PRs, or docs should be removed.

---

## Known gotchas

- **chumpy shims, not a chumpy-free fork.** FLAME's original pkl requires chumpy. Rather than maintain a chumpy-free reimplementation, v0.1 installs lightweight shims (`nodes/flame_core.py:_install_chumpy_shims`) before loading the pkl. Keeps the pinned `chumpy>=0.70` dep but avoids chumpy's full numerics at runtime. DECA detail (M3) will reuse this path.
- **fp16 underflow on expression coefficients.** The FLAME decoder head outputs small-magnitude expression values; half-precision can quantize them to zero and produce a dead-expression mesh. Cast to float32 before the FLAME head. Default to fp32 on CPU.
- **FLAME eyeball UVs default to (1,1)** in many exporters → samples corner pixels. Explicit eyeball texture binding is required in any production render (out of scope for the current mesh-only suite; slated for M2).

---

## Implementation status

### Landed

- `LoadKaoLRM`, `KaoLRMPreprocess`, `KaoLRMReconstruct`, `MeshPreview` implemented and registered via `nodes/__init__.py` + V3 `ComfyExtension` in `__init__.py`.
- `LoadSMIRK`, `SMIRKPredict`, `FLAMEParamsEdit`, `FLAMEParamsToMesh` scaffolded per [`plan/smirk-final-plan.md`](plan/smirk-final-plan.md) and registered. The earlier `KaoLRMParamsToFLAMEParams` shim and the stand-alone `FLAMEParamsMerge` node have been folded into `KaoLRMReconstruct`'s second output + `FLAMEParamsEdit`.
- `FLAME_PARAMS` wire type (`nodes/flame_params_wire.py`) with canonical `[B, N]` schema + `validate_flame_params` gatekeeper.
- `i_understand_non_commercial` gate on `LoadKaoLRM` and `LoadSMIRK` (Brutal Review #1).
- Missing-weight / missing-FLAME / missing-SMIRK error messages name the exact path and upstream URL (Brutal Review #2).
- `rembg` uses cached sessions via `new_session(model_name)`, opt-in by default.
- KaoLRM and SMIRK import surfaces isolated: `kaolrm_runtime.py` / `smirk_runtime.py` + `importlib.util.spec_from_file_location` — sidesteps upstream top-level `__init__.py` modules (Brutal Review #3 resolved).
- `config_path` threaded through `KAOLRM_MODEL` descriptor + `_KAOLRM_CACHE` key + `load_mesh_only_model`, so the KaoLRM release configs can live alongside the safetensors (`models/kaolrm/{variant}.config.json`).
- `KaoLRMReconstruct` emits `FLAME_PARAMS` directly as a second output (canonical `[1, N]` tensors + `fix_z_trans`); the stashed `mesh.flame_params` / `mesh.fix_z_trans` attrs remain for legacy inspection.
- `LoadFreeUV`, `FreeUVGenerate` scaffolded per [`plan/final-plan.md`](plan/final-plan.md) and registered. `nodes/freeuv_runtime.py` (`sys.path` injection, `FREEUV_ROOT` env var), `nodes/freeuv_assets.py` (bundled neutral reference UV loader with in-memory cache), `nodes/freeuv_load.py` (descriptor + weight-path resolution + NC/SA gate), `nodes/freeuv_generate.py` (`_FREEUV_CACHE` keyed by 7-tuple, upstream `detail_extractor.generate(...)` call). `FREEUV_MODEL` wire type (`io.Custom("FREEUV_MODEL")`).
- `i_understand_non_commercial` gate on `LoadFreeUV` surfaces CC BY-NC-SA 4.0 *ShareAlike* obligations in the tooltip (new viral clause on top of the KaoLRM/FLAME non-commercial constraints).
- `assets/FREEUV_LICENSE.txt` + `assets/README.md` shipped alongside the existing FLAME / SMIRK license texts.
- `diffusers>=0.27.0` added to `requirements.txt` for the FreeUV pipeline.
- `docs/pipeline-roadmap.md` layer 1 corrected from "Native 1K" to "Native 512×512" to match FreeUV's actual output resolution.
- Tests landed: 74 green across the KaoLRM, SMIRK, edit, and FreeUV suites (see Testing section for inventory).

### Open

- **BLOCKERS for first SMIRK release.** Pin the upstream `smirk` commit hash; confirm the SHA256 of `SMIRK_em1.pt` and record it in `docs/model_hashes.md`.
- **BLOCKERS for first FreeUV release.** Pin the upstream `YangXingchao/FreeUV` commit hash; confirm the SHA256 of `uv_structure_aligner.bin` + `flaw_tolerant_facial_detail_extractor.bin` and record in `docs/model_hashes.md`. Ship `assets/freeuv_reference_uv.jpg` (the 512×512 neutral reference used as the default `reference_uv` input).
- Vendor `third_party/kaolrm/`, `third_party/smirk/`, and `third_party/freeuv/` as pinned submodules for the default portable install. Runtime can still come from an installed package or `KAOLRM_ROOT` / `SMIRK_ROOT` / `FREEUV_ROOT`, but vendoring is the most reproducible default.
- Ship `assets/KAOLRM_LICENSE.txt` and `assets/EG3D_LICENSE.txt` alongside the existing `FLAME_LICENSE.txt`, `SMIRK_MIT_LICENSE.txt`, and `FREEUV_LICENSE.txt`.
- Bundle `workflows/kaolrm_smirk_mesh.json` — SMIRK-refined demo workflow — alongside the existing mesh-preview example. FreeUV demo workflow to follow once the `FLAMEProjectToUV` (mesh → flaw UV) node lands in v0.2.
- Comment block in `requirements.txt` documenting the manual KaoLRM + SMIRK + FreeUV install (submodule vs `pip install git+...@<sha>`).
- `tests/test_mesh_preview.py` — render-smoke against a synthetic 5023-vert mesh. `test_mesh_type.py` covers the helpers but not the node.
- FreeUV end-to-end smoke: requires real weights, the vendor checkout, and a manually produced flaw UV (no `FLAMEProjectToUV` node in v0.1). Mark `@pytest.mark.slow` and run locally before release.

---

## Brutal review (live risk list)

Kept inline because these are active risks for every PR, not resolved history.

1. **"Non-commercial" is viral and default.** KaoLRM + EG3D + FLAME → everything downstream is research-only. ComfyUI users skew commercial and will miss the banner. Mitigation: `i_understand_non_commercial` gate + red-text node descriptions.
2. **FLAME 2020 is gated.** Email-verified registration wall, no auto-download. Expect support load. Mitigation: error message names URL + path + filename.
3. **KaoLRM may eagerly import the world.** *Resolved* — `kaolrm_runtime.py` isolates the import surface and `kaolrm_mesh_model.py` loads individual source modules via `importlib.util.spec_from_file_location`, sidestepping the top-level `__init__.py`. Re-check if upstream KaoLRM restructures imports.
4. **2+ GB first-run download.** No HF mirror yet; `HF_MIRRORS = []`. Mitigation: document manual-drop as the primary install; mirror is aspirational.
5. **Single-author research code.** KaoLRM is 3DV 2026, one PhD maintainer, no semver, pinned `torch==2.9.1` (vs 2.5.x in most ComfyUI installs). Commit-hash pin is mandatory, not optional.
6. **Camera misuse.** `source_cam_dist=2.0` assumes the face fills most of the 224² frame in OpenLRM convention. Distant subjects or overly tight crops silently degrade predictions. Mitigation: document expected framing; auto-crop via `face_alignment` is a v0.2 candidate.
7. **Gender hardcoded `"generic"`.** KaoLRM trained on FFHQ-style data; the embedded FLAME is the generic pkl. No female/male selector in the inference API. Accept; document.
8. **"Mesh only" is under-demonstrated.** Without texture, the natural next question is "cool mesh, now what?". `MeshPreview` answers partially with a matte-gray render. Mitigation: bundle `workflows/kaolrm_mesh_preview.json`.
9. **Parked FLAME code is dead weight.** Six unused modules, bitrot risk. Acceptable for v0.1; revisit at v0.3 if the optimizer story slips.
10. **Scope creep.** Next ask after ship will be "add MoSAR texture" — not a follow-up PR. MoSAR has no public code/weights and its own license; it's a v0.2/v0.3 integration.
11. **fp16 underflow on expression coefficients.** See Known Gotchas. Test fp16 on a non-neutral face before claiming parity.
12. **No real KaoLRM model in CI.** Mocks + preprocess round-trips only. First real bug surfaces at user-report time. Document in CONTRIBUTING when authored.
13. **Plan/doc drift.** When behavior changes, update this file in the same PR or mark the old section historical. Example history: `source_cam_dist` moved from `LoadKaoLRM` to `KaoLRMReconstruct`; `remove_background`/`rembg_model` moved from `KaoLRMReconstruct` to `KaoLRMPreprocess`; `config_path` became a first-class descriptor field when KaoLRM release configs were moved out of `third_party/kaolrm/releases/` and into `models/kaolrm/`.
14. **SMIRK pose convention skew.** SMIRK returns `pose[1,3]` (global) and `jaw[1,3]` as separate tensors. Our canonical `FLAME_PARAMS` pose is `[global(3) | jaw(3)]`, and `KaoLRMReconstruct` already emits that layout. `SMIRKPredict` zeros the global slot (KaoLRM owns head pose post-merge) — do not change this without updating the merge policy inside `FLAMEParamsEdit` and `_expand_pose_6_to_15` together.
15. **FLAME head re-solve layering.** `FLAMEParamsToMesh` deliberately uses `nodes/flame_core.py:FlameCore` instead of the KaoLRM-vendored `flame.py` so the custom node suite doesn't hard-depend on the `third_party/kaolrm/` tree for the params→mesh path. Keep it that way: the KaoLRM path stays vendored, the edited path uses our own FLAME loader.
16. **Scale/translation application order.** `FLAMEParamsToMesh` applies scale and translation *outside* `FlameCore.forward` so `fix_z_trans=True` can zero translation z reliably regardless of what the FLAME core does with `zero_trans`. `KaoLRMReconstruct` goes through `model.flame2mesh(...)` which has its own code path — the two must stay numerically equivalent on a KaoLRM-only workflow. Covered by `test_flame_params_to_mesh.py::test_fix_z_trans_true_zeros_translation_z`; re-verify when `FlameCore` changes.
17. **Strength sliders are multiplicative, offsets are additive.** `FLAMEParamsEdit` clones `pose` before scaling jaw so the global slot stays untouched, then adds the global offsets. Translation offset applies pre-`fix_z_trans`, so `force_true` still zeros z downstream. If sliders ever gain a relative-to-neutral-face mode, spec it explicitly; don't redefine existing parameter semantics silently.
18. **FreeUV ShareAlike is viral.** CC BY-NC-SA 4.0 goes beyond the KaoLRM/FLAME non-commercial clause: every downstream artifact (UV albedo, final render, dataset, trained model) that incorporates a FreeUV output must be distributed under the same license on redistribution. The `i_understand_non_commercial` gate tooltip and `assets/README.md` call this out, but any UI banner for the full pipeline must keep ShareAlike visible — not fold it into a generic "non-commercial" notice.
19. **FreeUV is SD1.5 + ControlNet + detail_encoder — ~5 GB of weights.** The first-run download is larger than KaoLRM's and hits two separate HF repos (SD1.5 base + CLIP-L/14) plus two raw `.bin` files from the FreeUV release page. Missing-weight errors must name each path and URL separately — do not collapse them behind a single "FreeUV weights missing" message.
20. **`sys.path` injection for FreeUV, not `spec_from_file_location`.** FreeUV's `detail_encoder/` uses relative imports that only resolve when the vendor root is on `sys.path`. Loading submodules in isolation (as SMIRK/KaoLRM do) breaks the relative-import graph. Keep the divergence; if upstream FreeUV ever refactors to absolute imports, revisit.

---

## Verification

1. **Install.** `pip install -r requirements.txt` on a fresh venv (torch already present). Submodule or git install of KaoLRM resolves.
2. **Import smoke.** `python -c "from nodes.kaolrm_load import LoadKaoLRM"` works with KaoLRM installed, no weights needed.
3. **Weights resolution.** With weights at `models/kaolrm/mono.safetensors`, `LoadKaoLRM.execute()` returns a descriptor and warms the cache.
4. **Mesh smoke on real image.** Clean front portrait → `KaoLRMReconstruct` returns `MESH` with `vertices.shape == (1, 5023, 3)`, `faces.shape == (1, 9976, 3)`, params finite and non-zero.
5. **Preview sanity.** `MeshPreview` produces a visible silhouette + non-empty mask.
6. **CPU path.** Same smoke test on CPU with `dtype=fp32`. ~10× slower, acceptable for v0.1.
7. **Graceful failures.** Missing weights → `RuntimeError` with exact path; KaoLRM not installed → `ImportError` with install command; non-224 image → auto-resize with log.
8. **No regression on legacy helpers.** Registration still succeeds; existing `tests/test_flame_*` pass.
9. **SMIRK branch smoke.** With `SMIRK_em1.pt` at `models/smirk/` and the SMIRK source resolved, `LoadSMIRK.execute(i_understand_non_commercial=True)` returns a descriptor; `SMIRKPredict` on a front portrait returns `FLAME_PARAMS` with `pose[:, :3]` all zeros and `pose[:, 3:]` non-zero (jaw). `FLAMEParamsEdit(params=kaolrm, params_override=smirk)` → `FLAMEParamsToMesh` produces `vertices.shape == (1, 5023, 3)` with identity inherited from KaoLRM and mouth shape inherited from SMIRK. Sliders: `expression_strength=0.0` collapses to neutral mouth; `jaw_strength=0.0` closes the jaw while preserving expression; `scale_multiplier=0.5` shrinks the mesh uniformly.
10. **Pose-expansion sanity.** `_expand_pose_6_to_15` on `[0.1…0.6]` input produces `[0.1, 0.2, 0.3, 0, 0, 0, 0.4, 0.5, 0.6, 0, 0, 0, 0, 0, 0]`. Asserted in `test_flame_params_to_mesh.py`.
11. **FreeUV branch smoke.** With weights at `models/freeuv/{sd15,image_encoder_l,uv_structure_aligner.bin,flaw_tolerant_facial_detail_extractor.bin}` and the FreeUV checkout vendored or `FREEUV_ROOT` set, `LoadFreeUV.execute(i_understand_non_commercial=True)` returns a descriptor. `FreeUVGenerate.execute(descriptor, flaw_uv=uv_image_512)` returns a `[1, 512, 512, 3]` float32 IMAGE in [0,1]; the default reference UV (`assets/freeuv_reference_uv.jpg`) is loaded when `reference_uv` is not wired. Wiring an explicit `reference_uv` bypasses the bundled asset load. `seed=-1` samples a fresh seed per call; a fixed seed is deterministic.

---

## References

Primary: KaoLRM (3DV 2026), FLAME 2020 (MPI), SMIRK (CVPR 2024, `georgeretsi/smirk`). Roadmap papers (SMIRK, FreeUV, MoSAR, DECA, FFHQ-UV, Barré-Brisebois & Hill) and roles are in [`docs/pipeline-roadmap.md`](docs/pipeline-roadmap.md). Approved SMIRK-integration spec: [`plan/smirk-final-plan.md`](plan/smirk-final-plan.md). Approved FreeUV spec: [`plan/final-plan.md`](plan/final-plan.md). FLAME-Universe (github.com/TimoBolkart/FLAME-Universe) is the canonical index.
