# CLAUDE.md

Project guidance for Claude Code working on the WYSIWYG Studio ComfyUI face reconstruction pipeline.

**Current state.** A KaoLRM-based node suite that produces a FLAME-topology mesh from a single image. Mesh only — no texture, no relighting maps yet. Scaffolded in `nodes/kaolrm_*.py`, registered via the V3 `ComfyExtension` pattern in `__init__.py`.

**Target state.** A 5-layer WYSIWYG Studio pipeline — KaoLRM geometry → FreeUV albedo → MoSAR intrinsic maps → DECA expression detail → pore composite → export — producing fully-textured, relightable FLAME assets. Full roadmap in [`docs/pipeline-roadmap.md`](../docs/pipeline-roadmap.md).

**Rule of thumb.** Work forward from the KaoLRM mesh path. The old FLAME editor/viewer nodes are not the active product path, but `nodes/flame_core.py` and `nodes/flame_render_util.py` are still active dependencies of the KaoLRM scaffold and must not be treated as dead code.

---

## When the user gives a planning topic

For requests like "plan a spec for X" / "draft a design for Y", switch to orchestrator mode and run the multi-agent markdown planning loop. Read [`.claude/planning-workflow.md`](planning-workflow.md) before acting.

---

## Current implementation — KaoLRM mesh suite

### Node pipeline

```
LoadKaoLRM ─► KaoLRMPreprocess (optional) ─► KaoLRMReconstruct ─► MESH ─► MeshPreview
```

`KaoLRMPreprocess` does resize and optional `rembg` background removal (opt-in — `rembg` ships its own heavy weights). `MeshPreview` rasterizes the mesh for visual verification; it's the terminal node while there's no texture work to feed into.

### Wire types

- `KAOLRM_MODEL` — `io.Custom("KAOLRM_MODEL")`, a plain dict descriptor `{variant, ckpt_path, flame_pkl_path, device, dtype, kaolrm_root}`. The heavy model is cached in `_KAOLRM_CACHE` inside `nodes/kaolrm_reconstruct.py`, keyed by `(variant, device, dtype, ckpt_path, flame_pkl_path, kaolrm_root)`.
- `MESH` on the wire is the **built-in** `io.Mesh` (`MeshPayload` from `comfy_api.latest._util`), carrying `{vertices[B,V,3], faces[B,F,3]}` plus ad-hoc attrs `flame_params`, `gender="generic"`, `source_resolution=224`, `topology`, `num_sampling`, and the original FLAME topology in `base_vertices`/`base_faces` when a sampled point cloud is emitted. `nodes/mesh_types.py` declares an unused `io.Custom("MESH")` alongside live helpers (`coerce_mesh`, `compute_vertex_normals`) — the custom type is vestigial and can be deleted at cleanup.

### Nodes

- **`LoadKaoLRM`** (`nodes/kaolrm_load.py`) — resolves `model.safetensors` + FLAME pkl and returns the descriptor. Heavy build is lazy in `KaoLRMReconstruct`, not here. Inputs: `variant` (`mono`|`multiview`, default `mono`), `device` (`auto`|`cpu`|`cuda`), `dtype` (`auto`|`fp32`|`fp16`|`bf16`; `auto` → `fp16` on CUDA, forced `fp32` on CPU), `i_understand_non_commercial` boolean gate (must be `True` — see Brutal Review #1).
- **`KaoLRMPreprocess`** (`nodes/kaolrm_preprocess.py`) — resize to 224×224, then optional `rembg` via cached session (`_get_rembg_session`), composite against `background_color` with `composite_alpha`, resize mask in lockstep. Default `remove_background=False`. Outputs `IMAGE`, `MASK`.
- **`KaoLRMReconstruct`** (`nodes/kaolrm_reconstruct.py`) — the mesh predictor. Inputs include `source_cam_dist` float (default `2.0`) and `num_sampling` int (default `5023`, the FLAME vertex count). Background removal / rembg model choice live on `KaoLRMPreprocess`, not here. Output: built-in `MESH` with attrs, all tensors on CPU. When `num_sampling != 5023`, the output switches to a sampled point cloud and preserves the original FLAME topology in `base_vertices`/`base_faces`.
- **`MeshPreview`** (`nodes/mesh_preview.py`) — rasterizes `MESH` to IMAGE+MASK. Default renderer `soft_torch`; `pytorch3d` opt-in when installed. Uses `render_mesh()` for triangle meshes and `render_points()` for sampled point clouds. Depends on `nodes/flame_render_util.py`, which is active shared code.

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

- `nodes/kaolrm_load.py`, `nodes/kaolrm_preprocess.py`, `nodes/kaolrm_reconstruct.py`, `nodes/kaolrm_runtime.py`, `nodes/kaolrm_mesh_model.py` — node suite + runtime (KaoLRM runtime resolution, optional path injection, `import_kaolrm_symbols`, mesh-only `KaoLRMMesh` wrapper, `load_mesh_only_model`).
- `nodes/mesh_types.py` — `coerce_mesh`, `compute_vertex_normals` helpers.
- `nodes/mesh_preview.py` — `MeshPreview` node.

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
- `ComfyUI/models/flame/generic_model.pkl`

### Required framework deps (`requirements.txt`)

`huggingface_hub`, `safetensors`, `einops`, `rembg`, `chumpy`. Keep `pytorch3d`, `xformers`, `diff-surfel-rasterization` **out** of required deps — they're only needed for the Gaussian splat video path we don't use.

### Dependency integration strategy for KaoLRM itself

Supported runtime resolution order:

- **Option A (preferred)** — vendor a pinned checkout under `third_party/kaolrm/`.
- **Option B** — install upstream `kaolrm` into the active Python environment.
- **Option C** — set `KAOLRM_ROOT=/abs/path/to/KaoLRM` for local development.

`LoadKaoLRM` no longer requires the runtime checkout up front; the actual import/build path is resolved lazily in `KaoLRMReconstruct`.

### Testing

- Per-node unit tests: fixed input → expected output shape/range. Existing: `test_kaolrm_load.py`, `test_kaolrm_preprocess.py`, `test_kaolrm_reconstruct.py` (mocked model), `test_kaolrm_runtime.py`, `test_mesh_type.py`, `test_flame_core.py`, `test_flame_params.py`, `test_render.py`, `test_routes.py`.
- Run with `~/github/ComfyUI/venv311/bin/python -m pytest tests/` — all should pass with no real weights present.
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
- `i_understand_non_commercial` gate on `LoadKaoLRM` (Brutal Review #1).
- Missing-weight / missing-FLAME error messages name the exact path and upstream URL (Brutal Review #2).
- `rembg` uses cached sessions via `new_session(model_name)`, opt-in by default.
- KaoLRM import surface isolated: `nodes/kaolrm_runtime.py` + individual-module loading via `importlib.util.spec_from_file_location` in `kaolrm_mesh_model.py` — sidesteps upstream's top-level `__init__.py` (Brutal Review #3 resolved).
- Tests landed: `tests/test_kaolrm_load.py`, `tests/test_kaolrm_preprocess.py`, `tests/test_mesh_type.py` (covers `coerce_mesh` / `compute_vertex_normals` helpers).

### Open

- Vendor `third_party/kaolrm/` as a pinned submodule for the default portable install. The runtime can also come from an installed `kaolrm` package or `KAOLRM_ROOT`, but vendoring is still the most reproducible default.
- Ship `assets/KAOLRM_LICENSE.txt` + `assets/EG3D_LICENSE.txt` alongside the existing `assets/FLAME_LICENSE.txt`.
- Keep `workflows/flame_basic.json` aligned with the registered node set; it is now the KaoLRM mesh-preview example.
- Comment block in `requirements.txt` documenting the manual KaoLRM install (submodule vs `pip install git+...@<sha>`).
- `tests/test_mesh_preview.py` — render-smoke against a synthetic 5023-vert mesh. `test_mesh_type.py` covers the helpers but not the node.

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
13. **Plan/doc drift.** When behavior changes, update this file in the same PR or mark the old section historical. Example history: `source_cam_dist` moved from `LoadKaoLRM` to `KaoLRMReconstruct`; `remove_background`/`rembg_model` moved from `KaoLRMReconstruct` to `KaoLRMPreprocess`.

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

---

## References

Primary: KaoLRM (3DV 2026), FLAME 2020 (MPI). Roadmap papers (SMIRK, FreeUV, MoSAR, DECA, FFHQ-UV, Barré-Brisebois & Hill) and roles are in [`docs/pipeline-roadmap.md`](../docs/pipeline-roadmap.md). FLAME-Universe (github.com/TimoBolkart/FLAME-Universe) is the canonical index.
