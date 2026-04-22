# FreeUV Node Suite — Layer 1 Albedo UV Generation (v0.1)

## Scope
- Thin ComfyUI wrapper over FreeUV's `inference.py` stack (SD1.5 UNet + ControlNet UV structure aligner + detail encoder + DDIM scheduler).
- New node suite matching the SMIRK pattern: runtime resolver, `LoadFreeUV` descriptor node, and a single inference node that consumes two pre-supplied 512×512 UV IMAGEs and emits a 512×512 albedo UV IMAGE.
- Weight/config discovery for SD1.5 base, CLIP-ViT-L/14 image encoder, and the two FreeUV `.bin` checkpoints.
- License integration: surface FreeUV's CC BY-NC-SA 4.0 ShareAlike escalation on top of the existing non-commercial stack.
- Registration via the V3 `ComfyExtension` in `__init__.py`; category stays `KaoLRM` for v0.1 parity.

**Out of scope (v0.2+):** `FLAMEProjectToUV` (render KaoLRM mesh + photo → flaw UV) is deferred; v0.1 takes both UV inputs as IMAGEs. No data-process or training code. No MoSAR/DECA integration. No batch-size >1.

## Assumptions
- Upstream FreeUV (`YangXingchao/FreeUV`) inference API is stable at the commit we pin; `detail_encoder.generate(...)` signature and `.bin` key names match `main` at time of spec authoring.
- `detail_encoder/` must be importable as a Python package (uses relative imports `._clip`, `.attention_processor`, `.resampler`). Runtime resolver will need `sys.path` injection, not SMIRK-style isolated `spec_from_file_location`.
- Output is **512×512** (roadmap says "native 1K" — this is a known roadmap correction).
- SD1.5 weights live under `ComfyUI/models/freeuv/sd15/` by default; sharing with existing ComfyUI SD1.5 checkpoints is a planning decision (see Q1).
- `torch_dtype=torch.float32` matches upstream default; fp16 safety is unverified for ControlNet + detail_encoder path.

## Sections
1. Node boundaries & wire types — enumerate `LoadFreeUV` + `FreeUVInfer` (name TBD); reuse `io.Image` for both UV inputs and output; introduce `FREEUV_MODEL` custom descriptor only.
2. Runtime resolution (`nodes/freeuv_runtime.py`) — env var / vendored `third_party/freeuv/` / installed-package fallback; document the package-loading divergence from SMIRK.
3. Weight & config discovery (`LoadFreeUV`) — paths for SD1.5, CLIP-ViT-L/14, `uv_structure_aligner.bin`, `flaw_tolerant_facial_detail_extractor.bin`; address shared-vs-self-contained SD1.5.
4. `LoadFreeUV` node schema — inputs (device, dtype, non-commercial gate with ShareAlike wording), output `FREEUV_MODEL` descriptor, lazy build contract.
5. `FreeUVInfer` node schema + cache — inputs (`uv_structure_image`, `flaw_uv_image`, `seed`, `guidance_scale`, `num_inference_steps`), `_FREEUV_CACHE` key, forward flow mirroring `inference.py`, PIL↔IMAGE conversion.
6. File layout & registration — new `nodes/freeuv_*.py` files, `nodes/__init__.py` additions, `__init__.py` extension diff, category decision.
7. Testing strategy — mirror `tests/test_smirk_*.py`; mock pipeline + detail_encoder; descriptor and schema smoke tests; no real weights required.
8. Non-commercial + ShareAlike gate & licensing — tooltip wording, `assets/FREEUV_LICENSE.txt`, pipeline-wide ShareAlike implications for exports.
9. Known corrections — roadmap's "native 1K" is actually 512×512; `runwayml/stable-diffusion-v1-5` → `stable-diffusion-v1-5/stable-diffusion-v1-5` after HF deprecation.
10. Open questions — blockers for Phase 2 drafting (see below).

## Open Questions
- Q1. Does `LoadFreeUV` allow pointing at an existing ComfyUI SD1.5 checkpoint (reducing ~5 GB download), or require a self-contained `models/freeuv/sd15/` directory? HF snapshot dir vs single-file `.safetensors`?
- Q2. Bundle `data-process/resources/uv.jpg` as a default reference UV asset in `assets/`, or always require the user to supply both UV IMAGEs?
- Q3. `detail_encoder` package loading: vendor-only (`third_party/freeuv/` with `sys.path` injection), isolated `importlib` on a submodule, or allow installed-package fallback? How do we reconcile with SMIRK/KaoLRM's `spec_from_file_location` pattern given the relative imports?
- Q4. Expose `torch_dtype` (`fp32`/`fp16`/`bf16`) like other nodes, or hardcode fp32 per upstream? Any fp16 underflow risk on ControlNet/detail_encoder outputs analogous to Brutal Review #11?
- Q5. CLIP-ViT-L/14 image encoder — fixed `models/freeuv/image_encoder_l/` subdir, or reuse ComfyUI's `clip_vision/` directory via `folder_paths`?
- Q6. Enforce batch size 1 (SMIRK parity), or support `detail_encoder.generate`'s internal iteration batch loop for batch > 1?
- Q7. License escalation: does CC BY-NC-SA 4.0 require a whole-pipeline LICENSE statement (`assets/README.md` or repo `LICENSE`), and what does it mean for exported meshes/textures that touch this node?
- Q8. Final node name: `FreeUVInfer`, `FreeUVGenerate`, `FreeUVAlbedo`, or other? (SMIRK used `SMIRKPredict`; KaoLRM used `KaoLRMReconstruct`.)
- Q9. Commit hash + SHA256 pins for FreeUV repo and the two `.bin` checkpoints — blocker for first release, mirrors SMIRK's open BLOCKER.
- Q10. DDIM scheduler is swapped in after pipeline construction upstream — expose the scheduler as a user choice or lock to DDIM in v0.1?

## Success Criteria
- `01-skeleton.md` approved → researcher can expand each section into `final-plan.md` without re-debating scope.
- All 10 open questions have approved answers before Phase 2 drafting starts.
- Resulting spec lets an implementer land `LoadFreeUV` + `FreeUVInfer` with tests green against mocked pipeline/detail_encoder, matching `tests/test_smirk_*.py` depth (descriptor shape, schema gate, cache key, error messages naming exact paths + URLs).
- Non-commercial + ShareAlike gate wording is unambiguous enough that a commercial user cannot miss the escalation over the existing FLAME/KaoLRM constraints.
