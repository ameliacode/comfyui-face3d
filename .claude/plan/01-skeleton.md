# SMIRK Node Suite — Layer 0 Expression Refinement on KaoLRM

## Scope
- SMIRK predictor node: IMAGE → FLAME params (expression, jaw_pose; also shape, pose, eye_pose if upstream returns them).
- Param-merge node: combine KaoLRM shape/identity with SMIRK expression/jaw_pose into a single `FLAME_PARAMS` dict.
- Mesh re-solve strategy after merge (new node vs. reuse of `KaoLRMReconstruct`'s FLAME forward path).
- Runtime/source resolution (env var, vendored `third_party/smirk`, installed package) mirroring `nodes/kaolrm_runtime.py`.
- Weight/config discovery under `models/smirk/` with explicit missing-file errors; non-commercial gate surfacing.
- Unit tests with synthetic dummies; V3 `ComfyExtension` registration; cache-key design.
- **NOT covered**: face crop/alignment preprocessing details (assumed separate node or reused from KaoLRM), texture/UV work, DECA detail, multi-view SMIRK variants, training/fine-tuning.

## Assumptions
- SMIRK upstream exposes a predictor producing per-frame FLAME parameter tensors compatible with FLAME 2020 shape/expression/pose dims used by KaoLRM (100/50/6). *Unverified — see Q1.*
- SMIRK's face crop expectation differs from KaoLRM's 224² canonical crop and needs an explicit preprocess step. *Likely — see Q2.*
- `KaoLRMReconstruct` already emits the FLAME pkl + params in a form re-usable for a second forward pass; the FLAME forward can be factored out without pulling in the triplane encoder.
- SMIRK weights are research/non-commercial — same `i_understand_non_commercial` gate pattern applies.
- Merge policy is fixed by the roadmap: `shape ← KaoLRM`, `expression, jaw_pose ← SMIRK`; `pose (neck/global)`, `scale`, `translation` remain from KaoLRM unless overridden.

## Sections
1. Node Boundaries and Wire Types — SMIRK_MODEL descriptor, FLAME_PARAMS wire type, MESH re-emission rules.
2. Runtime Resolution — `nodes/smirk_runtime.py` mirroring `kaolrm_runtime.py` (env `SMIRK_ROOT`, `third_party/smirk`, installed package, isolated import surface).
3. Weight and Config Discovery — `models/smirk/` layout, filename constants, missing-file error contract.
4. LoadSMIRK Node — descriptor shape, non-commercial gate, device/dtype resolution, lazy build deferred to predictor.
5. SMIRKPredict Node — preprocess contract, module-scope cache, `_SMIRK_CACHE` key, forward call, param tensor normalization to FLAME_PARAMS.
6. FLAMEParamsMerge Node — merge policy, field-by-field override table, dim-compatibility checks, pass-through of `scale`/`translation`.
7. Mesh Re-solve Strategy — decision between (a) a new `FLAMEParamsToMesh` node that loads only the FLAME pkl, or (b) extending `KaoLRMReconstruct` to accept optional override params. Covers chumpy shim reuse and FLAME model caching.
8. Registration and Category — V3 `ComfyExtension` entry additions, category naming (`KaoLRM` vs. new `WYSIWYG/Face/SMIRK`).
9. Testing Strategy — synthetic dummy predictor, merge correctness tests, runtime resolution tests, missing-weight error tests, mesh re-solve round-trip.
10. Non-commercial and Licensing — gate propagation, `assets/SMIRK_LICENSE.txt`, tooltip red-text, roadmap M1 alignment.

## Open Questions
- Q1: Which SMIRK fork/commit is canonical (georgeretsi/smirk vs. other) and what is the exact predictor API signature — does it return a dict with keys matching KaoLRM's `{shape, expression, pose, ...}` or a different schema (e.g., separate `jaw_pose`, `eye_pose`, `neck_pose`)?
- Q2: What input size/crop does SMIRK expect, and does it assume MediaPipe/FAN landmarks upstream? Does the user need a SMIRK-specific preprocess node, or can `KaoLRMPreprocess` output be reused?
- Q3: Does SMIRK's shape dim match FLAME 2020's 100 used by KaoLRM, or a different count (e.g., 300)? If mismatched, how should the merge node project/truncate?
- Q4: Does SMIRK split `pose` into separate `jaw_pose` / `neck_pose` / `global_pose`, and is the 6-dim KaoLRM `pose` axis-angle concat of global+jaw (standard FLAME) or something custom?
- Q5: Mesh re-solve — prefer a dedicated `FLAMEParamsToMesh` node, or overload `KaoLRMReconstruct` with an optional `flame_params_override` input? Which better fits V3 Comfy conventions and caching?
- Q6: Should `FLAME_PARAMS` be a new `io.Custom("FLAME_PARAMS")` wire, or piggy-back on the existing `MESH.flame_params` attr channel?
- Q7: Does SMIRK pull heavy transitive deps (mediapipe, face-alignment, pytorch3d) that should be kept optional via import isolation like `kaolrm_mesh_model.py`?
- Q8: Are SMIRK weights redistributable under its license, or must users fetch from an upstream gated release? Determines whether `models/smirk/` instructions mirror KaoLRM's manual-drop or FLAME's registration wall.
- Q9: Batch semantics — SMIRK is per-frame; should the node enforce batch==1 like `KaoLRMReconstruct` or allow batched prediction?

## Success Criteria
- `LoadSMIRK` + `SMIRKPredict` + `FLAMEParamsMerge` + a mesh re-solve node registered via V3 `ComfyExtension`; `python -c "from nodes.smirk_load import LoadSMIRK"` succeeds with no weights present.
- Missing weight/config paths raise `RuntimeError` naming exact path and upstream URL.
- With a dummy predictor (tests), IMAGE → FLAME_PARAMS → merged params → 5023-vert FLAME mesh round-trips with `vertices.shape == (1, 5023, 3)`.
- Merge policy verified: merged `shape` equals KaoLRM's; merged `expression`/`jaw_pose` equal SMIRK's; other fields follow the documented override table.
- Non-commercial gate blocks execution until acknowledged; workflow JSON example checked in showing KaoLRM → SMIRK → Merge → Mesh → MeshPreview.
- All open questions Q1–Q9 either resolved or explicitly deferred with a named owner before the researcher drafts content.
