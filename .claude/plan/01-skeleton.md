# Layer 2 — Intrinsic Maps Node (IntrinsicMapsBake) — Skeleton

## Scope

- In-scope v0.1: single `IntrinsicMapsBake` node that takes the KaoLRM `MESH` and a FreeUV-produced albedo `IMAGE` and emits specular, AO, translucency, and normal UV maps on FLAME topology.
- In-scope v0.1: a pluggable `IntrinsicsBackend` abstraction with a `procedural` backend (mesh-curvature / cavity / pore-mask signal pipeline) shipping by default.
- In-scope v0.1: stubbed `mosar` and `learned` backend slots that raise `NotImplementedError` with a clear upgrade path.
- In-scope v0.1: canonical assets bundled with the repo — FLAME pore-density mask UV, translucency template UV, FLAME UV coordinate cache.
- In-scope v0.1: non-commercial + no-derivatives gate surface (separate from the existing NC gate) for the future MoSAR backend.
- Explicitly out-of-scope: training any head on FFHQ-UV-Intrinsics; shipping MoSAR weights; DECA expression-displacement (Layer 3); pore-tile RNM composite (Layer 4); export node; batch size > 1; HiFi3D↔FLAME UV converter authoring.

## Context & MoSAR reality

Layer 0 (KaoLRM + SMIRK + `FLAMEParamsEdit` + `FLAMEParamsToMesh`) and Layer 1 (`LoadFreeUV` + `FreeUVGenerate`) have landed and are the upstream inputs for this node. This layer sits between FreeUV albedo and DECA detail (Layer 3) in `docs/pipeline-roadmap.md`.

The roadmap originally framed this layer as a "MoSAR node". Research on 2026-04-21 found four facts that break that shape (quoted verbatim from `project_mosar_integration_reality.md`):

> 1. **No MoSAR inference code or weights are released.** Only the `FFHQ-UV-Intrinsics` dataset (10k subjects) is public at `github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics`. The `ubisoft-laforge-MoSAR` URL redirects back to the dataset repo — the model repo does not exist as a separate public thing.
> 2. **MoSAR output UV is Hifi3D topology, not FLAME.** The dataset README explicitly references a separate conversion tutorial for users who want FLAME.
> 3. **MoSAR does its own geometry fitting** via a GNN-based non-linear morphable model — it is not a pure texture-on-given-mesh network, so it cannot be plugged in as a downstream texturizer on KaoLRM's mesh without re-training a texture-only head.
> 4. **License is CC BY-NC-ND** on the dataset (stricter than KaoLRM's CC BY-NC — no derivatives allowed).

Three-path framing (Q1 — load-bearing):

- (a) Train a texture/intrinsics head on FFHQ-UV-Intrinsics → FLAME UV. Requires HiFi3D→FLAME UV converter + legal read on the ND clause.
- (b) **Working assumption.** Drop MoSAR; ship `IntrinsicMapsBake` with a procedural backend now; leave `mosar` and `learned` slots pluggable for later.
- (c) Punt Layer 2 entirely; watch `ubisoft-laforge` for a weights drop; pull DECA forward.

## Node surface

`IntrinsicMapsBake` — single user-facing node. Category `KaoLRM` (migrates to `WYSIWYG/Face/...` at v0.2 reorg, matching SMIRK/FreeUV house convention). Display name `Intrinsic Maps Bake`.

- Inputs (sketch):
  - `mesh` — `io.Mesh` from KaoLRM / `FLAMEParamsToMesh`. Used for curvature, cavity, and UV-coordinate lookups.
  - `albedo_uv` — `io.Image` `[1, 512, 512, 3]` from FreeUV. Used as a colorimetric prior for specular / translucency.
  - `backend` — `io.Combo` `["procedural", "mosar", "learned"]`, default `procedural`.
  - `specular_strength`, `ao_strength`, `translucency_strength`, `normal_strength` — `io.Float` sliders, 0.0–1.5, default 1.0 each.
  - `pore_density_mask` — optional `io.Image` override. Falls back to bundled canonical FLAME pore-density UV.
  - `output_resolution` — `io.Combo` `["512", "1024", "2048"]`, default `1024`. (Open question Q3.)
  - `seed` — `io.Int`, default `-1` for stochastic procedural variations (pore placement jitter, noise octaves).
- Outputs (Q2 — open): either 5 separate `io.Image` wires (`specular`, `ao`, `translucency`, `normal`, optionally `displacement`) or a single `INTRINSIC_MAPS` custom dict wire. Resolve at draft time.

## Backend interface

Minimal ABC/Protocol in `nodes/intrinsic_backends/__init__.py`:

```
class IntrinsicsBackend(Protocol):
    name: str
    def bake(self, mesh, albedo_uv, aux: dict) -> dict[str, torch.Tensor]: ...
```

- `procedural` (v0.1, default) — `nodes/intrinsic_backends/procedural.py`.
- `mosar_stub` — `nodes/intrinsic_backends/mosar_stub.py`. Raises `NotImplementedError` naming `ComfyUI/models/mosar/` and the upstream tracking issue; consumes a future `MOSAR_MODEL` descriptor from a parallel `LoadIntrinsicsMoSAR` node (see §8).
- `learned_stub` — `nodes/intrinsic_backends/learned_stub.py`. Raises `NotImplementedError`. Q7 — which learned candidate (AvatarMe++, NextFace, other)?

Backend instances are module-scope cached; `IntrinsicMapsBake` dispatches via a name → class registry.

## Procedural backend — per-map signal sketches

### Specular

- Signal source: cavity (inverted AO) × albedo luminance inverse (pores + crevices are shinier than sebum-rich flats — but this heuristic needs review).
- Bake strategy: rasterize per-vertex cavity into FLAME UV via the cached `flame_uv.npz` barycentric map.
- Open: whether to clamp against a lips/eyebrow mask so dark hair doesn't read as high-spec.

### AO

- Signal source: mesh curvature (Laplacian-based) + short-ray ambient occlusion sampled on the FLAME mesh.
- Bake strategy: vertex-space AO → UV via barycentric raster.
- Open: whether to use pytorch3d rays (opt-in dep) or a numpy/torch analytic cavity proxy.

### Translucency

- Signal source: canonical FLAME translucency template UV (authored once, shipped as asset), modulated by albedo luminance.
- Bake strategy: element-wise blend of template and albedo-derived thickness proxy.
- Open: how the template is authored — hand-painted, baked from a reference head, or generated by signed-distance from bone landmarks.

### Normal

- Signal source: mesh-derived tangent-space normals (flat baseline) combined with meso-band noise modulated by pore-density mask.
- Bake strategy: compute vertex normals → rasterize to tangent-space UV; add low-amplitude noise in high-density pore regions only. Full pore-tile RNM composite is Layer 4 and **not** part of this node.
- Open: OpenGL (+Y) vs DirectX (-Y) convention — default OpenGL per roadmap, export-time flip deferred to Export node.

### Displacement (Q4 — open)

- Candidate signal: curvature + noise, or deferred entirely to DECA Stage 2 (Layer 3).
- Open: whether a coarse procedural displacement belongs in Layer 2 at all, given DECA's expression-specific displacement is a strict superset. Resolve at draft time.

## Assets required

| Asset | Path | Format | Authoring owner | Blocker trigger |
|---|---|---|---|---|
| FLAME pore-density mask UV | `assets/flame_pore_density_uv.png` | 8-bit PNG, 1024² | Q5 — hand-painted vs. baked | Any procedural output uses it |
| FLAME translucency template | `assets/flame_translucency_template.png` | 8-bit PNG, 1024² | Q5 | Translucency output |
| Texturing.xyz pore tile set | `assets/pore_tiles/` | 16-bit TIFF/EXR | Licensed bundle — Q8 | Only if pore tiles inform Layer 2; otherwise Layer-4-only |
| FLAME UV coordinate cache | `assets/flame_uv.npz` | NPZ with `uv_coords[5023,2]`, `faces_uv[F,3]`, barycentric raster LUT | Derivable from generic FLAME pkl | Any backend that rasterizes vertex-space signals to UV |

## Weights / vendor layout

Reserve `ComfyUI/models/mosar/` per the one-model-type-per-folder convention in CLAUDE.md. Stays empty in v0.1. The `mosar_stub` backend's `NotImplementedError` points at this path to give a stable surface when weights eventually drop. No `third_party/mosar/` vendor dir in v0.1; add when code + weights become public. No `requirements.txt` additions — procedural backend uses `torch` + numpy + PIL already in the tree.

## Non-commercial + no-derivatives gate

- Procedural backend has no NC/ND surface — it ships under the repo's license and derives from no restricted dataset. No new checkbox needed on `IntrinsicMapsBake` itself.
- `LoadIntrinsicsMoSAR` (new node, parallel to `LoadFreeUV`, lazily introduced when the `mosar` backend becomes real) surfaces a second gate: `i_understand_non_commercial_no_derivatives`. Distinct from the existing `i_understand_non_commercial` because CC BY-NC-ND is strictly stricter than KaoLRM's CC BY-NC 4.0 and FreeUV's CC BY-NC-SA 4.0 — the ND clause likely forbids generating derivative maps at all, and the gate must name that risk verbatim.
- `assets/README.md` updated at MoSAR-backend landing time to document the ND escalation.

## Integration with existing pipeline

```
LoadKaoLRM ─► KaoLRMPreprocess ─► KaoLRMReconstruct ─► MESH ─┐
                                                             ├─► IntrinsicMapsBake ─► (specular, AO, translucency, normal, [displacement])
LoadFreeUV ─► FreeUVGenerate ─► IMAGE (albedo_uv) ───────────┘                                   │
                                                                                                  ▼
                                                                                      (future Export node)
```

`IntrinsicMapsBake` sits immediately downstream of FreeUV and before DECA (Layer 3) and the skin composite (Layer 4).

## Tests

- `tests/test_intrinsic_maps_bake.py` — node-level: backend dispatch, slider scaling, shape/range of each output on a synthetic 5023-vert mesh + 512² albedo, rejection of batch > 1, seed determinism.
- `tests/test_intrinsic_backends_procedural.py` — procedural backend per-map shape/range tests with no real weights.
- `tests/test_intrinsic_backends_stubs.py` — `mosar_stub` + `learned_stub` raise `NotImplementedError` with messages naming the expected future paths.
- `tests/test_intrinsic_assets.py` — canonical FLAME pore-density mask + translucency template loaders + `flame_uv.npz` cache.
- Any future learned-backend tests gated `@pytest.mark.slow` per house policy.
- All v0.1 tests pass with no real weights present.

## Brutal review placeholders

1. ND-clause legal risk on the FFHQ-UV-Intrinsics derivative path (option (a) of the three-path choice).
2. HiFi3D → FLAME UV remap complexity if a MoSAR or FFHQ-UV-Intrinsics backend ever lands.
3. Quality gap: procedural specular/AO vs. learned baselines is likely large on non-Caucasian or bearded subjects.
4. Pore-density-mask authoring cost — one artist-week minimum; bitrots if FLAME UV changes.
5. Tangent-space convention leakage between internal OpenGL and target-engine output.
6. Eyeball UV leakage at (1,1) — roadmap calls this out, but Layer 2 outputs are the first UV maps that will actually sample those corners.
7. Scope creep: each stubbed backend is an invitation for users to file "please implement" issues.
8. Dependency cost if procedural AO requires pytorch3d — keep it optional, same pattern as `MeshPreview`.

## Open questions

- **Q1.** Three-path decision: (a) train on FFHQ-UV-Intrinsics, (b) procedural-first with pluggable backend slots (working assumption), or (c) punt the layer? Resolve before draft to avoid writing a plan the user wants redirected.
- **Q2.** Output wire shape: 5× `io.Image` wires vs. one `INTRINSIC_MAPS` custom dict type? Custom type composes cleaner with a future Export node; 5 wires are more ComfyUI-idiomatic.
- **Q3.** Output resolution default — 512² (match FreeUV), 1024² (match roadmap specular/AO bit-depth target), or always bake at 1024² internally and downsample on output?
- **Q4.** Does coarse displacement belong in Layer 2 or Layer 3 (DECA)? DECA is expression-specific; a neutral displacement might still be useful as a base layer.
- **Q5.** Canonical FLAME pore-density mask and translucency template — hand-painted, baked from a scanned reference head, or procedurally generated from landmarks? Who authors them?
- **Q6.** Single-GPU vs. multi-GPU scheduling claim in the roadmap (FreeUV + MoSAR parallel across L40S) — does the procedural backend make this moot for v0.1, and does the pluggable backend interface need to preserve a device parameter for the eventual learned backend?
- **Q7.** Learned-backend identity — AvatarMe++, NextFace, an intrinsic-decomposition net, or leave as a plain `NotImplementedError` stub with no specific candidate?
- **Q8.** Texturing.xyz licensing stance for shipping pore tiles in-repo vs. requiring manual user acquisition. May be entirely Layer 4's problem, not ours.
- **Q9.** Does `IntrinsicMapsBake` need the input albedo at all for the procedural backend, or is it purely a geometry-driven bake? If geometry-only, the node signature simplifies.
- **Q10.** Canonical FLAME UV cache (`flame_uv.npz`) — derive it on first use from the generic FLAME pkl, or ship it pre-baked as a repo asset?

## Milestones

- **M0 — backend scaffold.** `IntrinsicsBackend` protocol, `procedural` backend returning zero-valued maps of correct shape, two stub backends raising `NotImplementedError`, node registered, full unit-test suite green. Exit: tests pass with no real weights.
- **M1 — procedural signal quality.** Specular / AO / translucency / normal generate plausible maps on a KaoLRM mesh + FreeUV albedo. Exit: visual smoke test on at least 3 reference portraits; canonical assets shipped.
- **M2 — backend plug-in readiness.** `LoadIntrinsicsMoSAR` node scaffolded with ND gate; `mosar` backend wires up to `MOSAR_MODEL` descriptor even while stubbed. Exit: the day MoSAR drops, only the backend body is missing — the surface is ready.

## References

- MoSAR — Dib et al., *MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction*, CVPR 2024.
- FFHQ-UV-Intrinsics — Ubisoft La Forge, `github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics`. CC BY-NC-ND.
- AvatarMe++ — Gecer et al., *AvatarMe++: Facial Shape and BRDF Inference with Photorealistic Rendering-Aware GANs*, TPAMI 2021.
- Barré-Brisebois & Hill, *Blending in Detail* (Reoriented Normal Mapping), 2012.
- NextFace — Dib et al., *Practical Face Reconstruction via Differentiable Ray Tracing*, Eurographics 2021.
- DECA — Feng et al., *Learning an Animatable Detailed 3D Face Model from In-the-Wild Images*, SIGGRAPH 2021.
- FLAME-Universe — `github.com/TimoBolkart/FLAME-Universe`.
