# Pipeline Roadmap — WYSIWYG Studio Face Pipeline

Target state for the full production pipeline. Current code implements only Layer 0 (KaoLRM mesh); see `.claude/CLAUDE.md` for the authoritative current-state spec.

## Overview

Single image → fully-textured, relightable FLAME-topology face asset, delivered as a suite of ComfyUI custom nodes.

The core idea is **layered reconstruction**: no single model handles everything. KaoLRM provides coarse geometry; successive nodes add texture, expression detail, meso-scale pores, and intrinsic relighting maps — each stage uses the current SOTA for that specific layer.

## Target output

Per input face image:

- FLAME mesh (`.obj`, 5023 verts, standard FLAME UV layout)
- Diffuse albedo UV (2K, 16-bit PNG)
- Specular albedo UV (1K, 16-bit PNG)
- Ambient occlusion UV (1K, 8-bit PNG)
- Translucency / thickness UV (1K, 8-bit PNG)
- Combined normal UV (2K, 16-bit PNG, OpenGL convention)
- Displacement UV (2K, EXR)
- FLAME parameters (`.json` — shape, expression, pose, jaw)

All maps converge on **FLAME UV layout at the output boundary**. Any remap from other topologies (HiFi3D++, MoSAR's retargeted layout) happens inside nodes, never at the output.

## Node layers

```
Input image
    │
    ├─► [KaoLRM]            coarse FLAME params + mesh
    │       │
    │       └─► [SMIRK]     expression refinement on top of KaoLRM
    │                       (KaoLRM = identity/shape, SMIRK = expression)
    │
    ├─► [FreeUV]            albedo UV from image, FLAME-native
    │
    ├─► [MoSAR]             intrinsic decomposition:
    │                       diffuse, specular, AO, translucency, normal
    │
    ├─► [DECA detail]       expression-dependent displacement UV
    │                       (Stage 2 of DECA only — discard its coarse branch)
    │
    ├─► [Skin Detail Composite]
    │                       base normal ⊕ tiled pore-detail normal via RNM,
    │                       masked by canonical pore-density UV template
    │
    └─► [Export]            write all maps + .obj + params .json
```

## Layer responsibilities

### Layer 0 — Geometry (KaoLRM + SMIRK)

- **KaoLRM** handles identity and shape. View-consistent, clean FLAME params.
- **SMIRK** refines expression only. Do not let SMIRK override KaoLRM's shape — merge `shape ← KaoLRM`, `expression, jaw_pose ← SMIRK`.
- Output: FLAME params dict + posed mesh.

### Layer 1 — Albedo (FreeUV)

- Primary albedo source. Stable Diffusion v1.5 + Cross-Assembly inference.
- FLAME-native UV output, no remap needed.
- Native 1K; upscale to 2K via ESRGAN-skin or similar only at export.
- Fallback: FFHQ-UV lookup + HiFi3D++ → FLAME remap when FreeUV fails on extreme poses.

### Layer 2 — Intrinsic maps (MoSAR)

- Runs on the rendered front view of the KaoLRM mesh with FreeUV albedo applied.
- Outputs specular, AO, translucency, displacement, normal (in MoSAR's retargeted topology).
- **Retarget to FLAME UV** inside the node. Cache the retarget transform per session.

### Layer 3 — Expression detail (DECA Stage 2)

- DECA's detail encoder + decoder only. Feed it the KaoLRM params and input image.
- Output: 256×256 UV displacement for expression-specific wrinkles.
- Upsample to 2K with bicubic + high-frequency preservation before compositing.

### Layer 4 — Skin detail composite

- Canonical pore-density mask UV (authored once on FLAME layout, reused per subject).
- Texturing.xyz tileable pore normal — 8 tiles across face region.
- Blend: Reoriented Normal Mapping (Barré-Brisebois & Hill 2012).
- Strength: 0.6 default, exposed as node parameter.
- RNM math + tangent-space conventions in `skin_detail_compositing.md` (not yet authored).

## Critical technical constraints

### UV topology

- KaoLRM, SMIRK, DECA, EMOCA → FLAME native, no remap
- FreeUV → FLAME native, no remap
- MoSAR → retargeted topology, remap at node output
- FFHQ-UV → HiFi3D++, remap via a FLAME-equivalent of `run_flame_apply_hifi3d_uv.sh`
- Texturing.xyz pore tiles → tileable, no UV — projected via pore mask

**Eyeball vertices** (FLAME indices 3931–5022) need a separate eyeball texture. Do not let facial albedo leak onto eyeballs. Minimum: mask them to (1.0, 1.0) UV. Preferred: load a dedicated eyeball texture in the export node.

### Tangent-space conventions

- Internal: OpenGL convention (+Y up).
- Unreal export: flip green channel.
- Unity export: OpenGL, no flip.
- Target engine is an explicit UI toggle on the export node.

### Bit depth

- Normal maps: 16-bit PNG minimum — 8-bit visibly bands under rim light.
- Displacement: EXR (float32). Do not quantize.
- Albedo, specular, AO, translucency: 16-bit PNG for masters, 8-bit only for previews.

### GPU budget

Target hardware: 4× L40S (48GB each), CUDA 13.0.

- KaoLRM + SMIRK fit comfortably on one GPU.
- FreeUV (SD1.5-based) needs ~12GB with FP16.
- MoSAR fits on one GPU.
- DECA is tiny.
- Run FreeUV and MoSAR on separate GPUs in parallel when possible. Never assume single-GPU execution — use `device` params exposed by ComfyUI.

## Target repository layout

```
wysiwyg-face-pipeline/               # target layout; current repo is `comfyui-flame/` with flat `nodes/`
├── CLAUDE.md
├── README.md
├── custom_nodes/
│   ├── kaolrm_node/
│   ├── smirk_node/
│   ├── freeuv_node/
│   ├── mosar_node/
│   ├── deca_detail_node/
│   ├── skin_detail_composite_node/
│   └── face_export_node/
├── assets/
│   ├── pore_mask_flame_uv.png
│   ├── eyeball_texture.png
│   ├── flame_to_hifi3d_mapping/
│   └── pore_tiles/                  # Texturing.xyz assets (licensed)
├── docs/
│   ├── pipeline-roadmap.md          # this file
│   ├── skin_detail_compositing.md   # RNM math + tile masking
│   ├── uv_topology.md               # remap rules, seam handling
│   ├── node_reference.md            # input/output schemas per node
│   └── model_hashes.md              # pinned SHA256 per weight file
├── tests/
│   ├── fixtures/
│   └── test_pipeline_e2e.py
├── workflows/
│   └── full_pipeline.json
└── pyproject.toml
```

## Milestones

### M1 — Minimum viable

- KaoLRM + SMIRK + FreeUV + Export nodes end-to-end.
- Produces FLAME mesh + FLAME-UV albedo. No detail, no relighting maps.
- Reference workflow JSON checked in.

### M2 — Relightable

- MoSAR node producing specular, AO, translucency, normal on FLAME UV.
- Export node writes all intrinsic maps.
- First pass on eyeball texture handling.

### M3 — Detail

- DECA detail node (expression displacement).
- Skin detail composite node with canonical mask and RNM blend.
- Tangent-space convention toggle for Unreal / Unity export.

### M4 — Production hardening

- Multi-GPU scheduling across the 4× L40S.
- Batch mode: process a directory of images without re-loading weights.
- Robustness on extreme poses, occlusions, facial hair.
- Integration with Directus for asset cataloging.

## Pipeline-level gotchas

- **MoSAR's differentiable shading** expects specific spherical harmonics coefficients. KaoLRM uses different SH conventions. Convert at the MoSAR node's entry point; document the transform in `uv_topology.md`.
- **FreeUV + Poetry**: FreeUV ships with a Poetry config that pins `diffusers` to a version incompatible with the ComfyUI env. Port to `pip install -e` against the pinned env; do not run Poetry inside ComfyUI's Python.
- **FLAME eyeball UVs** default to (1,1) in many exporters → samples corner pixels. Explicit eyeball texture binding is required in any production render.

## Out of scope

- Hair, eyelashes, teeth as geometry. Hair is a separate track; teeth use the FLAME default texture.
- Full-body reconstruction. Face-only.
- Real-time performance. Target is offline VFX quality, ~30s–2min per frame acceptable.
- Animation retargeting. Output is a static posed asset plus FLAME params; downstream tools handle animation.
- Gen-Anima — excluded from this project. Any reference in issues, PRs, or docs should be removed.

## References

- KaoLRM (3DV 2026) — FLAME regression via LRM triplane priors.
- SMIRK (CVPR 2024) — emotion-aware FLAME reconstruction.
- FreeUV (CVPR 2025) — ground-truth-free UV recovery via SD + Cross-Assembly.
- MoSAR (CVPR 2024) — monocular semi-supervised avatar with differentiable shading.
- DECA (SIGGRAPH Asia 2021) — expression-dependent displacement on FLAME.
- FFHQ-UV (CVPR 2023) — UV texture dataset + HiFi3D++ ↔ FLAME remap.
- Barré-Brisebois & Hill (2012) — Reoriented Normal Mapping.
- FLAME-Universe (github.com/TimoBolkart/FLAME-Universe) — canonical resource index.
