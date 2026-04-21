# ComfyUI KaoLRM Mesh Framework

Single-image FLAME mesh reconstruction for ComfyUI, centered on a KaoLRM pipeline.

## Status

The repo now has the overall framework for the intended pipeline:

- `Load KaoLRM`
- `KaoLRM Preprocess`
- `KaoLRM Reconstruct`
- `Mesh Preview`

The older FLAME editor/viewer nodes are not the active product path right now, but shared helpers in `nodes/flame_core.py` and `nodes/flame_render_util.py` are still used by the KaoLRM scaffold.

## Non-Commercial Notice

KaoLRM checkpoints and the required FLAME assets are non-commercial. Keep that constraint in mind before distributing outputs or integrating this package into a commercial workflow.

## Planned Pipeline

```text
IMAGE -> KaoLRM Preprocess -> KaoLRM Reconstruct -> MESH -> Mesh Preview
             optional               requires KaoLRM
```

`KaoLRM Preprocess` currently handles the 224x224 safety-net resize. The background-removal branch is scaffolded but not wired yet, so the safe default is `remove_background=False`. `KaoLRM Reconstruct` lazily resolves the upstream KaoLRM runtime from `third_party/kaolrm`, an installed `kaolrm` package, or `KAOLRM_ROOT`. The default output is the 5023-vertex FLAME mesh; higher `num_sampling` values emit a sampled point cloud plus the original FLAME topology as metadata.

## Model Paths

Expected runtime paths inside ComfyUI:

- `ComfyUI/models/kaolrm/mono.safetensors`
- `ComfyUI/models/kaolrm/multiview.safetensors`
- `ComfyUI/models/flame/generic_model.pkl`

If any of those files are missing, the loader fails with a direct path-specific error.

## Dependencies

Runtime requirements now track the KaoLRM framework direction:

- `safetensors`
- `einops`
- `rembg`
- `chumpy`

`pytorch3d` remains optional. `Mesh Preview` defaults to the shipped `soft_torch` backend.

## Current Scope

- Mesh-only workflow
- No texture generation
- No Gaussian-splat rendering
- No automatic download for gated FLAME assets
- No vendored KaoLRM source yet, though an installed `kaolrm` package or `KAOLRM_ROOT` also works

## Development

- Active framework nodes live under [`nodes/`](nodes)
- Mesh rendering helper is still reused from [`nodes/flame_render_util.py`](nodes/flame_render_util.py)
- The detailed approved plan lives in [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
