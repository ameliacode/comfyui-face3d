# FLAMEProjectToUV — Integration Spec

Single node. Projects the KaoLRM input IMAGE into FLAME UV space using the
same camera that KaoLRM used, producing the `flaw_uv_image` that
`FreeUVGenerate` consumes. No other pipeline layers in scope.

---

## Outstanding Blockers

Two items must clear before this node can be merged. They are also flagged
inline as HTML comments where they bite.

| # | Blocker | Owner | Exit criteria |
|---|---|---|---|
| B1 | **MPI license on `assets/flame_uv_template.npz`** — the FLAME 2020 UV layout (`head_template.obj`-derived `vt`/`ft`) may still carry MPI non-commercial terms even when repackaged as `.npz`. Do not commit the asset until confirmed. | First integrator with FLAME account access | Written confirmation from MPI or FLAME terms-of-use page that redistribution of the UV layout is allowed under the same non-commercial scope as the pkl. |
| B2 | **Eyeball vertex range verification** — `EYEBALL_VERT_LO = 3931` / `EYEBALL_VERT_HI = 5022` is derived from FLAME 2020 generic topology documentation; it must be verified against the actual `ft` rows in `assets/flame_uv_template.npz` before the skip logic is trusted. | First integrator | Confirm by inspecting committed `.npz`: `ft` rows whose three geometry indices all fall in `[3931, 5022]` map to eyeball faces; spot-check UV position of those rows. |

---

## 1. Scope

- One ComfyUI node: `FLAMEProjectToUV`.
- Input contract: a FLAME-topology `io.Mesh` (from `KaoLRMReconstruct`) + the
  original face `io.Image`.
- Output contract: `flaw_uv_image` (`io.Image` `[1, 512, 512, 3]` float32 in
  `[0,1]`) + `visibility` (`io.Mask` `[1, 512, 512]` float32 in `[0,1]`), both
  CPU.
- Fills the gap: FreeUV ships no script to produce `flaw_uv_image` from an
  arbitrary face image.
- Explicitly out of scope: MoSAR, intrinsic maps, albedo cleanup, texture
  compositing, FLAME optimizer hooks, face detection / auto-crop.

---

## 2. Node Boundaries and Wire Types

```python
class FLAMEProjectToUV(io.ComfyNode):
    node_id      = "FLAMEProjectToUV"
    display_name = "FLAME Project To UV"
    category     = "KaoLRM"
```

**Inputs**

| Name | Type | Default | Tooltip |
|---|---|---|---|
| `mesh` | `io.Mesh` | — | FLAME-topology mesh from `KaoLRMReconstruct`. If `num_sampling != 5023` was used, `base_faces` must be present. |
| `image` | `io.Image` | — | Original face image fed to `KaoLRMReconstruct`. Batch must be 1. Auto-resized to 224x224 before projection. |
| `source_cam_dist` | `io.Float` | `2.0` | Must match the value used in `KaoLRMReconstruct`. Range 1.0-4.0, step 0.1. |
| `uv_resolution` | `io.Int` | `512` | Output UV grid side length. Must be 512 for FreeUV compatibility; other values allowed for inspection. Range 64-2048. |

**Outputs**

| Name | Type |
|---|---|
| `flaw_uv_image` | `io.Image.Output` — `[1, 512, 512, 3]` float32 `[0,1]` |
| `visibility` | `io.Mask.Output` — `[1, 512, 512]` float32 `[0,1]` |

Output display name `flaw_uv_image` mirrors `FreeUVGenerate`'s input label; no
adapter shim required to connect the two nodes.

---

## 3. Camera Contract

**Resolution (Q1):** Duplicate the small matrix construction internally rather
than factor `_build_source_camera` out of `nodes/kaolrm_reconstruct.py` for
v0.1. The rejected alternative — importing `_build_source_camera` directly from
`kaolrm_reconstruct.py` — creates an import-time coupling that fires even when
the KaoLRM model is not loaded. Factoring into a shared helper is correct
long-term; deferred to v0.2 (non-breaking refactor, covered by
`test_camera_matches_kaolrm_reconstruct`).

The camera re-derived in `FLAMEProjectToUV` is numerically identical to
`nodes/kaolrm_reconstruct.py::_build_source_camera`. The snippet below mirrors
lines 58-65 of that file exactly, including the `device=` keyword on both
calls:

```python
# Extrinsics [1, 3, 4] — matches _default_source_camera(dist_to_center)
extrinsics = torch.tensor(
    [[[1, 0, 0, 0], [0, 0, -1, -source_cam_dist], [0, 1, 0, 0]]],
    dtype=torch.float32,
    device=device,           # must match model device; do NOT omit
)
# Intrinsics: hardcoded f=0.75, c=0.5. NOT exposed as node inputs.
intrinsics = runtime["create_intrinsics"](f=0.75, c=0.5, device=device).unsqueeze(0)
camera = runtime["build_camera_principle"](extrinsics, intrinsics)
```

`runtime` is `import_kaolrm_symbols()` from `nodes/kaolrm_runtime.py`.
`source_cam_dist` is the one user-facing parameter; `f=0.75` and `c=0.5` are
hardcoded (exposing them is drift-bait with no practical benefit for v0.1).

Omitting `device=` on either call puts tensors on CPU regardless of the model
device, which breaks the numerical equivalence test on CUDA and silently
misaligns the projection at inference time.

Input assumption: 224x224 image, face fills the frame. Off-framing degrades
projection quality but is not an error.

---

## 4. FLAME UV Data Source

**Confirmed from source:** `nodes/flame_core.py::FlameCore` loads only
`REQUIRED_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor",
"weights", "kintree_table", "f")`. `vt` (per-vertex UV coords) and `ft`
(per-face UV indices) are **not** present in `FlameCore`, and the FLAME 2020
`generic_model.pkl` does not carry them.

**Resolution (Q2):** Ship a fixed `assets/flame_uv_template.npz` in-repo
(Option B). The FLAME 2020 UV layout (`head_template.obj`-derived `vt`/`ft`) is
a single canonical artifact distributed with the FLAME 2020 release. Option A
(require user to drop `head_template.obj` at `models/flame/`) was rejected
because it adds a user-facing asset with no clear error message precedent.
Option C (reuse from `third_party/kaolrm/`) was rejected because the KaoLRM
vendor tree is not guaranteed to be present at node execution time.

<!-- BLOCKER B1: Confirm MPI license terms on the FLAME 2020 UV layout
(`head_template.obj`) before committing `assets/flame_uv_template.npz`. Until
confirmed, treat as MPI non-commercial. See Outstanding Blockers table above. -->

**Resolution (Q3):** Load UV data in a new `nodes/flame_uv_template.py` helper.
`FlameCore` stays geometry-only; UV is a rendering concern. The rejected
alternative — a `FlameCore.uv_coords()` method with a `uv_template_path` kwarg
— would couple the pkl loader to UV data that the FLAME pkl itself does not
carry, making the API misleading.

New file: `nodes/flame_uv_template.py`

```python
UV_TEMPLATE_PATH: Path   # REPO_ROOT / "assets" / "flame_uv_template.npz"
_UV_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

def load_uv_template(
    path: str | Path = UV_TEMPLATE_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (vt [V_uv, 2] float32, ft [F, 3] int64).

    V_uv > 5023: seam vertices appear twice in vt with distinct UV positions.
    F = 9976, same row count as the geometry faces tensor.
    ft[i] indexes into vt; paired geometry triangle is faces[i].
    Missing file raises RuntimeError naming path + https://flame.is.tue.mpg.de/
    """
```

---

## 5. Algorithm — Inverse-UV Rasterization

All UV-space barycentrics, world-space interpolation, and screen projection run
in float32; only the final `grid_sample` on the input IMAGE inherits the
upstream tensor dtype.

**Resolution (Q4):** Bilinear image sampling. Nearest-neighbor introduces block
artifacts at the 224-to-512 upscale; bilinear is the quality floor.

The rasterizer fills a `[R, R]` UV grid (`R = uv_resolution`) by iterating
over FLAME UV-space triangles:

```
vt, ft = load_uv_template()      # vt [V_uv, 2], ft [F, 3]
faces  = mesh.faces[0]           # [F, 3] geometry indices into vertices
verts  = mesh.vertices[0]        # [V, 3]

uv_out = zeros(R, R, 3)
mask   = zeros(R, R)

for tri_idx in range(F):         # F = 9976

    # Skip eyeball triangles (§9)
    ia, ib, ic = faces[tri_idx]
    if _is_eyeball_tri(ia, ib, ic): continue

    uv_a = vt[ft[tri_idx, 0]]   # [2] in [0,1]
    uv_b = vt[ft[tri_idx, 1]]
    uv_c = vt[ft[tri_idx, 2]]

    # 1. Texel bbox prune in UV space
    u_lo, u_hi = pixel_range(uv_a[0], uv_b[0], uv_c[0], R)
    v_lo, v_hi = pixel_range(uv_a[1], uv_b[1], uv_c[1], R)

    # 2. Barycentric coords of each texel in UV triangle
    #    UV-space triangle area (barycentric denominator) < 1e-9 -> skip
    w_a, w_b, w_c = bary2d(sub_grid_uv, uv_a, uv_b, uv_c)
    inside = (w_a >= 0) & (w_b >= 0) & (w_c >= 0)

    # 3. Back-face cull (§6)
    front = face_is_front(verts[ia], verts[ib], verts[ic], cam_origin)
    inside = inside & front

    # 4. Interpolate 3D position of geometry triangle
    world = (w_a[...,None] * verts[ia]
           + w_b[...,None] * verts[ib]
           + w_c[...,None] * verts[ic])          # [N, 3]

    # 5. Project to screen coords via extrinsic + intrinsic
    #    screen = intrinsic @ (extrinsic @ [world; 1])
    #    then perspective-divide to NDC [-1, 1]
    screen_xy = project_to_ndc(world, extrinsic, intrinsic)   # [N, 2]

    # 6. Bilinear sample input IMAGE (224x224, BCHW float32)
    rgb = grid_sample_bilinear(image_224, screen_xy)   # [N, 3]

    # 7. Write to UV buffer; mask = 1.0 on pass
    uv_out[v_px[inside], u_px[inside]] = rgb[inside]
    mask[v_px[inside], u_px[inside]]   = 1.0
```

**Projection detail.** `FLAMEProjectToUV` does **not** round-trip through
`build_camera_principle` for the per-vertex projection step. Instead, keep the
3×4 extrinsic and 3×3 intrinsic separately and compute:

```
# extrinsic: [3, 4], intrinsic: [3, 3], world: [N, 3]
world_h   = torch.cat([world, torch.ones(N, 1)], dim=1)  # [N, 4]
cam_space = (extrinsic @ world_h.T).T                    # [N, 3]
pixel_h   = (intrinsic @ cam_space.T).T                  # [N, 3]
ndc_xy    = pixel_h[:, :2] / pixel_h[:, 2:3]             # [N, 2], NDC [-1, 1]
```

`build_camera_principle` is only used in `test_camera_matches_kaolrm_reconstruct`
to verify that the re-derived matrices are numerically identical to what
`KaoLRMReconstruct` builds — it is not used in the hot per-vertex loop.

This mirrors the per-triangle bbox-prune + barycentric loop pattern from
`nodes/flame_render_util.py::_soft_torch_render`, with inverted sampling
direction: `_soft_torch_render` writes to screen pixels from 3D geometry; this
node writes to UV texels by sampling from screen pixels. No direct code reuse
is possible because the direction inverts.

v0.1 runs in Python over 9976 triangles. Wall time is comparable to
`MeshPreview`'s soft_torch render. Vectorized scatter is a v0.2 optimization.

---

## 6. Occlusion Strategy

**Resolution (Q5):** `dot` threshold `0.0` (strict back-face cull). A threshold
of `0.1` would let near-grazing faces write washed-out pixels into the flaw UV.
Strict zero is correct for the canonical front-portrait framing; the rejected
`0.1` tolerance is a v0.2 knob if fringe artifacts are reported.

For each UV triangle, compute the paired geometry face normal vs. view direction:

```python
n    = cross(verts[ib] - verts[ia], verts[ic] - verts[ia])   # [3] unnormalized
ctr  = (verts[ia] + verts[ib] + verts[ic]) / 3.0
view = cam_origin - ctr
front = dot(view, n) > 0.0
```

Camera origin in world space: inverting the KaoLRM extrinsic
`[[1,0,0,0],[0,0,-1,-d],[0,1,0,0]]` (a pure rotation + translation with no
scale) gives `cam_origin = (0, 0, -d)`. Derivation: the extrinsic maps world
point `p` to camera space as `R @ p + t` where `R = [[1,0,0],[0,0,-1],[0,1,0]]`
and `t = [0, -d, 0]^T`; the camera origin in world space is `R^T @ (-t) =
[[1,0,0],[0,0,1],[0,-1,0]] @ [0, d, 0]^T = (0, 0, -d)`. Precomputed once
outside the triangle loop as `cam_origin = torch.tensor([0.0, 0.0, -source_cam_dist])`.

**v0.2 candidate:** depth-buffer path behind `renderer` combo input
(`back_face_cull` | `depth_buffer`), mirroring `MeshPreview`. Required only if
nose/chin silhouette artifacts are reported.

---

## 7. Rasterization Backend

Pure-torch default. No new required dependencies.

- `renderer = "soft_torch"` (default): pure-torch per-triangle loop; works on
  CPU and CUDA.
- `renderer = "pytorch3d"` (optional): behind
  `nodes/_optional_deps.py::try_import_pytorch3d()`, gated by
  `io.Combo.Input("renderer", options=["soft_torch", "pytorch3d"], default="soft_torch")`.
  Not required for v0.1; UV-space rasterization via pytorch3d requires
  non-trivial camera setup and is deferred to v0.2.

---

## 8. Edge Cases

| Condition | Behavior | Enforced in |
|---|---|---|
| Point-cloud MESH (`faces` empty, `topology == "point_cloud"`) | Read `mesh.base_vertices` and `mesh.base_faces` instead. | `execute()` entry |
| `base_faces` missing on point-cloud MESH | `RuntimeError("FLAMEProjectToUV: mesh.base_faces not found. Set num_sampling=5023 in KaoLRMReconstruct, or ensure base_faces is present.")` | `execute()` entry |
| Batch > 1 (`image.shape[0] != 1`) | `RuntimeError` naming actual batch size. | `execute()` entry |
| Input IMAGE != 224x224 | Resize bicubic to 224x224; log warning. | `_prepare_image()` |
| `uv_resolution != 512` | Log warning that FreeUV expects exactly 512. Output follows `uv_resolution`. | `execute()` |
| Degenerate UV triangle | UV-space triangle area (barycentric denominator) < 1e-9 → skip | Inner loop |

**Resolution (Q6):** No `FLAME_PARAMS` input wire. `FLAMEParamsEdit ->
FLAMEParamsToMesh -> FLAMEProjectToUV` is the established chain for edited
params. Adding an internal re-solve path duplicates `FLAMEParamsToMesh` and
muddies the schema. Rejected.

---

## 9. Eyeball Handling

**Resolution (Q7, part 1 — vertex range):** FLAME geometry vertices `3931-5022`
are the eyeball region in the FLAME 2020 generic topology. This range must be
verified against the `ft` rows in `assets/flame_uv_template.npz` before
shipping (see Outstanding Blockers table and inline comment below).

**Resolution (Q7, part 2 — UV location):** In the canonical FLAME UV template
derived from `head_template.obj`, eyeball UV faces cluster near `(1.0, 1.0)`
(bottom-right corner in standard UV convention). Skipping eyeball triangles
entirely is more robust than zeroing the corner texel region, because the
corner location is template-specific.

```python
EYEBALL_VERT_LO = 3931   # inclusive
EYEBALL_VERT_HI = 5022   # inclusive

def _is_eyeball_tri(ia: int, ib: int, ic: int) -> bool:
    """True if all three geometry vertex indices are in the eyeball range."""
    return (EYEBALL_VERT_LO <= ia <= EYEBALL_VERT_HI and
            EYEBALL_VERT_LO <= ib <= EYEBALL_VERT_HI and
            EYEBALL_VERT_LO <= ic <= EYEBALL_VERT_HI)
```

Eyeball triangles: skip `uv_out` write and leave `mask = 0.0`.

<!-- BLOCKER B2: Verify eyeball vertex range 3931-5022 against
`assets/flame_uv_template.npz` ft rows once the template is committed. Confirm
whether eyeball ft rows map to the (1,1) UV corner in the chosen template or
elsewhere. See Outstanding Blockers table above. Owner: first integrator. -->

---

## 10. Outputs

- `flaw_uv_image`: `torch.Tensor [1, 512, 512, 3]` float32, `[0, 1]`, CPU,
  contiguous. Slots directly into `FreeUVGenerate.execute(flaw_uv_image=...)`.
  `FreeUVGenerate` validates `[B, H, W, 3]` and resizes to 512x512 internally;
  at the canonical `uv_resolution=512` default no resize occurs.
- `visibility`: `torch.Tensor [1, 512, 512]` float32, `[0, 1]`, CPU,
  contiguous. `1.0` = visible front-facing FLAME region with valid projection;
  `0.0` = back-facing, eyeball triangle, off-screen, or unsampled texel.
- Background texels (outside all `ft` triangles): RGB `0.0`, visibility `0.0`.
- Return: `io.NodeOutput(flaw_uv_image, visibility)`.

---

## 11. Testing — `tests/test_flame_project_to_uv.py`

All tests run without real FLAME weights. Uses `synthetic_flame_pkl` from
`tests/conftest.py` plus a new `synthetic_uv_template` fixture that builds
hand-crafted `vt [V_uv, 2]` + `ft [F, 3]` tensors for a minimal FLAME-like
geometry.

| Test | Fixtures | Asserts |
|---|---|---|
| `test_synthetic_mesh_projects_expected_color` | 2-triangle MESH, solid-color IMAGE 224x224 | Texels inside UV triangle contain image color; texels outside are `0.0`. |
| `test_back_face_marks_invisible` | MESH rotated 180 deg around Y | `visibility.sum() == 0`. |
| `test_eyeball_region_masked` | `synthetic_flame_pkl`; `ft` rows with all-eyeball `faces[tri_idx]` indices | Those texels have `mask == 0.0`. |
| `test_batch_gt_1_rejected` | `image [2, H, W, 3]` | `RuntimeError`. |
| `test_point_cloud_mesh_uses_base_attrs` | MESH with empty `faces`, populated `base_vertices`/`base_faces` | Executes; UV has non-zero texels. |
| `test_point_cloud_mesh_without_base_raises` | MESH with empty `faces`, no `base_faces` attr | `RuntimeError` naming `base_faces`. |
| `test_camera_matches_kaolrm_reconstruct` | KaoLRM runtime mocked | `torch.allclose(camera_node, camera_reconstruct, atol=1e-6)` at `source_cam_dist=2.0`. |
| `test_image_resized_before_projection` | 448x448 IMAGE | Warning emitted; output shape `[1, 512, 512, 3]`. |
| `test_output_dtype_and_device` | Synthetic inputs | `flaw_uv_image.dtype == torch.float32`, `device.type == "cpu"`, same for `visibility`. |

Slow end-to-end (real FLAME pkl + real portrait): `@pytest.mark.slow`, run
locally before release.

---

## 12. Integration

The downstream demo workflow chains `KaoLRMReconstruct -> FLAMEProjectToUV ->
FreeUVGenerate`. `KaoLRMReconstruct` emits `io.Mesh` on output 0;
`FLAMEProjectToUV` consumes the `io.Mesh` and the original `io.Image` and emits
`flaw_uv_image` (`io.Image`) plus `visibility` (`io.Mask`); `FreeUVGenerate`
consumes `flaw_uv_image` directly. The `visibility` output is available for
downstream masking but is not consumed by `FreeUVGenerate`. The workflow JSON
(`workflows/kaolrm_freeuv_albedo.json`) is out of scope for this spec.

---

## 13. License and Weights

No new external weights. `FLAMEProjectToUV` consumes only the `io.Mesh`
produced by the gated KaoLRM path and `assets/flame_uv_template.npz` (MPI
license status: see Outstanding Blockers, B1). No new `i_understand_non_commercial` gate.
The upstream `LoadKaoLRM` gate already covers everything this node produces.

---

## 14. Brutal-Review Risks

1. **Camera convention drift.** If `KaoLRMReconstruct._build_source_camera`
   changes its extrinsic matrix or intrinsic params, this node silently produces
   misaligned projections with no loud failure. Mitigation: `test_camera_matches_kaolrm_reconstruct`
   is the regression gate; factor into shared helper in v0.2.

2. **UV seam bleeding.** `V_uv > 5023` means seam vertices appear twice in `vt`
   with adjacent UV positions. Per-triangle rasterization can leave hairline gaps
   along UV seams where neither adjacent triangle's bbox reaches. Mitigation:
   conservative bbox (extend by 1 texel); one-texel mask dilation is a v0.2
   post-process option.

3. **Back-face cull false-positives on extreme poses.** Back-face cull alone
   misses self-occlusion at oblique angles. Under KaoLRM's canonical framing
   this is rare. Mitigation: depth-buffer fallback behind `renderer` combo in
   v0.2; framing assumption documented in node tooltip.

4. **Aliasing from 224-to-512 upscale.** Bilinear sampling from 224^2 into
   512^2 UV inherits the source resolution cap. FreeUV's SD1.5 stack
   hallucinates detail on top; flaw-UV softness is acceptable input. Accepted
   for v0.1; no mitigation required.

5. **Eyeball leakage if vertex range is wrong.** If `3931-5022` does not match
   the eyeball range in the chosen UV template, eyeball texels will carry sclera
   pixels into FreeUV and corrupt the albedo result. Mitigation: Blocker B2
   must clear before release; `test_eyeball_region_masked` guards the skip logic.

6. **fp16 numerical instability in barycentric math.** Small UV-space triangles
   can underflow the barycentric denominator in fp16. Mitigation: all barycentric
   and projection math runs in float32 regardless of upstream mesh dtype.

7. **UV template provenance.** `assets/flame_uv_template.npz` carries unconfirmed
   MPI license terms. Shipping before confirmation is a legal risk. Mitigation:
   Blocker B1 must clear; `load_uv_template` raises `RuntimeError` naming
   the path and `https://flame.is.tue.mpg.de/` if the file is absent.

8. **Point-cloud MESH trap.** `num_sampling != 5023` on `KaoLRMReconstruct`
   produces an empty `faces` tensor. The silent fallback to `base_faces` is
   correct but must be visible in the node tooltip so users are not surprised by
   a working UV projection from a "point cloud" mesh.

---

## Open Questions (deferred)

- **Q1 — camera helper refactor -> v0.2.** Factor `_build_source_camera` out of
  `nodes/kaolrm_reconstruct.py` into a shared module. Non-breaking;
  `test_camera_matches_kaolrm_reconstruct` is the regression gate.

- **Vectorized UV rasterizer -> v0.2.** Replace the per-triangle Python loop with
  a batched scatter: precompute a UV-grid-to-triangle lookup table at first
  execute (cached), then run a single `torch.nn.functional.grid_sample` call
  over all texels simultaneously.

- **Depth-buffer occlusion -> v0.2.** Add `renderer` combo input
  (`back_face_cull` | `depth_buffer`) mirroring `MeshPreview`. Required only if
  oblique-angle artifact reports come in post-release.

- **pytorch3d UV rasterization -> v0.2.** Non-trivial UV-space camera setup;
  deferred until the pure-torch path is validated in production.
