# FLAMEProjectToUV — Skeleton

Single node. Projects the KaoLRM input IMAGE into FLAME UV space using the same camera that KaoLRM used, producing the `flaw_uv_image` that `FreeUVGenerate` consumes. No other layers in scope.

Open questions are numbered Q1–Qn globally and appear inline where they bite.

---

## 1. Scope

- One ComfyUI node: `FLAMEProjectToUV`.
- Input contract: a FLAME-topology `MESH` (from `KaoLRMReconstruct`) + the original face `IMAGE`.
- Output contract: `flaw_uv_image` (`IMAGE [1, 512, 512, 3]` float32 in `[0,1]`) + `visibility` (`MASK [1, 512, 512]` float32 in `[0,1]`), both CPU.
- Fills the gap: FreeUV ships no script to produce `flaw_uv_image` from an arbitrary face.
- Explicitly out of scope: MoSAR, intrinsic maps, albedo cleanup, texture compositing, FLAME optimizer hooks, face detection / auto-crop.

## 2. Node Boundaries and Wire Types

- V3 `io.ComfyNode` + `io.Schema`. `category="KaoLRM"` for v0.1 (matches existing suites).
- `node_id="FLAMEProjectToUV"`, `display_name="FLAME Project To UV"`.
- Inputs: `mesh` (`io.Mesh`), `image` (`io.Image`), `source_cam_dist` float (default `2.0`), `uv_resolution` int (default `512`, must match FreeUV input).
- Outputs: `io.Image.Output(display_name="flaw_uv_image")`, `io.Mask.Output(display_name="visibility")`.
- Naming rationale: output display name `flaw_uv_image` mirrors `FreeUVGenerate`'s input label.

## 3. Camera Contract

- Must match `KaoLRMReconstruct._build_source_camera` numerically.
- **Recommendation:** re-derive internally. Use `runtime["create_intrinsics"](f=0.75, c=0.5)` + the canonical extrinsics `[[1,0,0,0],[0,0,-1,-d],[0,1,0,0]]`, composed via `build_camera_principle`. `f=0.75`, `c=0.5` are NOT exposed as node inputs — drift-bait.
- `source_cam_dist` IS exposed (default `2.0`) because the user may override it on `KaoLRMReconstruct`; the two values must match.
- Q1: factor `_build_source_camera` out of `kaolrm_reconstruct.py` into a shared helper, or duplicate the small matrix construction here? Duplication is lower-coupling but risks drift; recommendation is to factor out once this node lands.

## 4. FLAME UV Data Source

- Inverse-UV rasterization requires per-vertex UVs `vt` and per-face UV indices `ft`.
- Reality check: `nodes/flame_core.py:REQUIRED_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")` — `vt`/`ft` are **not** in the FLAME pkl. FLAME 2020's `generic_model.pkl` traditionally does not carry UV layout; UVs ship separately (e.g., `head_template.obj` or `FLAME_texture.npz` from the MPI release).
- Q2 (**highest priority, load-bearing**): where do `vt`/`ft` come from?
  - Option A — require user to drop a UV template alongside the FLAME pkl (e.g., `models/flame/head_template.obj`), load at first use; missing-asset error names path + upstream URL.
  - Option B — ship a fixed `assets/flame_uv_template.npz` in-repo since the FLAME 2020 UV layout is a single canonical artifact. Verify MPI licensing on the UV layout specifically.
  - Option C — reuse a UV template already present in `third_party/kaolrm/` (inspect before deciding).
- Q3: keep UV loading in a new `nodes/flame_uv_template.py` helper (recommended — `FlameCore` stays geometry-only), or extend `FlameCore` with a `uv_template_path` kwarg + `vt`/`ft` buffers?

## 5. Algorithm — Inverse-UV Rasterization

- For each texel `(u, v)` in the `[512, 512]` UV grid: find the UV-space triangle in `ft` it lies inside, compute UV-space barycentrics, interpolate the corresponding 3D vertex positions from `vertices[faces[tri]]`, project to screen via the KaoLRM camera, sample the input IMAGE bilinearly, write to the UV texel.
- Terse pseudocode:
  ```
  for each ft triangle T (with paired f triangle F):
      rasterize T in UV space → texels + uv_barycentrics
      world_xyz = bary · vertices[faces[F]]                 # [N, 3]
      screen_xy = project(world_xyz, kaolrm_camera)         # [N, 2] in [-1,1]
      rgb = bilinear_sample(image, screen_xy)               # [N, 3]
      uv_out[texels] = rgb
      visibility[texels] = occlusion_test(...)              # see §6
  ```
- v0.1 runs per-triangle in a Python loop (mirrors `_soft_torch_render` in `flame_render_util.py`). Vectorized scatter is a v0.2 optimization.
- Q4: bilinear vs nearest for image sampling? Recommendation: bilinear.

## 6. Occlusion Strategy

- **Recommendation (v0.1): back-face cull.** Compute the face normal in camera space; mark visibility `0` where `dot(view_dir, normal) < threshold`. O(F), no z-buffer required.
- Rationale: cheap, correct for the dominant case (front-facing portrait at KaoLRM framing).
- Fallback (v0.2 if artifacts appear at nose/chin silhouette): pure-torch z-buffer — rasterize all triangles into a screen-space depth buffer, then compare projected z during UV rasterization.
- Full depth-buffer occlusion explicitly skipped for v0.1 (expensive, not needed).
- Q5: pick `dot` threshold — `0.0` (strict) vs `0.1` (grazing-angle tolerance). Resolve empirically during smoke testing.

## 7. Rasterization Backend

- **Recommendation:** pure-torch default, mirroring `nodes/flame_render_util.py:_soft_torch_render` style (per-triangle bbox prune + barycentric interpolation, all torch ops).
- Optional pytorch3d / nvdiffrast path behind `nodes/_optional_deps.py:try_import_pytorch3d()`. **Not required for v0.1.**
- No new required deps (honors "avoid adding dependencies" rule).

## 8. Edge Cases

- Point-cloud MESH (`num_sampling != 5023`): read `mesh.base_vertices` and `mesh.base_faces` attrs instead of `mesh.vertices`/`mesh.faces`. Missing attr → `RuntimeError` naming the missing attr.
- Batch > 1: reject with `RuntimeError` in v0.1 (matches `KaoLRMReconstruct`, `SMIRKPredict`, `FreeUVGenerate`).
- Input IMAGE resolution ≠ 224: resize to 224×224 bicubic before projection to match the camera's assumed input plane. Document that projection quality is bounded by the 224² baseline.
- Q6: should the node also accept a `FLAME_PARAMS` wire and re-solve internally? Recommendation: no — `FLAMEParamsEdit → FLAMEParamsToMesh → FLAMEProjectToUV` is the established chain, and dual inputs muddy the schema.

## 9. Eyeball Handling

- FLAME vertices `3931–5022` are the eyeballs. In many FLAME UV exporters their UVs collapse to `(1, 1)` and bleed into corner texels, polluting FreeUV input.
- Mitigation: mark corner-texel region `0` in visibility MASK; additionally, exclude any `ft` triangle whose three UV-vertex indices correspond to eyeball vertices from rasterization.
- Q7: eyeball vertex range depends on the specific FLAME template — verify against the UV template chosen in Q2 before shipping. Also: does the chosen template actually put eyeballs at `(1,1)` or somewhere else?

## 10. Outputs

- `flaw_uv_image`: `torch.Tensor [1, 512, 512, 3]` float32, `[0, 1]`, CPU, contiguous.
- `visibility`: `torch.Tensor [1, 512, 512]` float32, `[0, 1]`, CPU, contiguous. `1` = visible face region with valid projection; `0` = back-facing, eyeballs, off-screen, or unsampled texel.
- Background texels (outside any `ft` triangle): RGB `0`, visibility `0`.

## 11. Testing — `tests/test_flame_project_to_uv.py`

- `test_synthetic_mesh_projects_expected_color` — synthetic 2-triangle MESH with known UVs + solid-color IMAGE → expected color inside triangle, zero outside.
- `test_back_face_marks_invisible` — rotate mesh 180° around Y so all triangles back-face → `visibility.sum() == 0`.
- `test_eyeball_region_masked` — FLAME-topology MESH (via `synthetic_flame_pkl` fixture) → visibility is `0` at texels corresponding to eyeball `ft` triangles.
- `test_batch_gt_1_rejected` — input `[2, H, W, 3]` IMAGE → `RuntimeError`.
- `test_point_cloud_mesh_uses_base_attrs` — point-cloud MESH (empty `faces`, populated `base_vertices`/`base_faces`) → node reads base attrs, produces valid UV.
- `test_point_cloud_mesh_without_base_raises` — point-cloud MESH missing `base_faces` → `RuntimeError` naming the missing attr.
- `test_camera_matches_kaolrm_reconstruct` — construct the same camera via both paths, assert tensor equality.
- All tests run without real FLAME weights (use `synthetic_flame_pkl` + a hand-crafted UV template fixture).

## 12. Integration

- Downstream follow-up (out of scope for this plan): bundle `workflows/kaolrm_freeuv_albedo.json` wiring `KaoLRMReconstruct → FLAMEProjectToUV → FreeUVGenerate → MeshPreview`.

## 13. License and Weights

- No new external weights.
- Consumes the FLAME pkl already required by KaoLRM / SMIRK branches.
- If Q2 resolves to shipping a UV template as a project asset, verify MPI licensing on the FLAME UV layout before committing to `assets/` — may still be non-commercial.
- No new `i_understand_non_commercial` gate — upstream inputs (MESH, IMAGE) already passed through gated nodes.

## 14. Brutal-review Style Risks

1. **Camera drift.** If `KaoLRMReconstruct`'s camera changes, this node must change in lockstep. Mitigation: shared camera helper (Q1) + camera-equivalence test (§11).
2. **UV seam bleeding.** Per-triangle rasterization can leave hairline gaps or one-texel bleed across UV seams. Mitigation: conservative bbox + inside-triangle test; v0.2 may dilate visibility mask by 1 texel.
3. **Occlusion false-negatives.** Back-face cull alone misses self-occlusion (nose shadowing cheek at oblique angles). Rare under KaoLRM's canonical framing. Mitigation: z-buffer fallback (§6).
4. **Aliasing at 512².** Bilinear up-sample from 224² input into 512² UV introduces softness. Accepted for v0.1 — FreeUV's SD1.5 stack hallucinates detail downstream.
5. **Eyeball leakage.** Q7. If the UV template's eyeball layout differs from the assumed `(1,1)` convention, corner texels will contaminate FreeUV input.
6. **fp16 underflow.** Mirrors Brutal Review #11. Cast projection + barycentric math to float32; IMAGE sampling may stay in fp16.
7. **UV template provenance.** Q2. If the template ships separately from FLAME pkl, users will forget it. Missing-asset error must name exact path + upstream URL, matching the project's support contract.
8. **Point-cloud MESH trap.** A user who sets `num_sampling != 5023` on `KaoLRMReconstruct` gets an empty `faces` tensor. Silent fallback to `base_faces` is the right default but must be documented in the node tooltip.
