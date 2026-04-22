# SMIRK Node Suite — Layer 0 Expression Refinement on KaoLRM

## 1. Node Boundaries and Wire Types

### New wire type: `FLAME_PARAMS`

**Resolution (Q6):** Introduce `io.Custom("FLAME_PARAMS")` as a first-class wire, not a piggy-back on `MESH.flame_params`. The merge node needs to consume params without mesh geometry, and DECA (M3) will also consume this wire. Two distinct nodes emitting the same custom type is cleaner than scraping attrs off a mesh payload.

Canonical schema on the wire — a plain Python dict, all tensors at `[B, N]` with `B=1` in v0.1:

```python
{
    "shape":       torch.Tensor[B, 100],   # float32
    "expression":  torch.Tensor[B, 50],    # float32
    "pose":        torch.Tensor[B, 6],     # float32, axis-angle [global(3)|jaw(3)]
    "scale":       torch.Tensor[B, 1],     # float32
    "translation": torch.Tensor[B, 3],     # float32
    "fix_z_trans": bool,                   # True for mono-origin KaoLRM params
}
```

**Important:** `KaoLRMReconstruct._params_to_cpu` strips the batch dimension with `[0]`, so `mesh.flame_params` tensors are currently flat `[100]`, `[50]`, `[6]`, `[1]`, `[3]`. A `KaoLRMParamsToFLAMEParams` extraction shim is responsible for unsqueezing these flat tensors back to `[1, N]` to conform to the `[B, N]` convention on the `FLAME_PARAMS` wire. `FLAMEParamsMerge` and `FLAMEParamsToMesh` consume `[B, N]` tensors only — they must never receive flat `[N]` tensors directly.

`KaoLRMParamsToFLAMEParams` (trivial shim, one input `MESH`, one output `FLAME_PARAMS`) reads `mesh.flame_params`, calls `.unsqueeze(0)` on each value tensor, and copies `fix_z_trans` from `mesh.flame_params.get("fix_z_trans", kaolrm_model.get("variant") == "mono")`. This is the compatibility layer — no other node unsqueezes.

### `SMIRK_MODEL` descriptor

Mirrors the `KAOLRM_MODEL` dict pattern from `nodes/kaolrm_load.py` (mirrors `KAOLRM_MODEL` minus `variant`/`config_path`, neither applies to SMIRK's single-checkpoint release):

```python
{
    "ckpt_path":   str,       # absolute path to models/smirk/SMIRK_em1.pt
    "device":      str,       # "cuda" | "cpu"
    "dtype":       str,       # "fp32" | "fp16" | "bf16"
    "smirk_root":  str | None # resolved by smirk_runtime.py
}
```

No `variant` key; there is one public checkpoint. Add `variant` if a second checkpoint is released.

---

## 2. Runtime Resolution — `nodes/smirk_runtime.py`

Mirrors `nodes/kaolrm_runtime.py` exactly: env-var first, then `third_party/smirk/`, then installed package.

```python
SMIRK_ENV_VAR = "SMIRK_ROOT"
SMIRK_CANDIDATES = [REPO_ROOT / "third_party" / "smirk"]

def _is_smirk_root(root: Path) -> bool:
    return (root / "src" / "smirk_encoder.py").is_file()

def resolve_smirk_root(*, required: bool = True) -> Path | None: ...
def ensure_smirk_on_path(*, required: bool = True) -> Path | None: ...
def import_smirk_encoder() -> type: ...  # returns SmirkEncoder class
```

**Resolution (Q7):** SMIRK's `demo.py` imports `mediapipe`, `face_alignment`, `scikit-image`, and `pytorch3d` — but only the demo uses them. `SmirkEncoder` itself (`src/smirk_encoder.py`) only needs `torch` and `timm`. Import isolation via `importlib.util.spec_from_file_location` for `src/smirk_encoder.py` sidesteps the heavier demo imports.

**`timm` is a real, explicit dependency of SMIRK.** `src/smirk_encoder.py` imports `timm` at the top level, and `requirements.txt` pins `timm==0.9.16` (source: `https://raw.githubusercontent.com/georgeretsi/smirk/main/requirements.txt`). `transformers` does **not** depend on `timm` — do not rely on a transitive install. Add `timm` explicitly to `requirements.txt` and document that SMIRK requires it.

---

## 3. Weight and Config Discovery — `nodes/smirk_load.py`

```python
SMIRK_SUBDIR = "smirk"
SMIRK_CKPT_FILENAME = "SMIRK_em1.pt"
# filename confirmed from demo usage in https://github.com/georgeretsi/smirk (demo.py --pretrained_model_path pretrained_models/SMIRK_em1.pt)
SMIRK_CKPT_URL = "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view"
```

<!-- BLOCKER: SHA256 hash for SMIRK_em1.pt not yet confirmed — requires downloading the file. Pin hash in docs/model_hashes.md before v0.1 release. Owner: first person to run the real weights. -->

`ensure_smirk_weights()` follows the exact pattern of `ensure_kaolrm_weights()`:

```python
def ensure_smirk_weights() -> Path:
    path = Path(folder_paths.models_dir) / SMIRK_SUBDIR / SMIRK_CKPT_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"Missing SMIRK checkpoint '{path.name}'. Place it at '{path}'. "
            f"Download from {SMIRK_CKPT_URL}"
        )
    return path
```

No config JSON needed — `SmirkEncoder` is constructed from hyperparameter defaults (`n_shape=300`, `n_exp=50`), not a config file. FLAME pkl reuses the same `ensure_generic_flame_pkl()` from `nodes/kaolrm_load.py`.

---

## 4. `LoadSMIRK` Node — `nodes/smirk_load.py`

```python
class LoadSMIRK(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSMIRK",
            display_name="Load SMIRK",
            category="KaoLRM",
            description=(
                "Resolve SMIRK expression encoder assets and output a model descriptor. "
                "SMIRK is MIT licensed, but the FLAME model it uses is MPI non-commercial — "
                "all outputs remain research-only."
            ),
            inputs=[
                io.Combo.Input("device", options=["auto","cpu","cuda"], default="auto", ...),
                io.Combo.Input("dtype",  options=["auto","fp32","fp16","bf16"], default="auto", ...),
                io.Boolean.Input(
                    "i_understand_non_commercial", default=False,
                    tooltip=(
                        "SMIRK itself is MIT licensed, but it runs on FLAME (MPI non-commercial). "
                        "Everything produced by this pipeline remains research-only."
                    ),
                ),
            ],
            outputs=[SMIRK_MODEL.Output(display_name="smirk_model")],
        )
```

`execute()` raises `RuntimeError` if the gate is False, resolves device/dtype (same `resolve_device`/`resolve_dtype` helpers from `kaolrm_load.py`), calls `ensure_smirk_weights()`, and returns the descriptor dict. Heavy model build is deferred to `SMIRKPredict`.

---

## 5. `SMIRKPredict` Node — `nodes/smirk_predict.py`

### Cache key

```python
_SMIRK_CACHE: dict[tuple[str, str, str, str | None], SmirkEncoder] = {}
# key = (device, dtype, ckpt_path, smirk_root)
```

No `variant` dimension; extend to `(variant, ...)` if a second checkpoint is added.

### Schema inputs

| Name | Type | Default | Tooltip |
|---|---|---|---|
| `smirk_model` | `SMIRK_MODEL` | — | Descriptor from `LoadSMIRK`. |
| `image` | `io.Image` | — | Pre-cropped face at 224×224. Must be a tight face crop — SMIRK expects MediaPipe alignment (v0.2). Batch must be 1. |

Output: `FLAME_PARAMS` wire.

### Inference flow

**Resolution (Q2):** SMIRK's `demo.py` confirms 224×224 input after MediaPipe face crop (source: `https://github.com/georgeretsi/smirk/blob/main/demo.py`). For v0.1, the node does not perform face detection; the user must supply a pre-cropped image. `KaoLRMPreprocess` output at 224×224 is suitable if the face already fills the frame. Auto-crop via `face_alignment` is deferred to v0.2 and explicitly scoped out here.

Forward:

1. Resize IMAGE to `(1, 3, 224, 224)`, clamp to [0, 1]. No ImageNet normalization — `SmirkEncoder` does its own internal norm.
2. Load/cache `SmirkEncoder` via `_get_cached_smirk(smirk_model)` which calls `_load_smirk_encoder(ckpt_path, device, dtype)` on first use.
3. Cast to model dtype; run `encoder(image)` → output dict with keys and shapes confirmed from `src/smirk_encoder.py` (source: `https://raw.githubusercontent.com/georgeretsi/smirk/main/src/smirk_encoder.py`, georgeretsi/smirk at time of spec authorship):
   - `shape_params`: `[1, 300]`
   - `expression_params`: `[1, 50]`
   - `pose_params`: `[1, 3]` (global rotation, axis-angle)
   - `jaw_params`: `[1, 3]` (jaw rotation, axis-angle)
   - `eyelid_params`: `[1, 2]` (clamped to [0, 1])
   - `cam`: `[1, 3]` (weak-perspective `[s, tx, ty]`)
4. Project outputs to `FLAME_PARAMS` canonical schema (see merge policy below). Set `fix_z_trans=False` — SMIRK has no mono/multiview concept; merged params carry `fix_z_trans` from the KaoLRM input.
5. Return `io.NodeOutput(flame_params_dict)` — all tensors on CPU, float32, at `[1, N]`.

<!-- BLOCKER: Commit hash for georgeretsi/smirk not pinned. Pin the exact commit SHA in smirk_runtime.py and docs before v0.1 release. Owner: first integrator. -->

**Resolution (Q9):** `SMIRKPredict` enforces batch size 1 in v0.1, matching `KaoLRMReconstruct`. Batched prediction is v0.2.

---

## 6. `FLAMEParamsMerge` Node — `nodes/flame_params_merge.py`

### Merge policy table

**Resolution (Q3/Q4):**

| Field | Shape | Source | Notes |
|---|---|---|---|
| `shape` | `[B, 100]` | KaoLRM | SMIRK emits `shape_params[B, 300]`; slice `[:, :100]` if used, but roadmap says `shape ← KaoLRM` — SMIRK's shape is discarded. KaoLRM's identity geometry is the anchor. |
| `expression` | `[B, 50]` | SMIRK | Dims match exactly (`n_exp=50` in both). |
| `pose[:, :3]` | `[B, 3]` | KaoLRM | Global rotation (axis-angle). KaoLRM `pose[6]` = `[global(3)\|jaw(3)]`; `neck_pose` is a hardcoded zero injected inside `flame.py` (lines 441–442: `self.neck_pose.expand(batch_size, -1)` is concatenated between `pose_params[:, :3]` and `pose_params[:, 3:]`). |
| `pose[:, 3:]` | `[B, 3]` | SMIRK | Jaw rotation from SMIRK `jaw_params[B, 3]`. KaoLRM's original `pose[:, 3:]` is overwritten. |
| `scale` | `[B, 1]` | KaoLRM | SMIRK's `cam[3]` is weak-perspective `[s, tx, ty]`, incompatible with FLAME scale+translation convention — discarded. |
| `translation` | `[B, 3]` | KaoLRM | Same reason; SMIRK's cam is not a world translation. |
| `fix_z_trans` | `bool` | KaoLRM | Passed through from `kaolrm_params["fix_z_trans"]` — honors the mono/multiview origin at mesh re-solve time. |
| `eyelid_params` | dropped | — | KaoLRM's `flame_model(shape, expr, pose)` takes 3 positional args; no eyelid slot. Stored as pass-through attr on `FLAME_PARAMS` for M3 DECA integration; not fed to the FLAME forward in v0.1. |

The merged `pose` is:

```python
merged_pose = torch.cat([kaolrm_pose[:, :3], smirk_jaw[:, :3]], dim=1)  # [B, 6]
```

Both inputs are `[B, N]` — `FLAMEParamsMerge` assumes `[B, N]` throughout and fires `RuntimeError` if shapes are flat `[N]`. The `KaoLRMParamsToFLAMEParams` shim is the guaranteed entry point for KaoLRM-origin params; `SMIRKPredict` always emits `[B, N]` directly.

Dim-compatibility assertions fire as `RuntimeError` before any tensor op.

### Schema inputs

| Name | Type | Notes |
|---|---|---|
| `kaolrm_params` | `FLAME_PARAMS` | From `KaoLRMParamsToFLAMEParams` shim. Must be `[B, N]`. |
| `smirk_params` | `FLAME_PARAMS` | From `SMIRKPredict`. Must be `[B, N]`. |

Output: `FLAME_PARAMS` wire (merged dict, `[B, N]` throughout, `fix_z_trans` copied from KaoLRM input).

---

## 7. Mesh Re-solve — `nodes/flame_params_to_mesh.py`

**Resolution (Q5):** A dedicated `FLAMEParamsToMesh` node is the right choice. Overloading `KaoLRMReconstruct` would couple SMIRK into the triplane-encoder node and add optional-input complexity that fights the V3 schema model. `FLAMEParamsToMesh` belongs to the **FLAME shared stack**, not the SMIRK suite — it can be consumed by any node that emits `FLAME_PARAMS`.

The FLAME forward itself lives in KaoLRM's vendored `flame.py` (loaded via `_load_source_module` in `kaolrm_mesh_model.py`). `FLAMEParamsToMesh` must call `resolve_kaolrm_root()` to find that file. This is an honest layering dependency: `FLAMEParamsToMesh` depends on the KaoLRM vendor tree for the FLAME pkl path and the `flame.py` module. The SMIRK nodes themselves (`smirk_load.py`, `smirk_predict.py`, `flame_params_merge.py`) do **not** import KaoLRM — only `FLAMEParamsToMesh` does.

A `_FLAME_MODEL_CACHE: dict[tuple[str, str], FLAMEModel]` keyed by `(flame_pkl_path, device)` avoids reloading when the FLAME pkl and device haven't changed. `_install_chumpy_shims()` from `nodes/flame_core.py` is called once before `pickle.load`.

Schema inputs: `flame_params` (`FLAME_PARAMS`), `num_sampling` int (default `5023`). Output: `io.Mesh`.

`fix_z_trans` is read from `flame_params["fix_z_trans"]` and honored at mesh solve time. KaoLRM mono-origin params carry `fix_z_trans=True`; SMIRK-only or merged params carry whatever the KaoLRM input declared. This ensures a KaoLRM → `KaoLRMParamsToFLAMEParams` → `FLAMEParamsMerge` → `FLAMEParamsToMesh` round-trip produces the same pose as the original `KaoLRMReconstruct` output. The invariant: **`FLAME_PARAMS.fix_z_trans` is always set at extraction time and always honored at mesh re-solve time.**

---

## 8. File Layout and Registration

New files:

```
nodes/smirk_runtime.py          # resolver pattern, mirrors kaolrm_runtime.py
nodes/smirk_load.py             # LoadSMIRK node + weight/descriptor helpers
nodes/smirk_predict.py          # SMIRKPredict node + _SMIRK_CACHE
nodes/flame_params_merge.py     # FLAMEParamsMerge node
nodes/flame_params_to_mesh.py   # FLAMEParamsToMesh node + _FLAME_MODEL_CACHE
nodes/kaolrm_params_shim.py     # KaoLRMParamsToFLAMEParams extraction shim
```

`nodes/__init__.py` addition:

```python
from .smirk_load import LoadSMIRK
from .smirk_predict import SMIRKPredict
from .flame_params_merge import FLAMEParamsMerge
from .flame_params_to_mesh import FLAMEParamsToMesh
from .kaolrm_params_shim import KaoLRMParamsToFLAMEParams

NODE_CLASSES = [
    LoadKaoLRM, KaoLRMPreprocess, KaoLRMReconstruct, MeshPreview,              # existing
    KaoLRMParamsToFLAMEParams,                                                  # shim
    LoadSMIRK, SMIRKPredict, FLAMEParamsMerge, FLAMEParamsToMesh,              # new
]
```

**Resolution (category):** Keep `category="KaoLRM"` for v0.1 to avoid breaking existing user workflows. Migrate all categories to `WYSIWYG/Face/...` in the v0.2 multi-suite reorganization already noted in CLAUDE.md.

`assets/SMIRK_MIT_LICENSE.txt` — copy the MIT license text from `https://github.com/georgeretsi/smirk/blob/main/LICENSE`. No registration wall; the file is informational parity with `assets/FLAME_LICENSE.txt`.

`requirements.txt` addition: `timm>=0.9.16` — SMIRK's `src/smirk_encoder.py` imports `timm` directly; `transformers` does not provide it transitively.

---

## 9. Testing Strategy

Per-node tests mirror `tests/test_kaolrm_*.py`:

| Test file | What it covers |
|---|---|
| `tests/test_smirk_runtime.py` | `resolve_smirk_root`: env var override, candidate dir detection, installed-package fallback, `required=True` raises with message. |
| `tests/test_smirk_load.py` | `LoadSMIRK.execute()`: gate False → `RuntimeError`; missing weight file → `RuntimeError` naming exact path and URL; with a dummy `.pt` file → returns descriptor dict with expected keys. |
| `tests/test_smirk_predict.py` | `SMIRKPredict.execute()` with a monkeypatched `_load_smirk_encoder` returning a dummy that outputs valid `[1, N]` tensors for all keys. Assert `FLAME_PARAMS` output has `shape == (1, 100)`, `expression == (1, 50)`, `pose == (1, 6)` on CPU float32. Assert `fix_z_trans` key is present. |
| `tests/test_kaolrm_params_shim.py` | `KaoLRMParamsToFLAMEParams.execute()`: flat `[N]` input tensors → `[1, N]` output; `fix_z_trans` correctly derived from mesh.flame_params or kaolrm_model variant. |
| `tests/test_flame_params_merge.py` | Merge correctness: assert merged `shape` equals KaoLRM input; merged `expression` equals SMIRK input; merged `pose[:, :3]` equals KaoLRM `pose[:, :3]`; merged `pose[:, 3:]` equals SMIRK `jaw_params`; `fix_z_trans` copied from KaoLRM. Flat `[N]` inputs (not `[B, N]`) raise `RuntimeError`. |
| `tests/test_flame_params_to_mesh.py` | Round-trip with a mocked FLAME model: `FLAME_PARAMS` → `MESH` with `vertices.shape == (1, 5023, 3)`, `faces.shape == (1, 9976, 3)`, all finite. Test that `fix_z_trans=True` vs `False` produces different vertex positions. |

All tests pass with no real weights present. Slow end-to-end (real `SMIRK_em1.pt` + real FLAME pkl) marked `@pytest.mark.slow`.

---

## 10. Non-commercial Gate and Licensing

SMIRK itself is MIT (source: `https://github.com/georgeretsi/smirk`). However, it runs on FLAME 2020 (MPI non-commercial), making the whole pipeline research-only exactly as with KaoLRM. The `i_understand_non_commercial` gate on `LoadSMIRK` is justified by FLAME downstream — not by SMIRK's own license. The tooltip must make this distinction explicit to prevent the "it's MIT, I can use it commercially" misreading (Brutal Review #1 failure mode applied to SMIRK).

The workflow example `workflows/kaolrm_smirk_mesh.json` should show:

```
LoadKaoLRM ─► KaoLRMPreprocess ─► KaoLRMReconstruct ─► KaoLRMParamsToFLAMEParams ─►
LoadSMIRK  ─►                      SMIRKPredict       ─►
                                                          FLAMEParamsMerge ─► FLAMEParamsToMesh ─► MeshPreview
```

**Resolution (Q8):** SMIRK weights are hosted on Google Drive with no registration wall (source: `https://github.com/georgeretsi/smirk`, `SMIRK_em1.pt` Google Drive link). The missing-weight error still names the exact URL — no auto-download, consistent with project policy on provenance and legal risk. SHA256 pin goes to `docs/model_hashes.md` once the hash is confirmed.
