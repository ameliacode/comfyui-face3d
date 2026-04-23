# Layer 2 — Intrinsic Maps Node (IntrinsicMapsBake) — Full Plan

## Scope

**Working direction: path (b).** MoSAR has no released inference code or weights; the `IntrinsicMapsBake` node ships a procedural backend now and leaves `mosar` and `learned` backend slots pluggable for the future. See §Context & MoSAR reality for the three-path framing.

**2048 output resolution dropped from v0.1.** The skeleton included `'2048'` as a Combo option for `output_resolution`. This is dropped in v0.1 to avoid 4× memory overhead (four 2048² float32 maps ≈ 256 MB) and because the normal-map super-resolution path at 2K requires a separate quality review. `'2048'` is a v0.2 candidate (see Q3 resolution). Cross-reference: §Node Surface `output_resolution` input.

In-scope v0.1:
- Single `IntrinsicMapsBake` node consuming the KaoLRM `MESH` and a FreeUV 512² albedo `IMAGE` and emitting specular, AO, translucency, and normal UV maps on FLAME topology via an `INTRINSIC_MAPS` custom wire.
- A `IntrinsicsBackend` protocol in `nodes/intrinsic_backends/__init__.py` with a `_BACKENDS` registry, a `procedural` backend shipping by default, and stubbed `mosar` and `learned` backends that raise `NotImplementedError`.
- Canonical bundled assets: `assets/flame_uv.npz` (pre-baked UV coordinate cache), `assets/flame_pore_density_uv.png`, `assets/flame_translucency_template.png`.
- A `LoadIntrinsicsMoSAR` node (new, parallel to `LoadFreeUV`) surfacing a CC BY-NC-ND gate for the future MoSAR backend. Gate lives on the load node only; `IntrinsicMapsBake` is backend-agnostic.
- `nodes/flame_uv_data.py` helper exposing `load_flame_uvs()` and `make_eyeball_face_mask()`.

Explicitly out of scope for v0.1:
- Training on FFHQ-UV-Intrinsics; shipping MoSAR weights; DECA expression displacement (Layer 3); pore-tile RNM composite (Layer 4); export node; batch size > 1; HiFi3D ↔ FLAME UV converter; output resolution > 1024².
- `specular_tint` as a node input — the hardcoded tint `(0.94, 0.92, 0.89)` ships as a named constant; a per-subject override is v0.2 work. See Brutal Review #2.
- `output_resolution='2048'` — v0.2. See Q3 resolution and note above.

---

## Context & MoSAR Reality

**Resolution (Q1): path (b) confirmed.** The original roadmap framed this layer as a "MoSAR node". Four facts break that framing:

1. **No MoSAR inference code or weights are released.** Only the `FFHQ-UV-Intrinsics` dataset (10k subjects) is public at `github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics`. The `ubisoft-laforge-MoSAR` URL redirects back to the dataset repo — the model repo does not exist as a separate public thing.
2. **MoSAR output UV is HiFi3D topology, not FLAME.** The dataset README explicitly references a separate conversion tutorial for users who want FLAME.
3. **MoSAR does its own geometry fitting** via a GNN-based non-linear morphable model — it cannot be plugged in as a downstream texturizer on KaoLRM's mesh without retraining a texture-only head.
4. **License is CC BY-NC-ND** on the dataset — stricter than KaoLRM's CC BY-NC 4.0 and FreeUV's CC BY-NC-SA 4.0. The ND clause likely forbids generating derivative maps from the dataset at all (see Brutal Review #1).

Three-path summary:
- **(a)** Train a texture/intrinsics head on FFHQ-UV-Intrinsics → FLAME UV. Requires HiFi3D → FLAME UV converter + legal read on the ND clause. Deferred; not specced here.
- **(b) Working direction.** Drop MoSAR framing; ship `IntrinsicMapsBake` with a procedural backend now; leave `mosar` and `learned` slots pluggable.
- **(c)** Punt the layer; watch `ubisoft-laforge` for a weights drop. Rejected; the procedural backend has standalone value for pore-mask AO and geometry-derived normals.

---

## Node Surface

### `IntrinsicMapsBake` — `nodes/intrinsic_maps_bake.py`

```python
class IntrinsicMapsBake(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntrinsicMapsBake",
            display_name="Intrinsic Maps Bake",
            category="KaoLRM",
            description=(
                "Bake specular, AO, translucency, and normal UV maps on FLAME topology "
                "from a KaoLRM mesh and a FreeUV albedo UV. Procedural backend runs with "
                "no additional weights. MoSAR and learned backends are stubbed for future "
                "weight drops. Non-commercial only when driven by KaoLRM / FLAME assets."
            ),
            inputs=[
                io.Mesh.Input(
                    "mesh",
                    tooltip=(
                        "FLAME-topology mesh from KaoLRMReconstruct or FLAMEParamsToMesh. "
                        "Must have faces (num_sampling=5023). Sampled point-cloud inputs "
                        "(num_sampling != 5023, faces empty) are rejected with RuntimeError."
                    ),
                ),
                io.Image.Input(
                    "albedo_uv",
                    tooltip=(
                        "512×512 albedo UV from FreeUVGenerate. Used as colorimetric prior "
                        "for specular tint and translucency luminance modulation. "
                        "Internally upsampled to 1024² for bake operations. "
                        "Batch must be 1."
                    ),
                ),
                MOSAR_MODEL.Input(
                    "mosar_model",
                    optional=True,
                    tooltip=(
                        "Descriptor from LoadIntrinsicsMoSAR. Required only when "
                        "backend='mosar'. Ignored by the procedural backend."
                    ),
                ),
                io.Combo.Input(
                    "backend",
                    options=["procedural", "mosar", "learned"],
                    default="procedural",
                    optional=True,
                    tooltip=(
                        "'procedural' — geometry + albedo pipeline; no weights required. "
                        "'mosar' — stub; raises NotImplementedError until weights land. "
                        "'learned' — stub; no candidate selected for v0.1. "
                        "See Brutal Review #12 for the rationale for retaining the learned slot."
                    ),
                ),
                io.Float.Input(
                    "specular_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Multiplicative scale on specular output. 1.0 = unchanged.",
                ),
                io.Float.Input(
                    "ao_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Multiplicative scale on AO output. 1.0 = unchanged.",
                ),
                io.Float.Input(
                    "translucency_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Multiplicative scale on translucency output. 1.0 = unchanged.",
                ),
                io.Float.Input(
                    "normal_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip=(
                        "Multiplicative scale applied to the XY channels of the tangent-space "
                        "normal before re-normalizing. 1.0 = unmodified geometry normals; "
                        "values < 1.0 flatten the surface detail."
                    ),
                ),
                io.Image.Input(
                    "pore_density_mask",
                    optional=True,
                    tooltip=(
                        "Optional 1024×1024 single-channel pore density mask in FLAME UV layout. "
                        "Falls back to bundled 'assets/flame_pore_density_uv.png' when not wired."
                    ),
                ),
                io.Combo.Input(
                    "output_resolution",
                    options=["512", "1024"],
                    default="1024",
                    optional=True,
                    tooltip=(
                        "UV map output resolution. 1024 matches the roadmap AO/specular bit-depth "
                        "target and the canonical pore-density mask resolution. Internal bake "
                        "always runs at 1024; '512' downsamples on output. "
                        "'2048' is deferred to v0.2 to avoid 4× memory overhead."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2**31 - 1,
                    optional=True,
                    tooltip=(
                        "Reserved for v0.2 noise-based procedural backends. "
                        "Ignored by the v0.1 procedural backend (pore placement is a static PNG; "
                        "no stochastic components). -1 or any fixed value behave identically "
                        "in v0.1."
                    ),
                ),
            ],
            outputs=[
                INTRINSIC_MAPS.Output(display_name="intrinsic_maps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        albedo_uv: torch.Tensor,
        mosar_model=None,
        backend: str = "procedural",
        specular_strength: float = 1.0,
        ao_strength: float = 1.0,
        translucency_strength: float = 1.0,
        normal_strength: float = 1.0,
        pore_density_mask=None,
        output_resolution: str = "1024",
        seed: int = -1,
    ) -> io.NodeOutput:
        # (1) Point-cloud rejection
        verts = mesh.vertices   # [B, V, 3]
        faces = mesh.faces      # [B, F, 3]
        if faces.numel() == 0 or verts.shape[1] != 5023:
            raise RuntimeError(
                "IntrinsicMapsBake requires a FLAME-topology mesh with 5023 vertices and "
                "9976 faces. Got a sampled point cloud (num_sampling != 5023). Re-run "
                "KaoLRMReconstruct with num_sampling=5023 or pass the mesh through "
                "FLAMEParamsToMesh."
            )
        # (2) Batch rejection
        if albedo_uv.shape[0] != 1:
            raise RuntimeError(
                f"IntrinsicMapsBake: albedo_uv batch size must be 1, got {albedo_uv.shape[0]}."
            )
        # (3) Output resolution int-coercion
        res = int(output_resolution or "1024")
        # (4) Resolve device and seed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actual_seed = seed if seed != -1 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        # (5) Build aux dict
        aux = {
            "pore_density_mask": pore_density_mask,
            "seed": actual_seed,
            "normal_strength": normal_strength,
        }
        # (6) Dispatch to backend
        maps = get_backend(backend).bake(mesh, albedo_uv, aux, device=device)
        # (7) Post-bake strength-slider multiply
        maps["specular"]     = (maps["specular"]     * specular_strength).clamp(0.0, 1.0)
        maps["ao"]           = (maps["ao"]           * ao_strength).clamp(0.0, 1.0)
        maps["translucency"] = (maps["translucency"] * translucency_strength).clamp(0.0, 1.0)
        # normal_strength already applied inside bake() per §Normal Map algorithm
        # (8) Downsample if res == 512
        if res == 512:
            for k in ("specular", "ao", "translucency", "normal"):
                t = maps[k].permute(0, 3, 1, 2)  # BHWC → BCHW
                t = F.interpolate(t, size=(512, 512), mode="bicubic", antialias=True)
                maps[k] = t.permute(0, 2, 3, 1)  # BCHW → BHWC
        maps["resolution"] = res
        # (9) Validate
        validate_intrinsic_maps(maps, source="IntrinsicMapsBake")
        # (10) Return
        return io.NodeOutput(maps)
```

**Point-cloud rejection** raises `RuntimeError` naming `num_sampling=5023` and `FLAMEParamsToMesh` as the remedy.

**Batch enforcement** raises `RuntimeError` if `albedo_uv.shape[0] != 1`. Batch size > 1 is v0.2.

**Output resolution** internal bake always runs at 1024². When `output_resolution == "512"`, all four maps are downsampled with `F.interpolate(..., size=(512, 512), mode="bicubic", antialias=True)` before packing into the `INTRINSIC_MAPS` dict.

---

## Wire Types

**Resolution (Q2): single `INTRINSIC_MAPS` custom type.** Five separate `io.Image` wires would be ComfyUI-idiomatic for each individual map, but a single custom type composes cleanly with the future Export node, allows a single validation call, and matches the `FLAME_PARAMS` precedent where a structured dict travels better than five separate tensors. Users who want individual maps can unpack them with a trivial `IntrinsicMapsSplit` utility node added later.

Declaration lives in `nodes/intrinsic_maps_wire.py`:

```python
from comfy_api.latest import io

INTRINSIC_MAPS = io.Custom("INTRINSIC_MAPS")

CANONICAL_MAP_SHAPES = {
    "specular":      (3,),   # [B, H, W, 3] float32 in [0, 1]
    "ao":            (1,),   # [B, H, W, 1] float32 in [0, 1]
    "translucency":  (3,),   # [B, H, W, 3] float32 in [0, 1], RGB scatter
    "normal":        (3,),   # [B, H, W, 3] float32 in [0, 1], OpenGL tangent-space
}

def validate_intrinsic_maps(maps: dict, *, source: str) -> None:
    """Assert [B, H, W, C] layout and canonical channel counts. Raises RuntimeError on mismatch."""
    for key, expected_channels in CANONICAL_MAP_SHAPES.items():
        if key not in maps:
            raise RuntimeError(
                f"INTRINSIC_MAPS from {source!r} missing key {key!r}."
            )
        t = maps[key]
        if t.ndim != 4 or t.shape[-1] != expected_channels[0]:
            raise RuntimeError(
                f"INTRINSIC_MAPS from {source!r}: {key!r} must be "
                f"[B, H, W, {expected_channels[0]}], got {tuple(t.shape)}."
            )
    if "resolution" not in maps:
        raise RuntimeError(
            f"INTRINSIC_MAPS from {source!r} missing 'resolution' int."
        )
```

Wire schema — plain Python dict:

```python
{
    "specular":     torch.Tensor,  # [1, H, W, 3] float32 in [0, 1]
    "ao":           torch.Tensor,  # [1, H, W, 1] float32 in [0, 1]
    "translucency": torch.Tensor,  # [1, H, W, 3] float32 in [0, 1] — RGB scatter, not scalar-in-R
    "normal":       torch.Tensor,  # [1, H, W, 3] float32 in [0, 1] — OpenGL tangent-space
    "resolution":   int,           # 512 or 1024, matches the actual H and W above
}
```

All tensors on CPU. Encoding note: `normal` stores OpenGL tangent-space; the neutral tangent-space normal `(0, 0, 1)` encodes to `[0.5, 0.5, 1.0]` after `(v + 1) * 0.5`. Green-flip for DirectX/Unreal is the Export node's responsibility, not this node's.

---

## Backend Interface

### `IntrinsicsBackend` protocol — `nodes/intrinsic_backends/__init__.py`

```python
from __future__ import annotations
from typing import Protocol
import torch

class IntrinsicsBackend(Protocol):
    name: str

    def bake(
        self,
        mesh,               # MeshPayload — vertices [1, 5023, 3], faces [1, 9976, 3]
        albedo_uv: torch.Tensor,  # [1, H, W, 3] float32 in [0, 1]
        aux: dict,          # pore_density_mask, seed, curvature_scale, etc.
        *,
        device: str,        # "cuda" | "cpu" — preserved for learned backend
    ) -> dict[str, torch.Tensor]:
        """Return a dict with keys 'specular', 'ao', 'translucency', 'normal'.
        All tensors [1, 1024, 1024, C] float32 in [0, 1], on CPU.
        Procedural backend ignores device; learned backend moves tensors there.
        """
        ...
```

Backend registration — module-level dict in `nodes/intrinsic_backends/__init__.py`:

```python
from .procedural import ProceduralBackend
from .mosar_stub import MosarStubBackend
from .learned_stub import LearnedStubBackend

_BACKENDS: dict[str, IntrinsicsBackend] = {
    "procedural": ProceduralBackend(),
    "mosar":      MosarStubBackend(),
    "learned":    LearnedStubBackend(),
}

def get_backend(name: str) -> IntrinsicsBackend:
    if name not in _BACKENDS:
        raise RuntimeError(
            f"Unknown IntrinsicsBackend {name!r}. Valid options: {list(_BACKENDS)}."
        )
    return _BACKENDS[name]
```

`_BACKENDS` is populated exactly once at package import. No registration is scattered across other modules. `IntrinsicMapsBake.execute()` dispatches via `get_backend(backend).bake(...)`.

### Backend files

```
nodes/intrinsic_backends/__init__.py    # protocol, _BACKENDS registry, get_backend()
nodes/intrinsic_backends/procedural.py  # ProceduralBackend — all four maps
nodes/intrinsic_backends/mosar_stub.py  # MosarStubBackend — NotImplementedError
nodes/intrinsic_backends/learned_stub.py # LearnedStubBackend — NotImplementedError
```

#### `mosar_stub.py`

```python
class MosarStubBackend:
    name = "mosar"

    def bake(self, mesh, albedo_uv, aux, *, device):
        raise NotImplementedError(
            "The 'mosar' backend is a stub. LoadIntrinsicsMoSAR itself raises "
            "NotImplementedError before it can return a descriptor — MoSAR weights "
            "are not yet publicly available. This backend slot is reserved for when "
            "weights drop. Place weights at 'ComfyUI/models/mosar/' and replace "
            "mosar_stub.py when released. "
            "Track: https://github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics"
        )
```

#### `learned_stub.py`

```python
class LearnedStubBackend:
    name = "learned"

    def bake(self, mesh, albedo_uv, aux, *, device):
        raise NotImplementedError(
            "The 'learned' backend has no candidate model selected for v0.1. "
            "This slot is reserved for a future intrinsic-decomposition network. "
            "Use backend='procedural' for geometry-derived maps."
        )
```

**Resolution (Q7): plain `NotImplementedError` stub with no candidate model.** AvatarMe++ (TPAMI 2021) and NextFace (Eurographics 2021) are FLAME-compatible candidates, but neither has a clean public checkpoint for FLAME-native UV inference. The `'learned'` key is retained in `_BACKENDS` as a stable API surface so user workflows that reference the name do not break when a candidate eventually lands; see Brutal Review #12 for the justification. Over-committing to a specific network now would require retracting the claim when weights don't materialize.

---

## Procedural Backend

All four maps in `ProceduralBackend.bake()` are pure torch + numpy — no new `requirements.txt` entries.

**Albedo resampling at bake entry.** The procedural backend's internal resolution is 1024². Albedo from FreeUV arrives at 512². At the top of `ProceduralBackend.bake()`, before any per-vertex sampling or texel lookup, albedo is upsampled once:

```python
albedo_1024 = F.interpolate(
    albedo_uv.permute(0, 3, 1, 2),  # BHWC → BCHW
    size=(1024, 1024),
    mode="bicubic",
    antialias=True,
).permute(0, 2, 3, 1)  # BCHW → BHWC — [1, 1024, 1024, 3]
```

All subsequent operations that read albedo (specular luminance, translucency luminance blend, `_sample_uv_at_vertices()`) use `albedo_1024`.

### FLAME UV Helper — `nodes/flame_uv_data.py`

All procedural rasterization goes through this module. Its central function, `load_flame_uvs()`, is cached at module scope:

```python
_FLAME_UV_CACHE: dict | None = None

def load_flame_uvs() -> dict:
    """Load the pre-baked FLAME UV coordinate cache. Returns a dict with:
        uv_coords:  torch.Tensor [5023, 2]  — per-vertex UV in [0, 1]
        faces_uv:   torch.Tensor [9976, 3]  — face indices into uv_coords
        bary_lut:   torch.Tensor [1024, 1024, 6] — per-texel layout:
                    [face_idx, w0, w1, w2, eyeball_flag, unused]; face_idx == -1
                    for texels not covered by any FLAME UV face.
    Reads from 'assets/flame_uv.npz'. Raises RuntimeError if not found.
    """
    global _FLAME_UV_CACHE
    if _FLAME_UV_CACHE is None:
        asset_path = Path(__file__).resolve().parents[1] / "assets" / "flame_uv.npz"
        if not asset_path.exists():
            raise RuntimeError(
                f"FLAME UV coordinate cache not found at '{asset_path}'. "
                "This file ships with the repo — check your install. "
                "To regenerate, run 'scripts/bake_flame_uv.py' with the FLAME pkl at "
                "'ComfyUI/models/flame/generic_model.pkl'."
            )
        data = np.load(str(asset_path))
        _FLAME_UV_CACHE = {
            "uv_coords": torch.from_numpy(data["uv_coords"]).float(),
            "faces_uv":  torch.from_numpy(data["faces_uv"]).long(),
            "bary_lut":  torch.from_numpy(data["bary_lut"]).float(),
        }
    return _FLAME_UV_CACHE
```

**Resolution (Q10): ship `assets/flame_uv.npz` pre-baked.** Deriving at runtime requires the FLAME pkl and the chumpy shim; that pulls `_install_chumpy_shims()` into tests that currently run with no weights. Pre-baking removes this dependency from all procedural-backend tests. See Q10 resolution in §Open Questions.

Authoring script `scripts/bake_flame_uv.py` reads `generic_model.pkl`, extracts `v_template`, `f`, and the UV layout from the FLAME mesh, and writes the npz. Needs the pkl; runs once; output is checked in as a repo asset. The script must print and assert the eyeball vertex range before writing the npz — this assert is the single source of truth for the `make_eyeball_face_mask` defaults (see §Eyeball face mask below and Brutal Review #4).

#### UV sampling at vertices

```python
def _sample_uv_at_vertices(
    image_BCHW: torch.Tensor,   # [1, C, H, W] float32
    uv_coords: torch.Tensor,    # [V, 2] float32 in [0, 1]
) -> torch.Tensor:
    """Sample an image at per-vertex UV coordinates using bilinear interpolation.

    Returns [V, C] float32. Uses F.grid_sample with align_corners=False.
    UV coords are remapped from [0, 1] to [-1, 1] for grid_sample convention.
    """
    # grid_sample expects [B, 1, V, 2] grid in [-1, 1]
    grid = (uv_coords * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # [1, 1, V, 2]
    sampled = F.grid_sample(image_BCHW, grid, mode="bilinear", align_corners=False)
    # sampled: [1, C, 1, V] → [V, C]
    return sampled[0, :, 0, :].permute(1, 0)
```

This helper is used by the specular and translucency sections to convert per-texel `albedo_1024` → per-vertex luminance (`lum_vertex`) and per-vertex pore density (`pore_vertex`).

#### Eyeball face mask

```python
def make_eyeball_face_mask(faces: torch.Tensor, lo: int = 3931, hi: int = 5023) -> torch.Tensor:
    """Return a [F] bool mask — True for faces that touch at least one eyeball vertex.

    Default range lo=3931, hi=5023 is sourced from FLAME-Universe documentation.
    TODO(research): confirm exact eyeball vertex range against FLAME 2020 generic_model.pkl
    by running `np.where(model.f > 3930)` on the packed faces tensor before authoring
    flame_uv.npz. This is an M0 exit criterion — the range must be asserted and printed
    by scripts/bake_flame_uv.py before the npz is committed; the asserted value becomes
    the single source of truth for these defaults.

    Any face with one or more indices in [lo, hi) is masked; the [lo, hi) interval is
    half-open to match Python slice convention.
    Usage: zero out masked texels in every UV map to prevent the (1,1) UV corner artifact.
    """
    return ((faces >= lo) & (faces < hi)).any(dim=-1)  # [F] bool
```

`make_eyeball_face_mask` is applied in `_bake_vertex_to_uv()` — any texel whose covering face is in the eyeball mask is clamped to `(1.0, 1.0)` UV and zeroed out in all four output maps.

### AO Map

**Algorithm: discrete cotangent-Laplacian mean curvature proxy.**

Per vertex `v`, the discrete Laplace-Beltrami operator gives the mean curvature normal:

```
L_v = 0.5 * Σ_{j ∈ N(v)} (cot(α_ij) + cot(β_ij)) * (x_j - x_v)
|H_v| = ‖L_v‖ / (2 A_v)
```

where `α_ij` and `β_ij` are the angles opposite edge `(v, j)` in the two incident triangles, and `A_v` is the Voronoi area of vertex `v` (sum of one-third of each incident triangle area).

AO is then:
```
AO(v) = clamp(1.0 - curvature_scale * |H_v|, 0.0, 1.0)
```

with `curvature_scale = 4.0` as the committed default (higher = more pronounced shadowing in creases). This default is pinned in a named test: `test_intrinsic_backends_procedural.py::test_ao_curvature_scale_default` checks that baking with a synthetic sphere mesh at `curvature_scale=4.0` produces an AO map with mean value in `[0.85, 1.0]` (a sphere has low absolute curvature relative to FLAME face geometry).

The cotangent weights and Voronoi areas are computed in pure torch:

```python
def _cotangent_laplacian_curvature(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return per-vertex mean curvature magnitude [V] float32."""
    # verts: [V, 3], faces: [F, 3] long
    v0 = verts[faces[:, 0]]   # [F, 3]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e01 = v1 - v0; e02 = v2 - v0
    e10 = v0 - v1; e12 = v2 - v1
    e20 = v0 - v2; e21 = v1 - v2

    def _cot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cross = torch.cross(a, b, dim=-1).norm(dim=-1).clamp(min=1e-8)
        dot   = (a * b).sum(-1)
        return dot / cross  # [F]

    cot0 = _cot(e10, e12)  # angle at vertex 1
    cot1 = _cot(e20, e21)  # angle at vertex 2
    cot2 = _cot(e01, e02)  # angle at vertex 0

    V = verts.shape[0]
    L = torch.zeros(V, 3, dtype=verts.dtype, device=verts.device)  # weighted sum of (x_j - x_v)
    A = torch.zeros(V,    dtype=verts.dtype, device=verts.device)  # Voronoi area accumulator

    # Scatter cotangent-weighted edge vectors onto each vertex
    for fi_v, fj_v, w_ij in [
        (faces[:, 0], faces[:, 1], cot2),
        (faces[:, 1], faces[:, 0], cot2),
        (faces[:, 1], faces[:, 2], cot0),
        (faces[:, 2], faces[:, 1], cot0),
        (faces[:, 0], faces[:, 2], cot1),
        (faces[:, 2], faces[:, 0], cot1),
    ]:
        diff = verts[fj_v] - verts[fi_v]
        L.scatter_add_(0, fi_v.unsqueeze(-1).expand_as(diff),
                       (w_ij.unsqueeze(-1) * diff))

    L = L * 0.5
    # Face areas (triangle area = 0.5 * |e01 × e02|)
    face_area = torch.cross(e01, e02, dim=-1).norm(dim=-1) * 0.5
    for fi_v in [faces[:, 0], faces[:, 1], faces[:, 2]]:
        A.scatter_add_(0, fi_v, face_area / 3.0)
    A = A.clamp(min=1e-10)

    curvature_mag = L.norm(dim=-1) / (2.0 * A)  # [V] — |H_v|
    return curvature_mag
```

Bake vertex AO to 1024² UV via `_bake_vertex_to_uv(ao_values, uv_data, eyeball_mask)` using the precomputed barycentric LUT in `flame_uv.npz`. Result is a `[1, 1024, 1024, 1]` float32 tensor. No `pytorch3d` required; no ray-casting.

**No new dependency** — no `trimesh`, no `pyembree`. Pure torch scatter-add operations are sufficient for cotangent weights on a fixed 5023-vertex, 9976-face mesh.

### Specular Map

**Signal:** cavity (inverted AO) × albedo-luminance inverse × optional pore-density modulation.

```python
specular_v = (1.0 - ao_vertex) * (1.0 - lum_vertex) * pore_vertex  # all [V] float32
```

where:
- `ao_vertex` is the AO per vertex from the curvature computation above.
- `lum_vertex` is the per-vertex luminance sampled from `albedo_1024` (the 1024²-upsampled albedo) at each vertex's UV via `_sample_uv_at_vertices()`: `lum = 0.2126*R + 0.7152*G + 0.0722*B`. High-luminance skin (sebum flats) → lower specular; dark creases → higher specular.
- `pore_vertex` is the per-vertex pore density sampled from the pore-density mask at each vertex's UV via `_sample_uv_at_vertices()`. **Fallback:** when `assets/flame_pore_density_uv.png` is absent, `pore_vertex` defaults to `1.0` everywhere and a warning is logged once via `log.warning(...)`. M0 ships and passes tests without the mask asset.

The scalar specular value is broadcast to RGB by multiplying against the neutral specular tint constant `SPECULAR_TINT = (0.94, 0.92, 0.89)` (a warm near-white appropriate for Caucasian/Asian skin). A per-subject `specular_tint` node input is out of scope for v0.1 — see §Scope. Output: `[1, 1024, 1024, 3]` float32.

Lips/eyebrow mask is not applied in v0.1 — dark hair reads as high-spec but the error is small compared to the geometry-driven signal. Mark as a v0.2 improvement.

### Translucency Map

**Signal:** canonical FLAME UV translucency template, modulated by albedo luminance.

The template `assets/flame_translucency_template.png` is authored as a 1024² 8-bit RGB PNG with per-region scatter colour pinned to the following FLAME UV regions:

| Region | RGB scatter colour | Rationale |
|---|---|---|
| Ears | `(0.85, 0.35, 0.25)` | High red-channel transmission (blood vessel density) |
| Cheeks | `(0.75, 0.55, 0.45)` | Moderate scattering, warm tint |
| Forehead | `(0.60, 0.50, 0.45)` | Lower scatter, thicker bone proximity |
| Nose | `(0.80, 0.40, 0.30)` | High translucency (thin cartilage) |
| Chin / jaw | `(0.65, 0.48, 0.40)` | Moderate, slightly cooler |
| Lips | `(0.90, 0.45, 0.40)` | High red, mucosa |
| Eyelids | `(0.70, 0.50, 0.42)` | Thin skin, moderate |
| Eyeballs | `(0, 0, 0)` | Zeroed out by eyeball mask |

Encoding: RGB scatter, **not scalar-in-R**. Three-channel output matches the roadmap's 16-bit PNG master target and allows engine-specific remapping without information loss.

Authoring procedure: painted in Substance 3D Painter or Mari on the FLAME UV template, using the FLAME UV island layout as a reference. The template is subject-independent — skin tone modulation comes from the albedo luminance term. Author once; ship as a repo asset.

Runtime blend (`template` and `lum_uv` are both at 1024² — `template` is loaded as 1024², and `albedo_1024` is the upsampled albedo from the bake-entry upsample step described at the top of §Procedural Backend):

```python
template = load_translucency_template()  # [1, 1024, 1024, 3] cached
lum_uv = rgb_to_luminance(albedo_1024)  # [1, 1024, 1024, 1] — uses upsampled albedo
# Luminance darkening: thick skin reads as less translucent
translucency = template * (1.0 - 0.3 * lum_uv)
translucency = translucency.clamp(0.0, 1.0)
```

`translucency_strength` is applied post-blend in `execute()` (step 7 of the canonical execute body): `maps["translucency"] = (maps["translucency"] * translucency_strength).clamp(0.0, 1.0)`. This follows the same multiplicative pattern as `specular_strength` and `ao_strength`.

Value range: `[0, 1]` float32 per channel. An implementer can convert to 16-bit PNG via `(t * 65535).round().to(torch.uint16)` at export time.

### Normal Map

**Algorithm: per-texel tangent-space normal from mesh geometry.**

For each triangle `(A, B, C)` with UV coordinates `(UV_A, UV_B, UV_C)`:

```
dP1 = B - A    dP2 = C - A
dUV1 = UV_B - UV_A  dUV2 = UV_C - UV_A
det = dUV1.u * dUV2.v - dUV2.u * dUV1.v
```

If `|det| < 1e-8`, the UV triangle is degenerate. Fallback: output `(0.5, 0.5, 1.0)` in `[0, 1]` encoding (pure up-facing normal), log once via `log.warning("degenerate UV triangle at face %d — using flat normal fallback", fi)`.

Otherwise:

```
T = (dUV2.v * dP1 - dUV1.v * dP2) / det   # tangent [3]
B_ = (-dUV2.u * dP1 + dUV1.u * dP2) / det  # bitangent [3]
N = vertex_normal at A (interpolated over face)
TBN = [normalize(T), normalize(B_), normalize(N)]
```

Per-texel normal in tangent space = `TBN^T @ geometry_normal` (which yields `(0, 0, 1)` for a flat face by construction). In v0.1, the geometry normal is the only signal — the pore-tile meso-normal RNM composite is Layer 4.

Output encoding (with Z-clamp applied before encoding to prevent hemisphere flip — see Brutal Review #10):

```python
n_scaled = torch.stack([
    n_tangent[..., 0] * normal_strength,
    n_tangent[..., 1] * normal_strength,
    n_tangent[..., 2],  # Z channel NOT scaled
], dim=-1)
n_norm = n_scaled / n_scaled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
# Clamp Z to >= 0.1 to prevent hemisphere flip at extreme normal_strength values
# (see Brutal Review #10). Renormalize after clamp.
n_norm[..., 2] = n_norm[..., 2].clamp(min=0.1)
n_norm = n_norm / n_norm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
n_out = (n_norm + 1.0) * 0.5  # encode to [0, 1]
```

OpenGL convention (+Y up). Green-flip for Unreal/DirectX is Layer 5 Export's responsibility.

Output tensor: `[1, 1024, 1024, 3]` float32. The roadmap calls for 16-bit PNG; the tensor is float32 in-pipeline and the Export node handles quantization.

### Displacement Map (Q4)

**Resolution (Q4): Layer 3 (DECA) owns displacement. No `displacement` key in `INTRINSIC_MAPS` for v0.1.**

A neutral-pose coarse displacement could be derived from curvature in Layer 2 (the same cotangent Laplacian signal, signed-projected along the vertex normal). However:

- DECA's displacement is expression-specific and represents a strict superset of any geometry-derived neutral displacement.
- A coarse Layer 2 displacement would need to be subtracted at compositing time to avoid double-counting, adding complexity.

If the user reverses this decision before M1, add a `displacement` key to `CANONICAL_MAP_SHAPES` with shape `(1,)` (single-channel EXR-range value) and update `flame_uv.npz` authoring to include a displacement-compatible LUT. The M1 exit criteria includes confirming this decision is final.

---

## Assets Required

| Asset | Path | Format | Resolution | Authoring | Fallback | Status |
|---|---|---|---|---|---|---|
| FLAME UV coordinate cache | `assets/flame_uv.npz` | NumPy npz (`uv_coords [5023,2]`, `faces_uv [9976,3]`, `bary_lut [1024,1024,6]`) | Derived | `scripts/bake_flame_uv.py` + FLAME pkl | None — `RuntimeError` if absent | M0 blocker |
| FLAME pore-density mask UV | `assets/flame_pore_density_uv.png` | 8-bit grayscale PNG | 1024² | Hand-painted in Substance 3D Painter / Mari on FLAME UV template | Uniform `1.0` everywhere; `log.warning` once | M1 blocker |
| FLAME translucency template | `assets/flame_translucency_template.png` | 8-bit RGB PNG | 1024² | Hand-painted on FLAME UV template; per-region scatter RGB per §Translucency Map | None — `RuntimeError` if absent | M1 blocker |
| Texturing.xyz pore tile set | `assets/pore_tiles/` | 16-bit TIFF or EXR | tileable | External acquisition via Texturing.xyz commercial license | Omit entirely | out of scope — Layer 4 only |

**Licensing note on Texturing.xyz tiles.** The pore tile set is a Layer 4 asset (skin detail composite, Barré-Brisebois & Hill RNM). Do not ship under `assets/pore_tiles/` in v0.1; it is a Layer 4 problem and requires a separate licensing review.

**`flame_uv.npz` authoring detail.** The `bary_lut` array is a `[1024, 1024, 6]` float32 tensor where each texel stores `[face_idx_float, w0, w1, w2, eyeball_flag, unused]`. Texels not covered by any FLAME UV face store `face_idx = -1`; the bake function skips them. The LUT is computed once by `scripts/bake_flame_uv.py` using rasterization over the FLAME UV triangle mesh. Binary search per-texel is not needed — the LUT is pre-computed and loaded at node init time.

---

## Weights and Vendor Layout

Reserve `ComfyUI/models/mosar/` per the one-model-type-per-folder convention in CLAUDE.md. This directory stays empty in v0.1. The `mosar_stub` backend's `NotImplementedError` message points at `ComfyUI/models/mosar/` so the path is stable when weights eventually drop.

Expected future filenames (when MoSAR releases inference code and weights):

```
ComfyUI/models/mosar/
```

TODO(research): confirm exact filenames from ubisoft-laforge-MoSAR when/if the model
repo separates from the dataset repo. Paper references a "GNN-based non-linear morphable
model" — likely one or more `.bin` or `.safetensors` checkpoint files. No filename can be
pinned from the paper alone.

No `third_party/mosar/` vendor dir in v0.1. No `requirements.txt` additions — the procedural backend is pure torch + numpy + PIL.

**Updated "Fixed asset paths" for CLAUDE.md:**

Add to the existing list:
```
ComfyUI/models/mosar/              (reserved, empty in v0.1)
assets/flame_uv.npz                (pre-baked FLAME UV coordinate cache)
assets/flame_pore_density_uv.png   (1024² grayscale pore density mask)
assets/flame_translucency_template.png (1024² RGB scatter template)
assets/MOSAR_NOTICE.txt            (M2 blocker — authored when MoSAR stub activates)
```

---

## Non-Commercial + No-Derivatives Gate

**Procedural backend has no NC/ND surface.** It derives maps from mesh geometry and the existing FreeUV albedo — no additional restricted dataset is consumed. `IntrinsicMapsBake` itself carries no `i_understand_non_commercial` gate. This keeps the backend-agnostic node runnable without a gate when the pipeline uses only procedural outputs.

**`LoadIntrinsicsMoSAR` — `nodes/mosar_load.py`** (new node, parallel to `LoadFreeUV`):

```python
MOSAR_MODEL = io.Custom("MOSAR_MODEL")

class LoadIntrinsicsMoSAR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadIntrinsicsMoSAR",
            display_name="Load Intrinsics MoSAR",
            category="KaoLRM",
            description=(
                "Resolve MoSAR intrinsic-decomposition model assets and output a descriptor. "
                "MoSAR dataset (FFHQ-UV-Intrinsics) is CC BY-NC-ND 4.0 — stricter than "
                "KaoLRM and FreeUV. The ND clause may prohibit generating derivative maps. "
                "This node is a stub; weights are not yet publicly available."
            ),
            inputs=[
                io.Combo.Input("device", options=["auto", "cpu", "cuda"], default="auto",
                    optional=True,
                    tooltip="'auto' picks cuda when available, cpu otherwise."),
                io.Combo.Input("dtype", options=["auto", "fp32", "fp16", "bf16"], default="auto",
                    optional=True,
                    tooltip="'auto' → fp32. fp16 support depends on upstream model."),
                io.Boolean.Input(
                    "i_understand_non_commercial_no_derivatives",
                    default=False,
                    tooltip=(
                        "MoSAR's training data (FFHQ-UV-Intrinsics, Ubisoft La Forge) is licensed "
                        "under CC BY-NC-ND 4.0. ND = No Derivatives. This clause is STRICTER than "
                        "KaoLRM's CC BY-NC 4.0 and FreeUV's CC BY-NC-SA 4.0. "
                        "You may not distribute adapted material including generated intrinsic maps "
                        "derived from the MoSAR model without explicit permission from Ubisoft. "
                        "Non-commercial use only. Read assets/MOSAR_NOTICE.txt before enabling."
                    ),
                ),
            ],
            outputs=[MOSAR_MODEL.Output(display_name="mosar_model")],
        )

    @classmethod
    def execute(cls, device="auto", dtype="auto",
                i_understand_non_commercial_no_derivatives: bool = False):
        if not i_understand_non_commercial_no_derivatives:
            raise RuntimeError(
                "Set 'i_understand_non_commercial_no_derivatives' to True before using "
                "LoadIntrinsicsMoSAR. The MoSAR dataset is CC BY-NC-ND 4.0 — the ND clause "
                "may prohibit generating derivative maps. Read assets/MOSAR_NOTICE.txt."
            )
        raise NotImplementedError(
            "MoSAR weights are not yet publicly available. "
            "Place weights at 'ComfyUI/models/mosar/' when released. "
            "Track: https://github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics"
        )
```

**Gate field name:** `i_understand_non_commercial_no_derivatives` — distinct from the existing `i_understand_non_commercial` on `LoadKaoLRM` / `LoadSMIRK` / `LoadFreeUV`. The longer name surfaces the ND escalation so a user cannot confuse it with the standard NC gate.

**`assets/MOSAR_NOTICE.txt`** — Mark as **M2 blocker**. Author when the MoSAR stub activates (i.e., when real weights are available and the stub is being replaced with a real implementation). Content will include: Ubisoft La Forge attribution, CC BY-NC-ND 4.0 full license text URL, and a plain-language explanation of the ND clause.

---

## Integration with Existing Pipeline

```
LoadKaoLRM ─► KaoLRMPreprocess ─► KaoLRMReconstruct ─► MESH ─────────────────────────┐
                                                                                       │
LoadFreeUV ─► FreeUVGenerate ─► IMAGE (albedo_uv, 512²) ──────────────────────────────┤
                                                                                       ▼
                                                               IntrinsicMapsBake ─► INTRINSIC_MAPS
                                                                        ▲
                                                               (optional) LoadIntrinsicsMoSAR ─► MOSAR_MODEL
                                                                        │
                                                                        ▼
                                                               (future) DECADetail (Layer 3)
                                                                        │
                                                                        ▼
                                                               (future) SkinDetailComposite (Layer 4)
                                                                        │
                                                                        ▼
                                                               (future) Export node (Layer 5)
```

`IntrinsicMapsBake` consumes both the `MESH` from KaoLRM and the `IMAGE` from FreeUV, producing an `INTRINSIC_MAPS` bundle that travels downstream through DECA, the skin composite, and eventually the export node.

---

## File Layout and Registration

New files:

```
nodes/intrinsic_maps_bake.py            # IntrinsicMapsBake node
nodes/intrinsic_maps_wire.py            # INTRINSIC_MAPS custom type + validate_intrinsic_maps()
nodes/intrinsic_backends/__init__.py    # IntrinsicsBackend protocol, _BACKENDS, get_backend()
nodes/intrinsic_backends/procedural.py  # ProceduralBackend
nodes/intrinsic_backends/mosar_stub.py  # MosarStubBackend
nodes/intrinsic_backends/learned_stub.py # LearnedStubBackend
nodes/flame_uv_data.py                  # load_flame_uvs(), make_eyeball_face_mask()
nodes/mosar_load.py                     # LoadIntrinsicsMoSAR + MOSAR_MODEL wire
assets/flame_uv.npz                     # pre-baked UV coordinate cache (M0 blocker)
assets/flame_pore_density_uv.png        # 1024² grayscale pore density (M1 blocker)
assets/flame_translucency_template.png  # 1024² RGB scatter template (M1 blocker)
assets/MOSAR_NOTICE.txt                 # CC BY-NC-ND notice (M2 blocker)
scripts/bake_flame_uv.py               # one-shot authoring script for flame_uv.npz
```

`nodes/__init__.py` additions:

```python
from .intrinsic_maps_bake import IntrinsicMapsBake
from .mosar_load import LoadIntrinsicsMoSAR

NODE_CLASSES = [
    # ... existing ...
    IntrinsicMapsBake,       # new
    LoadIntrinsicsMoSAR,     # new
]
```

Category: `"KaoLRM"` for v0.1. Migrate to `WYSIWYG/Face/...` in the v0.2 multi-suite reorganization per CLAUDE.md.

---

## Tests

All v0.1 tests pass with no real weights present. The procedural backend uses the pre-baked `assets/flame_uv.npz` plus synthetic mesh fixtures — no FLAME pkl required.

### `tests/test_intrinsic_maps_wire.py`

1. `test_valid_maps_passes_validation` — construct a valid `INTRINSIC_MAPS` dict and assert `validate_intrinsic_maps` returns without error.
2. `test_missing_key_raises` — omit `"ao"` key, assert `RuntimeError` naming the missing key.
3. `test_wrong_channel_count_raises` — pass a 1-channel tensor for `"specular"` (should be 3), assert error naming the expected shape.
4. `test_missing_resolution_raises` — omit `"resolution"` key, assert `RuntimeError`.
5. `test_source_name_in_error` — assert the `source` kwarg appears in the exception message.

### `tests/test_flame_uv_data.py`

1. `test_load_flame_uvs_returns_expected_keys` — monkeypatch asset path to a synthetic npz; assert `uv_coords`, `faces_uv`, `bary_lut` keys present with correct dtypes.
2. `test_load_flame_uvs_caches` — call twice; assert same dict object returned (`is` identity).
3. `test_load_flame_uvs_missing_raises` — point at nonexistent path; assert `RuntimeError` naming the path.
4. `test_make_eyeball_face_mask_shape` — synthetic `faces [F, 3]`; assert output shape `[F]` bool.
5. `test_make_eyeball_face_mask_includes_boundary` — face with vertex `3931` → `True`; face with all vertices `< 3931` → `False`.
6. `test_make_eyeball_face_mask_excludes_upper_bound` — face with vertex `5023` → `False` (half-open interval).

### `tests/test_intrinsic_assets.py`

Covers the pore-density and translucency template asset loaders, complementing `test_flame_uv_data.py`'s coverage of the UV coordinate cache.

1. `test_load_pore_density_returns_correct_shape` — monkeypatch `assets/flame_pore_density_uv.png` to a synthetic 1024² grayscale PNG; assert returned tensor shape `[1, 1024, 1024, 1]`, dtype float32, values in `[0, 1]`.
2. `test_load_pore_density_caches` — call twice; assert same tensor object returned (`is` identity).
3. `test_load_pore_density_missing_triggers_warning_and_fallback` — remove the asset path; assert `log.warning` is called exactly once and the returned tensor is uniformly `1.0` with shape `[1, 1024, 1024, 1]`.
4. `test_load_translucency_template_returns_correct_shape` — monkeypatch `assets/flame_translucency_template.png` to a synthetic 1024² RGB PNG; assert returned tensor shape `[1, 1024, 1024, 3]`, dtype float32, values in `[0, 1]`.
5. `test_load_translucency_template_caches` — call twice; assert same tensor object returned (`is` identity).
6. `test_load_translucency_template_missing_raises` — remove the asset path; assert `RuntimeError` naming the exact path.

### `tests/test_bake_flame_uv_script.py`

Covers the `scripts/bake_flame_uv.py` authoring script that produces the checked-in `assets/flame_uv.npz`.

1. `test_script_importable` — `import scripts.bake_flame_uv` (or `importlib.util.spec_from_file_location`) raises no error.
2. `test_happy_path_produces_correct_npz` — point the script at `conftest.py`'s `synthetic_flame_pkl` fixture; run against a tmpdir output path; assert the resulting npz contains keys `uv_coords`, `faces_uv`, `bary_lut`; assert `uv_coords.shape == (5023, 2)`; assert `faces_uv.shape == (9976, 3)`; assert `bary_lut.shape == (1024, 1024, 6)`.
3. `test_eyeball_range_assertion` — the happy-path run must print and assert an eyeball vertex range; assert the sentinel string `"eyeball"` appears in stdout or the script's logged output.

### `tests/test_intrinsic_backends_procedural.py`

Uses a synthetic 5023-vertex sphere-ish mesh (generated procedurally from icospheres or FLAME-shaped random verts) plus a synthetic `[1, 512, 512, 3]` albedo UV. Monkeypatches `load_flame_uvs` to return a synthetic LUT.

1. `test_ao_output_shape_and_range` — AO output is `[1, 1024, 1024, 1]`, values in `[0, 1]`.
2. `test_ao_curvature_scale_default` — bake on a sphere mesh with `curvature_scale=4.0`; assert mean AO in `[0.85, 1.0]` (sphere has low absolute curvature).
3. `test_specular_output_shape_and_range` — specular output is `[1, 1024, 1024, 3]`, values in `[0, 1]`.
4. `test_specular_fallback_no_pore_mask` — remove `pore_density_mask` from `aux`; assert no exception, assert warning logged once.
5. `test_translucency_output_shape_and_range` — translucency output `[1, 1024, 1024, 3]`, values in `[0, 1]`.
6. `test_translucency_rgb_not_grayscale` — verify that returned tensor is not identical across all three channels on a non-uniform albedo input.
7. `test_normal_output_shape_and_range` — normal output `[1, 1024, 1024, 3]`, values in `[0, 1]`.
8. `test_normal_degenerate_uv_fallback` — inject a degenerate UV triangle (`dUV1 == dUV2`); assert output texel is `(0.5, 0.5, 1.0)` and warning logged once.

### `tests/test_intrinsic_maps_bake.py`

Monkeypatches `get_backend` to return a dummy that returns zero-valued maps of the correct shape.

1. `test_point_cloud_mesh_rejected` — pass mesh with `faces.numel() == 0`; assert `RuntimeError` with descriptive message.
2. `test_batch_gt_1_rejected` — pass `albedo_uv.shape[0] == 2`; assert `RuntimeError`.
3. `test_procedural_backend_dispatched` — assert `get_backend("procedural")` is called.
4. `test_output_is_intrinsic_maps_dict` — output passes `validate_intrinsic_maps` without error.
5. `test_resolution_512_downsamples` — assert all map shapes end with `(512, 512, C)` when `output_resolution="512"`.
6. `test_resolution_1024_default` — assert maps are `(1024, 1024, C)` with default args.
7. `test_strength_sliders_scale_output` — set `ao_strength=0.5`; assert AO output max <= 0.5 (dummy backend returns uniform 1.0, slider halves it).
8. `test_translucency_strength_scales_output` — set `translucency_strength=0.5`; assert translucency output max <= 0.5 (dummy backend returns uniform 1.0).
9. `test_seed_minus_1_does_not_raise` — `seed=-1` resolves without error.

### `tests/test_intrinsic_backends_stubs.py`

1. `test_mosar_stub_raises_not_implemented` — `MosarStubBackend().bake(...)` raises `NotImplementedError`.
2. `test_mosar_stub_error_names_model_path` — assert `"ComfyUI/models/mosar/"` in the exception message.
3. `test_mosar_stub_error_acknowledges_stub_nature` — assert the string `"stub"` or `"not yet publicly available"` in the error message (confirms the rewritten error from R14).
4. `test_learned_stub_raises_not_implemented` — `LearnedStubBackend().bake(...)` raises `NotImplementedError`.
5. `test_backends_registry_has_all_three` — `_BACKENDS` has keys `"procedural"`, `"mosar"`, `"learned"`.
6. `test_get_backend_unknown_raises` — `get_backend("nonexistent")` raises `RuntimeError` naming valid options.

### `tests/test_load_intrinsics_mosar.py`

1. `test_gate_false_raises` — `execute(i_understand_non_commercial_no_derivatives=False)` raises `RuntimeError`.
2. `test_gate_true_raises_not_implemented` — gate=True still raises `NotImplementedError` (weights absent).
3. `test_error_names_mosar_path` — error message contains `"ComfyUI/models/mosar/"`.
4. `test_error_names_tracking_url` — error message contains the FFHQ-UV-Intrinsics GitHub URL.
5. `test_gate_field_name_distinct` — verify the gate input name is `"i_understand_non_commercial_no_derivatives"`, not `"i_understand_non_commercial"`.

---

## Brutal Review

**1. CC BY-NC-ND "No Derivatives" clause is the highest legal risk in the pipeline.**
The FFHQ-UV-Intrinsics dataset (CC BY-NC-ND 4.0) may prohibit generating any derivative material — including running trained model inference to produce UV maps that were learned from the dataset. Even if no FFHQ-UV-Intrinsics images are directly visible at inference time, a trained model's outputs may be considered a derivative under ND. This is an open legal question; the Ubisoft La Forge team has not published a public FAQ on this. Path (a) (training on the dataset) is almost certainly affected; path (b) (procedural backend) is unaffected.
**Mitigation:** Gate is labeled `i_understand_non_commercial_no_derivatives`, not the generic NC gate. Tooltip names the ND clause in plain language. `assets/MOSAR_NOTICE.txt` (M2 blocker) will carry the full legal text. Do not activate the MoSAR backend without explicit legal review.

**2. Quality gap between procedural and learned baselines is large.**
Procedural specular is curvature × luminance inverse — a reasonable heuristic but blind to specular lobes, roughness variation, or subsurface depth. On dark-skinned subjects or subjects with facial hair, the curvature signal conflates hair-shadow with skin shininess. On very smooth young skin, the curvature signal is near-zero and specular collapses to albedo luminance only. The hardcoded specular tint `(0.94, 0.92, 0.89)` produces incorrect specular colour on dark skin; a `specular_tint` node input is deferred to v0.2 (see §Scope).
**Mitigation:** Expose `specular_strength` and `ao_strength` sliders so artists can modulate the output. Document the heuristic clearly in node tooltips. Treat procedural maps as a working default, not ground truth. The pluggable backend slot and the v0.2 `specular_tint` input are the long-term fixes.

**3. `flame_uv.npz` is a derived artifact from a gated asset.**
The `flame_uv.npz` coordinate cache is derived from `generic_model.pkl` (FLAME 2020, MPI non-commercial, email-gated). Its distribution as a repo asset may require that anyone who receives it also agrees to the FLAME terms. Shipping it pre-baked in the repo is convenient but potentially problematic for forks.
**Mitigation:** Include a note in `assets/README.md` that `flame_uv.npz` is derived from the FLAME 2020 generic model (MPI non-commercial) and that using it implies acceptance of the FLAME license terms. Provide `scripts/bake_flame_uv.py` as an alternative for users who prefer to derive it themselves.

**4. Eyeball vertex range must be confirmed against the actual FLAME 2020 pkl.**
The range `[3931, 5023)` is sourced from the FLAME-Universe documentation and community usage, but has not been confirmed by running `np.where(model.f > 3930)` on the actual `generic_model.pkl` in this repo's context. An off-by-one here would bleed eyeball UV corners into adjacent facial skin regions in every map.
**Mitigation:** `scripts/bake_flame_uv.py` must print and assert the eyeball face range before generating the npz — the assert output is the single source of truth for the `lo`/`hi` defaults in `make_eyeball_face_mask`. The `TODO(research)` tag in `flame_uv_data.py`'s function body must be resolved before `assets/flame_uv.npz` is authored. Resolving this is an **M0 exit criterion** (owner: whoever runs `bake_flame_uv.py` against the real pkl first). `test_flame_uv_data.py::test_make_eyeball_face_mask_excludes_upper_bound` pins the half-open interval in the test suite.

**5. HiFi3D ↔ FLAME UV remap is non-trivial if MoSAR or FFHQ-UV-Intrinsics ever lands.**
The roadmap says "retarget to FLAME UV inside the node" for MoSAR. HiFi3D has a different vertex count and UV layout. The retarget requires a mapping that the FFHQ-UV dataset provides via `run_flame_apply_hifi3d_uv.sh` — but this is a shell script wrapping a Python tool, not a library call. Integrating it as a node input transform would require significant engineering not scoped here.
**Mitigation:** Keep the MoSAR backend as a stub. Do not spec the retarget until MoSAR releases inference code. When the stub is activated, the remap is a first-class engineering task, not an afterthought.

**6. Procedural AO is a cavity proxy, not a real AO bake.**
The cotangent-Laplacian curvature signal approximates surface concavity but does not account for self-shadowing by distant geometry (e.g., the nose casting a shadow on the cheek at oblique angles). A real AO bake via ray-casting would require `pytorch3d` (opt-in dep) or a reference pose render. The proxy reads well in creases (nasolabial folds, eye corners) but will miss large-scale AO from head shape.
**Mitigation:** Name the limitation in the node description: "AO is a curvature-derived proxy; large-scale self-shadowing is not captured." Keep `pytorch3d` optional (matching `MeshPreview` convention); a future `ao_mode` input could select between `curvature` and `raycast` when `pytorch3d` is available.

**7. Scope creep invitation from stubbed backends.**
Every `NotImplementedError` stub is a public surface that users will file issues against. "When does MoSAR support land?" will generate a long GitHub issue thread. Stub messages must be self-explanatory and point at the upstream tracking issue.
**Mitigation:** Stub error messages name the upstream dataset repo URL and the specific constraint (no public weights). Do not promise a timeline. Close issue reports with a redirect to the upstream ubisoft-laforge tracker.

**8. `output_resolution = "1024"` as a string in a Combo is fragile.**
Combo inputs return strings. Downstream code that does `int(output_resolution)` is safe, but if `output_resolution` is somehow `None` (optional field not provided), `int(None)` raises `TypeError`, not a descriptive error.
**Mitigation:** `execute()` normalizes `output_resolution` to int at the top of the function body: `res = int(output_resolution or "1024")`. Test `test_intrinsic_maps_bake.py::test_resolution_1024_default` uses the default path, catching the common case.

**9. Translucency template must not ship with inaccurate pigmentation values.**
The per-region RGB scatter values in the spec are derived from literature (e.g., Dib et al. MoSAR paper, Barré-Brisebois & Hill, Jensen et al. 2001 skin scattering measurements) but have not been validated against FLAME topology on a real render. Wrong values produce visually wrong subsurface looks in physically-based renderers.
**Mitigation:** The template is an M1 blocker — it must not ship as a placeholder. The artist who authors it must validate against a PBR renderer (Marmoset Toolbag, Arnold, or equivalent) before M1 exit. The spec values here are starting points.

**10. `normal_strength` XY-only scaling does not renormalize correctly at extreme values.**
Multiplying XY by `normal_strength < 1.0` flattens the normal toward `(0, 0, 1)` (correct). But multiplying by `normal_strength > 1.0` amplifies XY without redistributing Z, producing a non-unit vector before renormalization. The division by `max(norm, 1e-8)` corrects it, but for very large `normal_strength` on steep surface angles, Z can approach 0 and the surface will appear pathologically curved.
**Mitigation:** Z is clamped to `>= 0.1` in the canonical algorithm block in §Normal Map (after renormalization, before `[0, 1]` encoding) — this prevents hemisphere flip at the `1.5` max slider value. Document the `1.5` max as a safe upper bound for FLAME geometry. `test_intrinsic_backends_procedural.py::test_normal_output_shape_and_range` asserts values in `[0, 1]`, catching the zero-Z case.

**11. `device` parameter on `bake()` is ignored by the procedural backend — future inconsistency.**
The `device` kwarg on the `IntrinsicsBackend.bake()` protocol is preserved for learned backends that need GPU tensors. The procedural backend ignores it and runs on CPU. If a user with a tight GPU budget sets `device="cpu"` on a future learned backend via the node, the protocol must pass that intent through correctly.
**Mitigation:** Document in `__init__.py` that `device` is a hint, not a guarantee, and that the procedural backend always runs on CPU. When the learned backend is implemented, it must move tensors to `device` at the start of `bake()` and back to CPU before returning.

**12. `'learned'` backend Combo option raises `NotImplementedError` on selection — retained as stable API surface.**
The `'learned'` Combo option is a permanent dead option in v0.1 (Q7 resolution: no candidate model selected). Exposing it creates user confusion. However, removing it would break any saved workflow that references the name when a candidate eventually lands.
**Mitigation:** Retain `'learned'` in the Combo and in `_BACKENDS` as a stable API surface; the tooltip documents it explicitly as "no candidate selected for v0.1." The stub error message names this limitation plainly. Cross-reference: Q7 resolution.

---

## Open Questions

**Q1 — Three-path decision.**
**Resolution:** Path (b) — procedural-first with pluggable backend slots. Justified by: (1) MoSAR has no released weights, (2) the procedural backend has standalone value for geometry-driven AO and normals, (3) path (a) has unresolved ND-clause legal risk. Decision is final for v0.1.

**Q2 — Output wire shape.**
**Resolution:** Single `INTRINSIC_MAPS` custom dict type declared in `nodes/intrinsic_maps_wire.py`. Matches the `FLAME_PARAMS` precedent; composes cleanly with the future Export node. Five-wire variant would require the Export node to accept five separate optional inputs; custom type defers that complexity.

**Q3 — Output resolution default.**
**Resolution:** `"1024"` default, with `["512", "1024"]` Combo options. The `"2048"` option from the skeleton is dropped from v0.1 to avoid 4× memory overhead (four 2048² float32 maps ≈ 256 MB) and because normals at 2K require a separate quality review. `"2048"` is a v0.2 candidate. See §Scope for the explicit scope note.

**Q4 — Displacement in Layer 2 or Layer 3.**
**Resolution:** Layer 3 (DECA) owns displacement. No `displacement` key in `INTRINSIC_MAPS` v0.1. DECA's expression-specific displacement is a strict superset of anything Layer 2 could derive from curvature alone; adding a coarse neutral displacement here creates a double-counting problem at compositing time. If the user reverses this before M1, add `"displacement"` to `CANONICAL_MAP_SHAPES` with shape `(1,)` (single-channel EXR-range value) and update `flame_uv.npz` authoring to include a displacement-compatible LUT.

**Q5 — Pore-density mask and translucency template authoring.**
**Resolution:** Both are hand-painted in Substance 3D Painter or Mari on the FLAME UV template. Owner: @melisdiary, deadline: before M1 exit. The spec provides authoritative per-region RGB values for the translucency template and defines the greyscale range for the pore-density mask. Both are M1 blockers; M0 ships with fallbacks.

**Q6 — Multi-GPU scheduling.**
**Resolution:** Procedural backend makes GPU scheduling moot for v0.1. The `device` parameter is preserved on the `IntrinsicsBackend.bake()` protocol so learned backends can target a specific GPU without a node-surface change. No multi-GPU scheduling logic in v0.1.

**Q7 — Learned backend identity.**
**Resolution:** Plain `NotImplementedError` stub with no candidate model specified. AvatarMe++ and NextFace are FLAME-compatible candidates but lack clean public checkpoints for FLAME-native UV inference. The `'learned'` key is retained as a stable API surface — see Brutal Review #12.

**Q8 — Texturing.xyz pore tiles in-repo vs. external.**
**Resolution:** External acquisition only; Layer 4 problem entirely. Not shipped in v0.1. `assets/pore_tiles/` path from the roadmap is a future placeholder.

**Q9 — Is albedo input required for the procedural backend?**
**Resolution:** Yes, albedo is used — specular tint modulation uses `lum_vertex` from the albedo UV, and translucency uses it as a luminance gate. Removing albedo from the node signature would simplify M0 but break M1 specular quality. Keep as a required input.

**Q10 — Derive `flame_uv.npz` at runtime or pre-bake?**
**Resolution:** Pre-baked as a repo asset. Deriving at runtime requires the FLAME pkl and chumpy shims, pulling `_install_chumpy_shims()` into all procedural-backend tests that currently run weight-free. Pre-baked keeps the M0 test suite independent of the FLAME pkl. See §FLAME UV Helper for the full rationale and the `load_flame_uvs()` implementation.

---

## Milestones

### M0 — Scaffold + procedural stub + passing tests

- `IntrinsicsBackend` protocol, `_BACKENDS` registry, `ProceduralBackend`, `MosarStubBackend`, `LearnedStubBackend` all implemented.
- `IntrinsicMapsBake` registered, point-cloud rejection tested, backend dispatch tested.
- `LoadIntrinsicsMoSAR` registered; gate and stub behavior tested.
- `nodes/flame_uv_data.py` implemented; `assets/flame_uv.npz` authored and checked in.
- **M0 exit criterion: eyeball vertex range** confirmed — `scripts/bake_flame_uv.py` must print and assert the range before the npz is committed; the `TODO(research)` in `make_eyeball_face_mask` must be resolved with the confirmed values.
- Procedural backend returns maps of correct shape — AO and normals use real geometry signals; specular and translucency may use simplified placeholders.
- Full pytest suite passes; new tests (`test_intrinsic_maps_bake.py`, `test_intrinsic_maps_wire.py`, `test_flame_uv_data.py`, `test_intrinsic_assets.py`, `test_bake_flame_uv_script.py`, `test_intrinsic_backends_procedural.py`, `test_intrinsic_backends_stubs.py`, `test_load_intrinsics_mosar.py`) green.
- No real weights required.

**M0 exit criteria:** `~/github/ComfyUI/venv311/bin/python -m pytest tests/` passes with no weights. `IntrinsicMapsBake` node appears in the ComfyUI node graph with correct inputs/outputs. Eyeball range confirmed and recorded.

### M1 — Procedural signal quality + canonical assets

- `assets/flame_pore_density_uv.png` authored (hand-painted, validated in a PBR renderer).
- `assets/flame_translucency_template.png` authored (per-region RGB scatter confirmed against literature; validated in a PBR renderer).
- Specular map integrates pore-density mask (no longer uniform `1.0` fallback).
- Translucency map integrates authored template (not placeholder).
- Normal map correctly outputs OpenGL tangent-space TBN from FLAME geometry.
- Visual smoke test on at least 3 reference portraits: one front-facing, one 3/4 profile, one non-Caucasian subject. Maps reviewed for plausibility against reference intrinsic decompositions.
- Q4 displacement ownership confirmed final (Layer 3) before M1 exit.

**M1 exit criteria:** Visual review passes; `assets/flame_pore_density_uv.png` and `assets/flame_translucency_template.png` checked in; no M1 blocker assets missing.

### M2 — Backend plug-in readiness

- `LoadIntrinsicsMoSAR` wires to the `MOSAR_MODEL` descriptor consumed by `mosar_stub`; the surface is ready for a real implementation.
- `assets/MOSAR_NOTICE.txt` authored with full CC BY-NC-ND 4.0 license text and ND-clause plain-language explanation.
- When MoSAR weights are released publicly, only the `mosar_stub.py` body needs replacement — the node surface, wire type, gate, and file layout are already in place.

**M2 exit criteria:** `LoadIntrinsicsMoSAR` + `mosar_stub` wiring complete; `assets/MOSAR_NOTICE.txt` present. With real MoSAR weights (when available): `LoadIntrinsicsMoSAR.execute(i_understand_non_commercial_no_derivatives=True)` returns a non-stub descriptor; `IntrinsicMapsBake(backend="mosar")` produces `INTRINSIC_MAPS` with MoSAR-predicted maps.

---

## Verification

1. **Import smoke.** `python -c "from nodes.intrinsic_maps_bake import IntrinsicMapsBake"` succeeds with no weights installed.
2. **Registration.** `IntrinsicMapsBake` and `LoadIntrinsicsMoSAR` appear in ComfyUI's node graph; schemas show correct inputs and outputs.
3. **Point-cloud rejection.** Feeding a mesh with `faces.numel() == 0` to `IntrinsicMapsBake` raises `RuntimeError` with text mentioning `num_sampling=5023`.
4. **Procedural bake shape.** On a synthetic 5023-vertex mesh with a `[1, 512, 512, 3]` albedo UV, `IntrinsicMapsBake.execute(backend="procedural")` returns an `INTRINSIC_MAPS` dict; all four maps are `[1, 1024, 1024, C]` float32 in `[0, 1]`.
5. **Resolution downsampling.** `output_resolution="512"` produces maps of shape `[1, 512, 512, C]`.
6. **Strength sliders.** `ao_strength=0.0` produces a uniformly white (1.0) AO map (no shadowing). `normal_strength=0.0` produces a flat normal `(0.5, 0.5, 1.0)` everywhere. `translucency_strength=0.0` produces a zero translucency map.
7. **Stub rejection.** `IntrinsicMapsBake.execute(backend="mosar")` raises `NotImplementedError` naming `ComfyUI/models/mosar/`.
8. **NC/ND gate.** `LoadIntrinsicsMoSAR.execute(i_understand_non_commercial_no_derivatives=False)` raises `RuntimeError`; `True` raises `NotImplementedError` (weights absent).
9. **Eyeball zeroing.** `IntrinsicMapsBake` with the procedural backend produces zero-valued texels at the eyeball UV region in all four maps (confirmed by sampling the bary_lut at eyeball face indices).
10. **Full pytest suite passes; new tests green.** `~/github/ComfyUI/venv311/bin/python -m pytest tests/` — all existing tests pass; new intrinsic-maps tests pass: `test_intrinsic_maps_bake.py`, `test_intrinsic_maps_wire.py`, `test_flame_uv_data.py`, `test_intrinsic_assets.py`, `test_bake_flame_uv_script.py`, `test_intrinsic_backends_procedural.py`, `test_intrinsic_backends_stubs.py`, `test_load_intrinsics_mosar.py`. No weight files required.
11. **M2 verification (with real MoSAR weights, when available).** `LoadIntrinsicsMoSAR.execute(i_understand_non_commercial_no_derivatives=True)` returns a descriptor; `IntrinsicMapsBake(backend="mosar", mosar_model=descriptor)` produces `INTRINSIC_MAPS` with learned maps.

---

## References

- MoSAR — Dib et al., *MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Implicit Differentiable Renderer*, CVPR 2024.
- FFHQ-UV-Intrinsics — Ubisoft La Forge, `github.com/ubisoft/ubisoft-laforge-FFHQ-UV-Intrinsics`. CC BY-NC-ND 4.0.
- AvatarMe++ — Gecer et al., *AvatarMe++: Facial Shape and BRDF Inference with Photorealistic Rendering-Aware GANs*, IEEE TPAMI 2021.
- Barré-Brisebois & Hill, *Blending in Detail* (Reoriented Normal Mapping), GDC 2012.
- NextFace — Dib et al., *Practical Face Reconstruction via Differentiable Ray Tracing*, Eurographics 2021.
- DECA — Feng et al., *Learning an Animatable Detailed 3D Face Model from In-the-Wild Images*, SIGGRAPH 2021, ACM TOG 40(4).
- FLAME-Universe — `github.com/TimoBolkart/FLAME-Universe`. Canonical index for FLAME topology, UV layout, and eyeball vertex range.
- Jensen et al., *A Practical Model for Subsurface Light Transport*, SIGGRAPH 2001. (Source for per-region scatter colour priors in the translucency template.)
- KaoLRM (3DV 2026) — FLAME regression via LRM triplane priors.
- FreeUV (CVPR 2025) — ground-truth-free UV recovery via SD1.5 + Cross-Assembly.
