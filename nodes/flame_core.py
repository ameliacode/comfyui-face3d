"""FLAME forward pass — shape+expression blendshapes + LBS skinning.

Adapted from a working FLAME 2020 implementation; targets FLAME 2020 and 2023
Open pickles (same key schema: v_template, shapedirs, posedirs, J_regressor,
weights, kintree_table, f). Process-level (gender, device) cache lets the
editor HTTP route amortize pickle-load cost across requests.
"""
from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

N_VERTICES = 5023
N_FACES = 9976
N_JOINTS = 5
POSE_DIM = N_JOINTS * 3          # 15

REQUIRED_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")


def _install_chumpy_shims() -> None:
    """chumpy predates python 3.11 and numpy 1.24 — apply minimal compat shims."""
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # chumpy does `from numpy import bool, int, float, complex, object, unicode, str, nan, inf`
    # which fails on numpy>=1.20. Patch np with the legacy aliases (Python builtins).
    for name, value in (
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("unicode", str), ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)
    if "nan" not in np.__dict__:
        np.nan = float("nan")
    if "inf" not in np.__dict__:
        np.inf = float("inf")


def _load_flame_pkl(pkl_path: Path) -> dict:
    # FLAME pickles routinely reference chumpy; install compat shims first,
    # then try to import chumpy. If chumpy isn't installed we still attempt
    # the load — newer pre-converted pickles may not need it.
    _install_chumpy_shims()
    try:
        import chumpy  # noqa: F401
    except Exception:
        pass
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"FLAME pickle requires chumpy: {e}. Install with `pip install chumpy` "
            "or pre-convert the pickle."
        ) from e

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise RuntimeError(
            f"FLAME pickle {pkl_path} missing required keys: {missing}. "
            f"Present: {sorted(data.keys())}"
        )

    kintree = np.asarray(data["kintree_table"])
    if kintree.shape[1] != N_JOINTS:
        raise RuntimeError(
            f"FLAME pickle reports {kintree.shape[1]} joints, expected {N_JOINTS} "
            "(global, neck, jaw, eye_L, eye_R). A 'no_jaw' variant is not supported."
        )
    weights = np.asarray(data["weights"])
    if weights.shape[1] != N_JOINTS:
        raise RuntimeError(
            f"FLAME skinning weights have {weights.shape[1]} joints, expected {N_JOINTS}."
        )
    return data


class FlameCore:
    """Buffers + forward pass. Read-only after construction; safe to share."""

    def __init__(self, pkl_path: str | Path, device: str | torch.device = "cpu"):
        pkl_path = Path(pkl_path)
        raw = _load_flame_pkl(pkl_path)

        def _arr(x) -> np.ndarray:
            if hasattr(x, "toarray"):
                x = x.toarray()
            return np.asarray(x, dtype=np.float32)

        v_template = torch.from_numpy(_arr(raw["v_template"]))
        shapedirs  = torch.from_numpy(_arr(raw["shapedirs"]))
        posedirs   = torch.from_numpy(_arr(raw["posedirs"]))
        J_reg      = torch.from_numpy(_arr(raw["J_regressor"]))
        lbs_w      = torch.from_numpy(_arr(raw["weights"]))
        faces      = torch.from_numpy(np.asarray(raw["f"], dtype=np.int64))
        parents    = np.asarray(raw["kintree_table"][0], dtype=np.int64).copy()
        parents[0] = -1
        parents_t  = torch.from_numpy(parents)

        self.device = torch.device(device)
        self.v_template = v_template.to(self.device)          # [V, 3]
        self.shapedirs  = shapedirs.to(self.device)            # [V, 3, Ns+Ne]
        self.posedirs   = posedirs.to(self.device)             # [V*3, (J-1)*9]
        self.J_regressor = J_reg.to(self.device)               # [J, V]
        self.lbs_weights = lbs_w.to(self.device)               # [V, J]
        self.faces   = faces.to(self.device)                   # [F, 3]
        self.parents = parents_t.to(self.device)               # [J]

        self.n_vertices = int(self.v_template.shape[0])
        self.n_faces = int(self.faces.shape[0])
        self.max_shape_dim = int(self.shapedirs.shape[2]) - 100  # FLAME: last 100 are expr
        self.max_expr_dim = 100

        if self.n_vertices != N_VERTICES:
            log.warning("FLAME core has %d vertices (expected %d)", self.n_vertices, N_VERTICES)
        if self.n_faces != N_FACES:
            log.warning("FLAME core has %d faces (expected %d)", self.n_faces, N_FACES)

    @torch.no_grad()
    def forward(
        self,
        shape: torch.Tensor,   # [B, Ns]
        expr:  torch.Tensor,   # [B, Ne]
        pose:  torch.Tensor,   # [B, 15]
        trans: torch.Tensor,   # [B, 3]
    ) -> torch.Tensor:
        """Return posed vertices [B, V, 3] in FLAME object space."""
        B = shape.shape[0]
        dev = self.v_template.device
        shape = shape.to(dev)
        expr = expr.to(dev)
        pose = pose.to(dev)
        trans = trans.to(dev)

        Ns = shape.shape[1]
        Ne = expr.shape[1]
        # shapedirs concatenates shape (first 300) and expression (last 100) bases.
        sd_shape = self.shapedirs[:, :, :Ns]
        sd_expr  = self.shapedirs[:, :, self.max_shape_dim : self.max_shape_dim + Ne]

        # v_shaped: [B, V, 3]
        v_shaped = (
            self.v_template.unsqueeze(0)
            + torch.einsum("vxb,nb->nvx", sd_shape, shape)
            + torch.einsum("vxb,nb->nvx", sd_expr, expr)
        )

        # Joint locations per batch: [B, J, 3]
        J = torch.einsum("jv,nvx->njx", self.J_regressor, v_shaped)

        # Rotations: [B, J, 3, 3]
        rot_mats = _batch_rodrigues(pose.view(-1, 3)).view(B, N_JOINTS, 3, 3)

        # Pose correctives from non-root joints: [B, V, 3]
        # posedirs layout in FLAME pkl: [V, 3, (J-1)*9].
        ident = torch.eye(3, device=dev)
        pose_feature = (rot_mats[:, 1:] - ident).view(B, -1)          # [B, (J-1)*9]
        v_posed = v_shaped + torch.einsum("vxk,bk->bvx", self.posedirs, pose_feature)

        # LBS
        A = _batch_rigid_transform(rot_mats, J, self.parents)          # [B, J, 4, 4]
        T = torch.einsum("vj,njpq->nvpq", self.lbs_weights, A)         # [B, V, 4, 4]

        ones = torch.ones(B, self.n_vertices, 1, device=dev)
        v_h = torch.cat([v_posed, ones], dim=2)                        # [B, V, 4]
        v_out = torch.einsum("nvpq,nvq->nvp", T, v_h)                  # [B, V, 4]
        verts = v_out[..., :3] + trans.unsqueeze(1)
        return verts

    def compute_vertex_normals(self, verts: torch.Tensor) -> torch.Tensor:
        """verts [V, 3] → normals [V, 3]."""
        return _vertex_normals(verts, self.faces)


# ---------------------------------------------------------------------------
# LBS helpers

def _batch_rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """[N, 3] axis-angle → [N, 3, 3] rotation matrices."""
    angle = rvec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    axis = rvec / angle
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=rvec.device).unsqueeze(0)
    return I + sin * K + (1 - cos) * torch.bmm(K, K)


def _batch_rigid_transform(
    rot_mats: torch.Tensor,   # [B, J, 3, 3]
    joints:   torch.Tensor,   # [B, J, 3]
    parents:  torch.Tensor,   # [J] parents[0] == -1
) -> torch.Tensor:
    """Return per-joint transforms relative to rest pose, [B, J, 4, 4]."""
    B, J = rot_mats.shape[:2]
    device = rot_mats.device

    rel = joints.clone()
    rel[:, 1:] = rel[:, 1:] - joints[:, parents[1:]]

    def make_T(R, t):
        T = torch.zeros(B, 4, 4, device=device)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1.0
        return T

    A = []
    for j in range(J):
        local = make_T(rot_mats[:, j], rel[:, j])
        A.append(local if j == 0 else torch.bmm(A[int(parents[j].item())], local))
    A = torch.stack(A, dim=1)                                          # [B, J, 4, 4]

    inv_rest = torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, J, -1, -1).clone()
    inv_rest[:, :, :3, 3] = -joints

    return torch.bmm(A.view(B * J, 4, 4), inv_rest.view(B * J, 4, 4)).view(B, J, 4, 4)


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return F.normalize(vn, dim=1)


# ---------------------------------------------------------------------------
# Process-level cache

_CACHE: dict[tuple[str, str, str], FlameCore] = {}
_CACHE_LOCK = threading.Lock()


def get_flame_core(gender: str, pkl_path: str | Path, device: str | torch.device = "cpu") -> FlameCore:
    device = torch.device(device)
    key = (str(gender), str(Path(pkl_path)), str(device))
    with _CACHE_LOCK:
        core = _CACHE.get(key)
        if core is None:
            core = FlameCore(pkl_path, device=device)
            _CACHE[key] = core
        return core
