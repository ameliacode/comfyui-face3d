from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def uv_template_stub(monkeypatch, tmp_path):
    """Replace `load_uv_template` with a 2-triangle synthetic layout.

    Geometry faces are `[[0,1,2], [0,2,3]]`. UV triangle 0 covers the bottom-
    left half of the unit square, UV triangle 1 covers the top-right half.
    """
    vt = torch.tensor(
        [
            [0.0, 0.0],  # vt0  ↔ v0
            [1.0, 0.0],  # vt1  ↔ v1
            [1.0, 1.0],  # vt2  ↔ v2
            [0.0, 1.0],  # vt3  ↔ v3
        ],
        dtype=torch.float32,
    )
    ft = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)

    from nodes import flame_project_to_uv as fptu

    monkeypatch.setattr(fptu, "load_uv_template", lambda *a, **kw: (vt, ft))
    return vt, ft


def _make_mesh(vertices: torch.Tensor, faces: torch.Tensor, *, topology: str = "mesh"):
    ns = types.SimpleNamespace(
        vertices=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        topology=topology,
        base_vertices=vertices.unsqueeze(0),
        base_faces=faces.unsqueeze(0),
    )
    return ns


def _solid_image(color: tuple[float, float, float], h: int = 224, w: int = 224) -> torch.Tensor:
    arr = torch.zeros(1, h, w, 3, dtype=torch.float32)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    return arr


def test_front_facing_quad_projects_solid_color(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    # Quad in z=0 plane. Camera at (0, 0, -d) looks toward +Z. For face
    # normal = cross(v1-v0, v2-v0) to point -Z (toward camera) the winding
    # order is (v0 bottom-left, v1 top-left, v2 top-right, v3 bottom-right).
    # Then view = cam_origin - ctr = (0, 0, -d); dot(view, n) = d > 0 (front).
    vertices = torch.tensor(
        [
            [-0.2, -0.2, 0.0],
            [-0.2,  0.2, 0.0],
            [ 0.2,  0.2, 0.0],
            [ 0.2, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = _make_mesh(vertices, faces)
    image = _solid_image((1.0, 0.5, 0.25))

    flaw_uv, mask = FLAMEProjectToUV.execute(
        mesh=mesh, image=image, source_cam_dist=2.0, uv_resolution=64,
    )
    assert flaw_uv.shape == (1, 64, 64, 3)
    assert mask.shape == (1, 64, 64)
    hit = mask[0] > 0
    assert hit.any(), "expected at least some UV texels to be filled"
    rgb = flaw_uv[0][hit]
    assert torch.allclose(rgb.mean(dim=0), torch.tensor([1.0, 0.5, 0.25]), atol=1e-3)


def test_back_face_marks_invisible(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    # Reverse winding so the face normal points +Z (away from camera at z=-d).
    vertices = torch.tensor(
        [
            [-0.2, -0.2, 0.0],
            [ 0.2, -0.2, 0.0],
            [ 0.2,  0.2, 0.0],
            [-0.2,  0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = _make_mesh(vertices, faces)
    image = _solid_image((1.0, 1.0, 1.0))

    flaw_uv, mask = FLAMEProjectToUV.execute(
        mesh=mesh, image=image, source_cam_dist=2.0, uv_resolution=64,
    )
    assert mask.sum() == 0.0
    assert flaw_uv.sum() == 0.0


def test_batch_gt_1_rejected(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    vertices = torch.tensor(
        [[-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.2, -0.2, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = _make_mesh(vertices, faces)
    image = torch.zeros(2, 224, 224, 3, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="batch size 1"):
        FLAMEProjectToUV.execute(mesh=mesh, image=image)


def test_image_resized_before_projection(uv_template_stub, caplog):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    vertices = torch.tensor(
        [[-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.2, -0.2, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = _make_mesh(vertices, faces)
    image = _solid_image((0.25, 0.5, 0.75), h=448, w=448)

    with caplog.at_level("INFO", logger="nodes.flame_project_to_uv"):
        flaw_uv, mask = FLAMEProjectToUV.execute(
            mesh=mesh, image=image, uv_resolution=64,
        )
    assert any("resizing input image" in rec.message for rec in caplog.records)
    assert flaw_uv.shape == (1, 64, 64, 3)


def test_output_dtype_and_device(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    vertices = torch.tensor(
        [[-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.2, -0.2, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = _make_mesh(vertices, faces)
    image = _solid_image((0.1, 0.2, 0.3))

    flaw_uv, mask = FLAMEProjectToUV.execute(
        mesh=mesh, image=image, uv_resolution=64,
    )
    assert flaw_uv.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert flaw_uv.device.type == "cpu"
    assert mask.device.type == "cpu"


def test_point_cloud_mesh_uses_base_attrs(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    vertices = torch.tensor(
        [[-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.2, -0.2, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    sampled_points = torch.randn(1000, 3)

    mesh = types.SimpleNamespace(
        vertices=sampled_points.unsqueeze(0),
        faces=torch.empty((1, 0, 3), dtype=torch.int64),
        topology="point_cloud",
        base_vertices=vertices.unsqueeze(0),
        base_faces=faces.unsqueeze(0),
    )
    image = _solid_image((1.0, 0.0, 0.0))

    flaw_uv, mask = FLAMEProjectToUV.execute(
        mesh=mesh, image=image, uv_resolution=64,
    )
    assert mask.sum() > 0


def test_point_cloud_mesh_without_base_raises(uv_template_stub):
    from nodes.flame_project_to_uv import FLAMEProjectToUV

    mesh = types.SimpleNamespace(
        vertices=torch.randn(1, 1000, 3),
        faces=torch.empty((1, 0, 3), dtype=torch.int64),
        topology="point_cloud",
        base_vertices=None,
        base_faces=None,
    )
    image = _solid_image((1.0, 1.0, 1.0))

    with pytest.raises(RuntimeError, match="base_vertices|base_faces"):
        FLAMEProjectToUV.execute(mesh=mesh, image=image)


def test_eyeball_triangle_mask_boundaries():
    from nodes.flame_project_to_uv import _eyeball_triangle_mask

    faces = torch.tensor(
        [
            [0, 1, 2],               # non-eyeball
            [3931, 4000, 5022],      # all eyeball
            [3930, 4000, 5022],      # one vert below range → not eyeball
            [3931, 4000, 5023],      # one vert above range → not eyeball
            [4500, 4600, 4700],      # all eyeball
        ],
        dtype=torch.int64,
    )
    mask = _eyeball_triangle_mask(faces)
    assert mask.tolist() == [False, True, False, False, True]


def test_uv_template_missing_raises(tmp_path):
    from nodes import flame_uv_template

    flame_uv_template._UV_CACHE.clear()
    missing = tmp_path / "missing.npz"

    with pytest.raises(RuntimeError) as exc:
        flame_uv_template.load_uv_template(missing)
    msg = str(exc.value)
    assert "missing.npz" in msg
    assert "flame.is.tue.mpg.de" in msg
    assert "build_flame_uv_template.py" in msg


def test_uv_template_malformed_raises(monkeypatch, tmp_path):
    from nodes import flame_uv_template

    flame_uv_template._UV_CACHE.clear()
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, vt=np.zeros((10, 2), dtype=np.float32))  # missing ft

    with pytest.raises(RuntimeError, match=r"\bft\b"):
        flame_uv_template.load_uv_template(bad_path)


def test_uv_template_caches(monkeypatch, tmp_path):
    from nodes import flame_uv_template

    flame_uv_template._UV_CACHE.clear()
    path = tmp_path / "template.npz"
    np.savez(path,
             vt=np.zeros((4, 2), dtype=np.float32),
             ft=np.zeros((2, 3), dtype=np.int64))

    vt1, ft1 = flame_uv_template.load_uv_template(path)
    vt2, ft2 = flame_uv_template.load_uv_template(path)
    assert vt1 is vt2
    assert ft1 is ft2


def test_camera_origin_at_neg_d():
    from nodes.flame_project_to_uv import _camera_origin

    origin = _camera_origin(2.0, torch.device("cpu"))
    assert torch.allclose(origin, torch.tensor([0.0, 0.0, -2.0]))


def test_camera_intrinsic_matches_f075_c05():
    from nodes.flame_project_to_uv import _intrinsic_matrix

    intrinsic = _intrinsic_matrix(torch.device("cpu"))
    expected = torch.tensor(
        [[0.75, 0.0, 0.5], [0.0, 0.75, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(intrinsic, expected)
