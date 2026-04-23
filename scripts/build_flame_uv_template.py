"""Generate `assets/flame_uv_template.npz` from FLAME 2020 `head_template.obj`.

FLAME's pkl carries geometry (verts, faces, shapedirs, ...) but not UV
coordinates. The UV layout lives in `head_template.obj` on the MPI release
page. This script repackages `vt` + `ft` (indices paired with `ft`) into an
NPZ the `FLAMEProjectToUV` node consumes.

Usage:
    python scripts/build_flame_uv_template.py /path/to/head_template.obj

License: the resulting NPZ inherits MPI non-commercial terms. Do not
redistribute unless that has been cleared separately (Outstanding Blocker B1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_obj_uv(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vt [V_uv, 2] float32, ft [F, 3] int64).

    `ft` is 0-indexed (OBJ indices are 1-indexed, converted on read).
    """
    vt_list: list[tuple[float, float]] = []
    ft_list: list[tuple[int, int, int]] = []

    with obj_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("vt "):
                parts = line.split()
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
                vt_list.append((u, v))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise RuntimeError(
                        f"Only triangulated faces are supported; got {len(tokens)}-gon "
                        f"in {obj_path.name}: {line.strip()!r}"
                    )
                uvs = []
                for tok in tokens:
                    parts = tok.split("/")
                    if len(parts) < 2 or not parts[1]:
                        raise RuntimeError(
                            f"Face token {tok!r} has no vt index; "
                            "ensure the OBJ was exported with UVs."
                        )
                    uvs.append(int(parts[1]) - 1)
                ft_list.append((uvs[0], uvs[1], uvs[2]))

    if not vt_list:
        raise RuntimeError(f"No `vt` entries found in {obj_path}.")
    if not ft_list:
        raise RuntimeError(f"No face/vt rows found in {obj_path}.")

    vt = np.asarray(vt_list, dtype=np.float32)
    ft = np.asarray(ft_list, dtype=np.int64)
    return vt, ft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj_path", type=Path, help="Path to head_template.obj")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "flame_uv_template.npz",
        help="Output NPZ path (default: assets/flame_uv_template.npz)",
    )
    args = parser.parse_args()

    if not args.obj_path.is_file():
        print(f"error: {args.obj_path} does not exist", file=sys.stderr)
        return 2

    vt, ft = parse_obj_uv(args.obj_path)
    print(f"Parsed vt={vt.shape}, ft={ft.shape}")
    print(f"  vt range u=[{vt[:,0].min():.4f}, {vt[:,0].max():.4f}] "
          f"v=[{vt[:,1].min():.4f}, {vt[:,1].max():.4f}]")
    print(f"  ft max index = {int(ft.max())}, V_uv = {vt.shape[0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, vt=vt, ft=ft)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
