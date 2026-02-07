#!/usr/bin/env python3
"""
URDF inertial tag generator from STL mesh.
STLメッシュからURDF用inertialタグを生成する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load triangle mesh from file."""
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    mesh = trimesh.load(str(path))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded geometry is not a single triangle mesh")
    return mesh


def compute_inertial_params(
    mesh: trimesh.Trimesh,
    density: float,
    origin_offset: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    質量・重心・慣性テンソルを計算
    """
    mass = float(mesh.volume * density)
    com = mesh.center_mass  # mesh座標系
    inertia_com = mesh.moment_inertia * density  # 重心周り

    # 平行軸定理で原点へ変換
    d = origin_offset - com
    I_o = inertia_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mass, com, I_o


def build_inertial_tag(mass: float, com: np.ndarray, inertia: np.ndarray) -> str:
    """URDF <inertial> タグ文字列を生成"""
    return f"""<inertial>
  <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>
  <mass value="{mass:.6f}"/>
  <inertia ixx="{inertia[0,0]:.6f}" ixy="{inertia[0,1]:.6f}" ixz="{inertia[0,2]:.6f}"
           iyy="{inertia[1,1]:.6f}" iyz="{inertia[1,2]:.6f}" izz="{inertia[2,2]:.6f}"/>
</inertial>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate URDF <inertial> from STL")
    parser.add_argument("mesh", type=Path, help="Path to STL file")
    parser.add_argument(
        "--density",
        type=float,
        default=2700.0,
        help="Material density [kg/m^3] (default: 2700 for aluminum)",
    )
    parser.add_argument(
        "--origin-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Offset from mesh origin to link origin [m]",
    )
    args = parser.parse_args(argv)

    mesh = load_mesh(args.mesh)
    mass, com, inertia = compute_inertial_params(
        mesh, args.density, np.array(args.origin_offset)
    )
    print(build_inertial_tag(mass, com, inertia))


if __name__ == "__main__":
    main()