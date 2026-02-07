#!/usr/bin/env python3
"""
比剛性・比強度でヒューマノイド構造材料をランク付けするCLIツール
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# 材料データ型定義
@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/m^3
    young_modulus: float    # GPa
    yield_strength: float   # MPa

    @property
    def specific_stiffness(self) -> float:
        """比剛性 E/ρ [m^2/s^2]"""
        return self.young_modulus * 1e9 / self.density

    @property
    def specific_strength(self) -> float:
        """比強度 σ_y/ρ [m^2/s^2]"""
        return self.yield_strength * 1e6 / self.density


# デフォルト材料データベース
DEFAULT_MATERIALS: Tuple[Material, ...] = (
    Material("Steel_1045", 7850, 210.0, 530.0),
    Material("Al_7075",    2810,  71.7, 503.0),
    Material("Ti6Al4V",    4430, 114.0, 880.0),
    Material("CFRP",       1600, 120.0, 900.0),  # 異方性近似
)


def load_materials(path: Path | None = None) -> Dict[str, Material]:
    """JSONファイルから材料リストを読み込む（未指定時はデフォルトを返す）"""
    if path is None:
        return {m.name: m for m in DEFAULT_MATERIALS}

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {
        name: Material(name, d["density"], d["young_modulus"], d["yield_strength"])
        for name, d in data.items()
    }


def rank_materials(
    materials: Dict[str, Material],
    key: str = "stiffness",
    top_k: int | None = None,
) -> List[Tuple[str, Material, float, float]]:
    """指定キーで降順ソートし、top_k件返す"""
    key_func = (
        (lambda m: m.specific_stiffness) if key == "stiffness"
        else (lambda m: m.specific_strength)
    )
    ranked = sorted(
        materials.items(),
        key=lambda kv: key_func(kv[1]),
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(name, m, m.specific_stiffness, m.specific_strength) for name, m in ranked]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="構造材料の比剛性・比強度ランキング")
    parser.add_argument("-f", "--file", type=Path, help="材料定義JSONパス")
    parser.add_argument("-k", "--key", choices=["stiffness", "strength"], default="stiffness",
                        help="ランキング指標")
    parser.add_argument("-n", "--top", type=int, help="上位n件を表示")
    parser.add_argument("--json", action="store_true", help="結果をJSON形式で出力")
    args = parser.parse_args(argv)

    materials = load_materials(args.file)
    ranked = rank_materials(materials, key=args.key, top_k=args.top)

    if args.json:
        print(json.dumps([
            {
                "name": name,
                "specific_stiffness": s_stiff,
                "specific_strength": s_str,
            }
            for name, _, s_stiff, s_str in ranked
        ], ensure_ascii=False, indent=2))
    else:
        for name, _, s_stiff, s_str in ranked:
            print(f"{name}: specific_stiff={s_stiff:.2e}, specific_strength={s_str:.2e}")


if __name__ == "__main__":
    main()