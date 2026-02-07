#!/usr/bin/env python3
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

# ログ設定：本番ではファイルへ出力
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Material:
    name: str
    mass: float
    GWP: float
    recyclability: float
    crit: float
    tox: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Weighting:
    alpha: float = 1.0
    beta: float = 1.5
    gamma: float = 2.0
    delta: float = 1.0
    eta: float = 1.0


@dataclass(slots=True)
class Reference:
    mass: float = 1.0
    GWP: float = 1.0


class MaterialRanker:
    def __init__(
        self,
        weights: Weighting = Weighting(),
        reference: Reference = Reference(),
    ) -> None:
        self.weights = weights
        self.ref = reference

    def score(self, mat: Material) -> float:
        """単一材料の環境負荷スコアを算出（小さいほど優位）"""
        w = self.weights
        r = self.ref
        m, G, R, C, T = mat.mass, mat.GWP, mat.recyclability, mat.crit, mat.tox
        return (
            w.alpha * (m / r.mass)
            + w.beta * (G * m / (r.GWP * r.mass))
            - w.gamma * R
            + w.delta * C
            + w.eta * T
        )

    def rank(self, materials: List[Material]) -> List[Material]:
        """スコア昇順でソート"""
        return sorted(materials, key=self.score)


def load_materials(path: Path) -> List[Material]:
    """JSONファイルから材料リストを読み込む"""
    if not path.exists():
        logger.warning("材料DBが見つからないためサンプルデータを使用")
        return _sample_materials()

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [Material(**item) for item in data]


def _sample_materials() -> List[Material]:
    """サンプルデータ（本番では外部DBへ移行）"""
    return [
        Material("Al7075", 0.8, 8.1, 0.9, 0.2, 0.1),
        Material("CF_recycled", 0.5, 45.0, 0.4, 0.6, 0.2),
        Material("Stainless304", 1.2, 6.0, 0.85, 0.3, 0.15),
    ]


def main() -> None:
    db_path = Path("materials.json")
    materials = load_materials(db_path)

    ranker = MaterialRanker()
    ranked = ranker.rank(materials)

    for mat in ranked:
        logger.info(f"{mat.name} score={ranker.score(mat):.3f}")


if __name__ == "__main__":
    main()