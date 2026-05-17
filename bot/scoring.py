"""Systeme de scoring par facteurs.

Philosophie : le bot voit TOUT, rapporte TOUT, et chaque element observe
devient un Factor avec un statut, un poids, et un detail.

4 filtres DURS (REJECT auto si l'un casse, peu importe le score) :
1. Killzone active a l'entree
2. OB forme dans une killzone
3. Sweep + BOS confirmes
4. HTF non contradictoire
5. RR >= 2

Tout le reste = score pondere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FactorStatus = Literal["present", "absent", "warning", "neutral"]
FactorCategory = Literal[
    "temporal", "structure", "liquidity", "ob_quality",
    "fvg", "zone", "volume", "entry", "htf",
]
Verdict = Literal["premium", "good", "mediocre", "weak", "reject"]


@dataclass
class Factor:
    """Un facteur observe sur le setup."""
    name: str                # cle technique stable: "killzone_active"
    label: str               # libelle UI: "Killzone active"
    category: FactorCategory
    status: FactorStatus
    weight: int              # +/- contribution si "present" (0 si absent/neutral selon logique)
    detail: str = ""         # texte humain pour la revue

    @property
    def applied_score(self) -> int:
        """Score effectivement applique au total selon le statut."""
        if self.status == "present":
            return self.weight
        if self.status == "warning":
            return -abs(self.weight) // 2  # mi-penalite
        return 0


@dataclass
class HardFilter:
    """Filtre dur : doit etre OK pour que le trade soit pris en live."""
    name: str
    label: str
    passed: bool
    detail: str = ""


@dataclass
class SetupAnalysis:
    """Snapshot complet d'un setup detecte.

    Stocke TOUT ce qui a ete observe, meme les absences.
    """
    hard_filters: list[HardFilter] = field(default_factory=list)
    factors: list[Factor] = field(default_factory=list)

    @property
    def all_hard_passed(self) -> bool:
        return all(f.passed for f in self.hard_filters)

    @property
    def failed_hard_filters(self) -> list[HardFilter]:
        return [f for f in self.hard_filters if not f.passed]

    @property
    def score(self) -> int:
        """Score sur ~100 (peut depasser theoriquement, on clamp)."""
        total = sum(f.applied_score for f in self.factors)
        return max(0, min(100, total))

    @property
    def verdict(self) -> Verdict:
        if not self.all_hard_passed:
            return "reject"
        s = self.score
        if s >= 80: return "premium"
        if s >= 55: return "good"
        if s >= 35: return "mediocre"
        return "weak"

    def add_hard(self, name: str, label: str, passed: bool, detail: str = "") -> None:
        self.hard_filters.append(HardFilter(name, label, passed, detail))

    def add_factor(self, name: str, label: str, category: FactorCategory,
                   status: FactorStatus, weight: int, detail: str = "") -> None:
        self.factors.append(Factor(name, label, category, status, weight, detail))

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "all_hard_passed": self.all_hard_passed,
            "hard_filters": [
                {"name": f.name, "label": f.label, "passed": f.passed, "detail": f.detail}
                for f in self.hard_filters
            ],
            "factors": [
                {
                    "name": f.name, "label": f.label, "category": f.category,
                    "status": f.status, "weight": f.weight, "detail": f.detail,
                    "applied_score": f.applied_score,
                }
                for f in self.factors
            ],
        }


# Score minimum pour qu'un trade soit pris (parametrable)
DEFAULT_SCORE_THRESHOLD = 65
