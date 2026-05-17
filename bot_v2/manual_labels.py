"""Gestion des labels manuels du user (decision 2026-05-16).

Le user juge VISUELLEMENT chaque OB Vizion brut :
- VALID   : "je prendrais ce trade"
- INVALID : "je ne prendrais pas"
- DRAWN   : OB dessine manuellement par le user (zone qu'il aurait pris)

Ces labels alimentent un modele ML v4 qui apprend le style du user
(pas seulement WIN/LOSS comme v3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd


LABELS_PATH = Path("c:/Users/Shadow/TradingBot/data/manual_labels.json")
DRAWN_OBS_PATH = Path("c:/Users/Shadow/TradingBot/data/drawn_obs.json")


LabelValue = Literal["bug", "pass"]


@dataclass
class ManualLabel:
    """Un label manuel sur un OB Vizion brut existant."""
    ts: str                      # timestamp OB validation_ts ISO
    direction: str               # bullish/bearish
    label: LabelValue
    labeled_at: str              # horodatage du clic
    bug_category: str | None = None   # categorie de bug si label='bug'


@dataclass
class DrawnOB:
    """Un OB dessine par le user (zone que Vizion n'a PAS detectee)."""
    start_ts: str
    end_ts: str
    ob_high: float
    ob_low: float
    direction: str
    drawn_at: str


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_labels() -> list[ManualLabel]:
    return [ManualLabel(**d) for d in _load(LABELS_PATH)]


def save_label(label: ManualLabel) -> None:
    """Save ou update un label (1 par OB ts+direction)."""
    raw = _load(LABELS_PATH)
    key = (label.ts, label.direction)
    raw = [d for d in raw if (d["ts"], d["direction"]) != key]
    raw.append(asdict(label))
    _save(LABELS_PATH, raw)


def save_labels_bulk(labels: list[ManualLabel]) -> int:
    raw = _load(LABELS_PATH)
    seen = {(d["ts"], d["direction"]) for d in raw}
    keep = list(raw)
    for lb in labels:
        key = (lb.ts, lb.direction)
        keep = [d for d in keep if (d["ts"], d["direction"]) != key]
        keep.append(asdict(lb))
        seen.add(key)
    _save(LABELS_PATH, keep)
    return len(keep)


def get_label(ts: str, direction: str) -> ManualLabel | None:
    for d in _load(LABELS_PATH):
        if d["ts"] == ts and d["direction"] == direction:
            return ManualLabel(**d)
    return None


def labels_summary() -> dict:
    raw = _load(LABELS_PATH)
    return {
        "total": len(raw),
        "bug": sum(1 for d in raw if d["label"] == "bug"),
        "pass": sum(1 for d in raw if d["label"] == "pass"),
    }


def load_drawn_obs() -> list[DrawnOB]:
    return [DrawnOB(**d) for d in _load(DRAWN_OBS_PATH)]


def save_drawn_ob(ob: DrawnOB) -> None:
    raw = _load(DRAWN_OBS_PATH)
    raw.append(asdict(ob))
    _save(DRAWN_OBS_PATH, raw)


def labels_as_dataframe() -> pd.DataFrame:
    """Retourne un DataFrame des labels pour entrainement ML v4."""
    raw = _load(LABELS_PATH)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["target"] = (df["label"] == "valid").astype(int)
    return df
