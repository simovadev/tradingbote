"""Harnais de verification : compare la sortie des 4 fonctions ICT avant/apres
optimisation, sur des donnees reelles (XAUUSD + NAS100, 88000 M1).

Usage :
  python verify_optim_identity.py --baseline   # capture l'etat actuel (pickle)
  python verify_optim_identity.py --check      # compare apres optim (vert/rouge)

Regle : egalite STRICTE (bit-a-bit float, len, ordre, dataclasses recursifs).
Une seule divergence -> rollback de l'etape qui l'a causee.
"""
import argparse
import dataclasses
import math
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.liquidity import find_swings, find_sweeps
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.breaker import detect_breakers

DATA = ROOT / "data_vantage"
GOLDEN = ROOT / "golden_snapshots"
ASSETS = ["XAUUSD", "NAS100"]
N_BARS_M1 = 88000


def load_df(asset):
    pq = DATA / f"{asset}_M1.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tail(N_BARS_M1).iloc[:-1]


def compute_all(df):
    """Calcule les 4 sorties dans l'ordre exact du runner."""
    swings = find_swings(df, strength=2)
    sweeps = find_sweeps(df, swings)
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df, swings=swings, sweeps=sweeps, max_group_size=2)
    breakers = detect_breakers(df)  # NB: detect_breakers recalcule en interne (max_group_size=5)
    return {"swings": swings, "sweeps": sweeps, "fvgs": fvgs, "obs": obs, "breakers": breakers}


def assert_eq(a, b, path):
    """Comparaison recursive stricte. Leve sur la 1ere difference."""
    if type(a) is not type(b):
        raise AssertionError(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_eq(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_eq(x, y, f"{path}[{i}]")
        return
    if isinstance(a, float):
        if a == b:
            return
        if math.isnan(a) and math.isnan(b):
            return
        raise AssertionError(f"{path}: float {a!r} != {b!r}")
    if isinstance(a, pd.Timestamp):
        if a == b:
            return
        raise AssertionError(f"{path}: ts {a} != {b}")
    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def save_baseline():
    GOLDEN.mkdir(exist_ok=True)
    for asset in ASSETS:
        df = load_df(asset)
        if df is None:
            print(f"[SKIP] {asset}: parquet absent")
            continue
        out = compute_all(df)
        path = GOLDEN / f"{asset}.pkl"
        with open(path, "wb") as f:
            pickle.dump(out, f)
        sizes = {k: len(v) for k, v in out.items()}
        print(f"[OK] {asset} baseline: {sizes} -> {path.name}")


def check():
    if not GOLDEN.exists():
        print("[FAIL] pas de golden_snapshots/, lance d'abord --baseline")
        return 1
    all_ok = True
    for asset in ASSETS:
        path = GOLDEN / f"{asset}.pkl"
        if not path.exists():
            print(f"[SKIP] {asset}: pas de baseline ({path.name})")
            continue
        df = load_df(asset)
        if df is None:
            print(f"[SKIP] {asset}: parquet absent")
            continue
        with open(path, "rb") as f:
            ref = pickle.load(f)
        cur = compute_all(df)
        for key in ["swings", "sweeps", "fvgs", "obs", "breakers"]:
            try:
                assert_eq(cur[key], ref[key], f"{asset}.{key}")
                print(f"[OK]   {asset}.{key}: {len(cur[key])} elements identiques")
            except AssertionError as e:
                print(f"[FAIL] {e}")
                all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.baseline:
        save_baseline()
    else:
        sys.exit(check())
