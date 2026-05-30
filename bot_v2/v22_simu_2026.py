"""v22_simu_2026.py - Simulation 200E sur 2026 avec :
- 22 actifs avec config dediee (TOP per-asset)
- Sizing DYN tier S/A/B/C/D agressif (10/7/3/1/skip %)
- Leverage max 100x
- Circuit breaker journalier : STOP trades si -20% intraday vs open du jour
- Liquidation si capital < 50E
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"

# 22 actifs RETENUS apres tri user
PER_ASSET_CONFIG = {
    # actif : (h1_mom, atr_max, h_min, h_max, disp, sl_buf, cons)
    "BTCUSD":    (1.00, 2.5,  8, 20, 0.0, 0.00, 2),
    "GER40":     (0.40, None, 7, 22, 0.0, 0.15, 2),
    "USDZAR":    (0.30, 3.0,  7, 21, 0.0, 0.00, 2),
    "USDMXN":    (0.30, None, 6, 20, 0.0, 0.20, 2),
    "FRA40":     (0.30, 2.5,  8, 22, 0.0, 0.20, 2),
    "USDCHF":    (0.20, 2.0,  6, 22, 0.0, 0.20, 2),
    "Cocoa-C":   (0.70, None, 0, 17, 0.3, 0.15, 2),
    "CL-OIL":    (0.50, 2.5,  8, 19, 0.3, 0.20, 2),
    "NAS100":    (0.30, 3.0,  9, 21, 0.3, 0.05, 2),
    "BVSPX":     (0.00, None, 9, 22, 0.5, 0.15, 2),
    "DJ30":      (0.30, 2.5,  9, 22, 0.0, 0.20, 2),
    "UK100":     (0.10, 2.5,  7, 24, 0.5, 0.20, 2),
    "SP500":     (0.20, None, 9, 24, 0.3, 0.00, 2),
    "ETHUSD":    (1.00, 2.0,  9, 22, 0.3, 0.15, 2),
    "Nikkei225": (0.30, 2.0,  0, 21, 0.5, 0.10, 2),
    "GBPUSD":    (0.20, 3.0,  6, 21, 0.0, 0.20, 2),
    "HK50":      (0.40, None, 0, 19, 0.3, 0.20, 2),
    "XAUUSD":    (0.40, 3.0,  6, 21, 0.0, 0.00, 2),
    "USDCAD":    (0.10, None, 9, 17, 0.0, 0.20, 2),
    "GAS-C":     (0.70, 2.5,  9, 19, 0.0, 0.15, 2),
    "AUDUSD":    (0.10, 2.0,  9, 22, 0.3, 0.20, 2),
    "USDJPY":    (0.20, 2.5,  7, 22, 0.0, 0.00, 2),
}

SPREADS = {
    "AUDUSD": 0.00010, "BTCUSD": 6.0, "BVSPX": 8.0, "CL-OIL": 0.06,
    "Cocoa-C": 5.0, "DJ30": 2.0, "ETHUSD": 1.5,
    "EURUSD": 0.00016, "FRA40": 1.5, "GAS-C": 0.005, "GBPUSD": 0.0002,
    "GER40": 0.8, "HK50": 5.0, "NAS100": 1.0, "NZDUSD": 0.00018,
    "Nikkei225": 10.0, "SP500": 0.3, "UK100": 1.0,
    "USDCAD": 0.0002, "USDCHF": 0.0002, "USDJPY": 0.025,
    "USDMXN": 0.002, "USDZAR": 0.004, "USDJPY": 0.025,
    "XAUUSD": 0.14,
}

# Parametres simu
CAPITAL_START = 200.0
RISK_BASE = 0.02
LIQUIDATION = 50.0
LOT_MAX = 100.0                  # plafond ABSOLU : max 100 lots par trade
DAILY_STOP_PCT = 0.20            # circuit breaker journalier

# Contract size par actif (Vantage Demo, recupere depuis MT5)
CONTRACT_SIZE = {
    "AUDUSD": 100000.0, "BTCUSD": 1.0, "BVSPX": 1.0, "CL-OIL": 1000.0,
    "Cocoa-C": 10.0, "DJ30": 1.0, "ETHUSD": 1.0, "FRA40": 1.0,
    "GAS-C": 10000.0, "GBPUSD": 100000.0, "GER40": 1.0, "HK50": 1.0,
    "NAS100": 1.0, "Nikkei225": 1.0, "SP500": 1.0, "UK100": 1.0,
    "USDCAD": 100000.0, "USDCHF": 100000.0, "USDJPY": 100000.0,
    "USDMXN": 100000.0, "USDZAR": 100000.0, "XAUUSD": 100.0,
}
# Min lot (impose par broker)
MIN_LOT = {
    "AUDUSD": 0.01, "BTCUSD": 0.01, "BVSPX": 0.1, "CL-OIL": 0.01,
    "Cocoa-C": 0.1, "DJ30": 0.1, "ETHUSD": 0.01, "FRA40": 0.1,
    "GAS-C": 0.1, "GBPUSD": 0.01, "GER40": 0.1, "HK50": 0.1,
    "NAS100": 0.1, "Nikkei225": 1.0, "SP500": 0.1, "UK100": 0.1,
    "USDCAD": 0.01, "USDCHF": 0.01, "USDJPY": 0.01,
    "USDMXN": 0.01, "USDZAR": 0.01, "XAUUSD": 0.01,
}
SIM_START = pd.Timestamp("2026-01-01", tz="UTC")
SIM_END   = pd.Timestamp("2026-05-30", tz="UTC")

# Sizing par tier
TIER_RISK = {"S": 0.10, "A": 0.07, "B": 0.03, "C": 0.01, "D": 0.0}

TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50


def score_setup(hour_fr, h1_mom, disp, ob_size_atr):
    """Renvoie tier S/A/B/C/D selon V22_COMPLET.md section 9."""
    s = 0
    # Heure FR
    if 14 <= hour_fr < 17: s += 3
    elif 17 <= hour_fr < 21: s += 2
    elif 8 <= hour_fr < 11: s += 2
    elif 11 <= hour_fr < 14: s += 1
    elif 5 <= hour_fr < 8: s += 1
    else: s -= 2
    # H1 momentum
    if h1_mom > 1.0: s += 3
    elif h1_mom > 0.5: s += 2
    elif h1_mom > 0.3: s += 1
    # Displacement
    if disp > 1.0: s += 2
    elif disp > 0.5: s += 1
    # OB size
    if 0.5 <= ob_size_atr <= 3.0: s += 1
    # Tier
    if s >= 7: return "S", s
    if s >= 5: return "A", s
    if s >= 3: return "B", s
    if s >= 1: return "C", s
    return "D", s


def passes_user_rule(df_m5, df_h1, df_d1, ob):
    """D1 + H1 + PD daily (J-1)."""
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    if not d1["ok"]: return False
    d1_h = d1["bias"] == "haussier"
    if not ((ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)):
        return False
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    if h1_close is None: return False
    pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
    if pos < 0: return False
    h1_old = float(df_h1["close"].iloc[pos])
    h1_h = h1_close > h1_old
    if not ((ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)):
        return False
    if df_d1 is None or len(df_d1) < 2: return False
    target_day = pd.Timestamp(val_ts).normalize()
    df_d1_past = df_d1[df_d1.index < target_day]
    if len(df_d1_past) < 1: return False
    pdh = float(df_d1_past["high"].iloc[-1]); pdl = float(df_d1_past["low"].iloc[-1])
    mid = (pdh + pdl) / 2
    price = float(df_m5["close"].iloc[val_idx])
    return (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)


def simulate_trade_outcome(df_m5, ob, atr, spread, sl_buf):
    """Retourne (pnl_R, entry_ts, exit_ts) ou (None, None, None) si pas pris."""
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5): return None, None, None
    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - sl_buf * atr; entry = ob.ob_high
        entry_eff = entry + hs; sl_eff = sl - hs
    else:
        sl = ob.ob_high + sl_buf * atr; entry = ob.ob_low
        entry_eff = entry - hs; sl_eff = sl + hs
    if abs(entry_eff - sl_eff) < 1e-9: return None, None, None
    risk = abs(entry_eff - sl_eff)
    if ob.direction == "bullish":
        tp_1r = entry_eff + risk; tp_2r = entry_eff + TP_RR * risk
    else:
        tp_1r = entry_eff - risk; tp_2r = entry_eff - TP_RR * risk
    fh = df_m5["high"].values[val_idx+1:val_idx+1+MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx+1:val_idx+1+MAX_FILL_BARS]
    if len(fh) == 0: return None, None, None
    entry_idx_rel = None
    for i in range(len(fh)):
        if ob.direction == "bullish":
            if fl[i] <= entry:
                if fl[i] <= sl: return None, None, None
                entry_idx_rel = i; break
        else:
            if fh[i] >= entry:
                if fh[i] >= sl: return None, None, None
                entry_idx_rel = i; break
    if entry_idx_rel is None: return None, None, None
    entry_global = val_idx + 1 + entry_idx_rel
    entry_ts = df_m5.index[entry_global]
    fe = entry_global + 1
    en = min(fe + MAX_HOLD_BARS, len(df_m5))
    if en - fe < 5: return None, None, None
    pfh = df_m5["high"].values[fe:en]; pfl = df_m5["low"].values[fe:en]
    half = False; stop = sl_eff
    for i in range(len(pfh)):
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
        exit_ts = df_m5.index[fe+i]
        if hit_sl and hit_2r:
            if half: return 0.5 * PARTIAL_R, entry_ts, exit_ts
            return -1.0, entry_ts, exit_ts
        if hit_sl:
            if half: return 0.5 * PARTIAL_R, entry_ts, exit_ts
            return -1.0, entry_ts, exit_ts
        if not half and hit_1r:
            half = True; stop = entry_eff
            if hit_2r: return 0.5 * PARTIAL_R + 0.5 * TP_RR, entry_ts, exit_ts
        elif half and hit_2r:
            return 0.5 * PARTIAL_R + 0.5 * TP_RR, entry_ts, exit_ts
    exit_ts = df_m5.index[en-1] if en > 0 else entry_ts
    if half: return 0.5 * PARTIAL_R, entry_ts, exit_ts
    return 0.0, entry_ts, exit_ts


def collect_trades_for_asset(name):
    """Pour 1 actif : applique sa config dediee + scoring tier, retourne liste de trades."""
    cfg = PER_ASSET_CONFIG[name]
    h1m_min, atr_max, h_min, h_max, disp_min, sl_buf, mc = cfg
    sp = SPREADS.get(name, 0.001)
    try:
        df_m5 = pd.read_parquet(DATA / f"{name}_M5.parquet")[["open","high","low","close"]]
        df_h1 = pd.read_parquet(DATA / f"{name}_H1.parquet")[["open","high","low","close"]]
        df_d1 = pd.read_parquet(DATA / f"{name}_D1.parquet")[["open","high","low","close"]]
    except Exception as e:
        print(f"  ERR {name}: {e}")
        return []
    # Contexte avant 2026 pour les OBs + D1
    ctx_start = SIM_START - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx_start) & (df_m5.index < SIM_END)]
    df_h1 = df_h1[(df_h1.index >= ctx_start) & (df_h1.index < SIM_END)]
    df_d1 = df_d1[(df_d1.index >= SIM_START - pd.Timedelta(days=30)) & (df_d1.index < SIM_END)]
    if len(df_m5) < 500: return []

    obs = find_obs_simple(df_m5, as_of_index=len(df_m5)-1, min_consecutive=mc)
    obs = [ob for ob in obs if SIM_START <= ob.validation_ts < SIM_END]
    trades = []
    for ob in obs:
        if not passes_user_rule(df_m5, df_h1, df_d1, ob): continue
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        # Filtre h_min/h_max (heure FR)
        val_ts = ob.validation_ts
        hour_fr = val_ts.tz_convert("Europe/Paris").hour + val_ts.tz_convert("Europe/Paris").minute/60
        if not (h_min <= hour_fr < h_max): continue
        # Filtre h1 momentum
        h1_close = htf_value_at_t(df_h1, val_ts, "close")
        if h1_close is None: continue
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos < 0: continue
        h1_old = float(df_h1["close"].iloc[pos])
        h1_mom = abs((h1_close - h1_old)/h1_old*100) if h1_old > 0 else 0
        if h1_mom < h1m_min: continue
        # Filtre displacement
        if abs(ob.displacement_atr) < disp_min: continue
        # Filtre atr_max
        if atr_max is not None:
            atr_50 = atr_safe(df_m5, ob.validation_index, period=50)
            if atr_50 > 0 and (atr / atr_50) >= atr_max: continue
        # Simu trade
        pnl_R, entry_ts, exit_ts = simulate_trade_outcome(df_m5, ob, atr, sp, sl_buf)
        if pnl_R is None: continue
        # Scoring tier
        ob_size_atr = (ob.ob_high - ob.ob_low) / atr if atr > 0 else 0
        tier, score = score_setup(hour_fr, h1_mom, abs(ob.displacement_atr), ob_size_atr)
        # Risk distance en pts (pour calcul size)
        risk_pts = abs(ob.ob_high - (ob.ob_low - sl_buf*atr)) if ob.direction == "bullish" else \
                   abs((ob.ob_high + sl_buf*atr) - ob.ob_low)
        entry_price = ob.ob_high if ob.direction == "bullish" else ob.ob_low
        trades.append({
            "asset": name, "tier": tier, "score": score,
            "pnl_R": pnl_R, "entry_ts": entry_ts, "exit_ts": exit_ts,
            "validation_ts": val_ts,
            "risk_pts": risk_pts, "entry_price": entry_price,
        })
    return trades


def run_simulation(all_trades):
    """Joue chronologiquement les trades avec sizing tier + leverage 100 + circuit breaker."""
    # Trier par entry_ts (ordre chronologique de prise)
    all_trades = [t for t in all_trades if t["entry_ts"] is not None]
    all_trades.sort(key=lambda t: t["entry_ts"])

    capital = CAPITAL_START
    equity_curve = []
    daily_open_capital = {}
    daily_blocked = set()
    n_taken = 0; n_skipped_d = 0; n_skipped_blocked = 0
    n_liquidated = False
    by_tier = {"S": [0, 0, 0.0], "A": [0, 0, 0.0], "B": [0, 0, 0.0], "C": [0, 0, 0.0], "D": [0, 0, 0.0]}
    # par tier : [n trades pris, n wins, pnl_euros_total]
    peak_capital = capital
    max_dd_pct = 0.0
    peak_at_max_dd = capital
    bottom_at_max_dd = capital

    for t in all_trades:
        if n_liquidated: break
        day = t["entry_ts"].normalize()
        # Init open du jour
        if day not in daily_open_capital:
            daily_open_capital[day] = capital
        # Circuit breaker journalier : si capital < 80% de l'open du jour
        if day in daily_blocked:
            n_skipped_blocked += 1
            continue
        # Sizing par tier
        if t["tier"] == "D":
            n_skipped_d += 1
            continue
        risk_pct = TIER_RISK[t["tier"]]
        risk_amount_target = capital * risk_pct
        # Calcule les lots necessaires :
        # risk_amount = lots * risk_pts * contract_size  (en supposant 1 USD = 1 EUR pour simu)
        cs = CONTRACT_SIZE.get(t["asset"], 1.0)
        min_l = MIN_LOT.get(t["asset"], 0.01)
        if t["risk_pts"] <= 0 or cs <= 0:
            n_skipped_d += 1
            continue
        lots_target = risk_amount_target / (t["risk_pts"] * cs)
        # Plafond ABSOLU : 100 lots
        lots = min(lots_target, LOT_MAX)
        # Min lot du broker
        if lots < min_l:
            # Pas assez de capital pour mettre meme le min_lot a ce risk_pct
            n_skipped_d += 1
            continue
        # Risk reel apres plafonnement
        risk_amount = lots * t["risk_pts"] * cs
        # PnL euros = pnl_R * risk_amount
        pnl_euros = t["pnl_R"] * risk_amount
        capital_before = capital
        capital += pnl_euros
        n_taken += 1
        capped = lots >= LOT_MAX
        by_tier[t["tier"]][0] += 1
        if t["pnl_R"] > 0: by_tier[t["tier"]][1] += 1
        by_tier[t["tier"]][2] += pnl_euros
        equity_curve.append({
            "ts": t["entry_ts"], "asset": t["asset"], "tier": t["tier"],
            "pnl_R": t["pnl_R"], "risk_amount": risk_amount,
            "lots": lots, "capped": capped,
            "pnl_euros": pnl_euros, "capital_before": capital_before, "capital": capital,
        })
        # Update peak + DD
        if capital > peak_capital:
            peak_capital = capital
        dd_pct = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            peak_at_max_dd = peak_capital
            bottom_at_max_dd = capital
        # Check circuit breaker journalier
        loss_today_pct = (daily_open_capital[day] - capital) / daily_open_capital[day]
        if loss_today_pct >= DAILY_STOP_PCT:
            daily_blocked.add(day)
        # Liquidation
        if capital < LIQUIDATION:
            n_liquidated = True
            break

    return {
        "capital_final": capital,
        "n_trades_taken": n_taken,
        "n_skipped_tier_D": n_skipped_d,
        "n_skipped_circuit_breaker": n_skipped_blocked,
        "liquidated": n_liquidated,
        "peak_capital": peak_capital,
        "max_dd_pct": max_dd_pct,
        "peak_at_max_dd": peak_at_max_dd,
        "bottom_at_max_dd": bottom_at_max_dd,
        "by_tier": by_tier,
        "n_days_blocked": len(daily_blocked),
        "equity_curve": equity_curve,
    }


def main():
    print("=" * 100)
    print("V22 SIMULATION 200E 2026 - 22 actifs config dediee")
    print(f"Risk base 2% | Sizing DYN tier S/A/B/C/D = 10/7/3/1/skip %")
    print(f"Lot max {LOT_MAX:.0f} | Circuit breaker -{DAILY_STOP_PCT*100:.0f}% intraday | Liquidation < {LIQUIDATION}E")
    print(f"Periode : {SIM_START.date()} -> {SIM_END.date()}")
    print("=" * 100)

    # Collect trades parallele 22 actifs
    print(f"\n>>> Phase 1 : collecte trades pour {len(PER_ASSET_CONFIG)} actifs...")
    sys.stdout.flush()
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_trades = []
    per_asset_n = {}
    with ProcessPoolExecutor(max_workers=22) as ex:
        futures = {ex.submit(collect_trades_for_asset, name): name for name in PER_ASSET_CONFIG}
        for f in as_completed(futures):
            name = futures[f]
            trades = f.result()
            per_asset_n[name] = len(trades)
            all_trades.extend(trades)
            print(f"  {name:<12} : {len(trades):>4} trades")
            sys.stdout.flush()
    print(f"\nTotal : {len(all_trades)} trades. Collecte en {time.time()-t0:.0f}s")

    # Run simu
    print(f"\n>>> Phase 2 : simulation chronologique...")
    sys.stdout.flush()
    res = run_simulation(all_trades)

    print("\n" + "=" * 100)
    print("RESULTATS SIMULATION 2026")
    print("=" * 100)
    print(f"Capital depart       : {CAPITAL_START:>10.2f} EUR")
    print(f"Capital final        : {res['capital_final']:>10.2f} EUR")
    pct = (res['capital_final'] / CAPITAL_START - 1) * 100
    print(f"Performance          : {pct:>+10.1f}%")
    print(f"Peak capital         : {res['peak_capital']:>10.2f} EUR")
    print(f"Max DD               : {res['max_dd_pct']:>10.1f}% (peak {res['peak_at_max_dd']:.0f} -> {res['bottom_at_max_dd']:.0f} EUR)")
    print(f"Liquidation (<{LIQUIDATION}E)  : {'OUI ❌' if res['liquidated'] else 'NON ✅'}")
    print(f"Trades pris          : {res['n_trades_taken']}")
    print(f"Trades skipped D     : {res['n_skipped_tier_D']}")
    print(f"Trades skipped CB    : {res['n_skipped_circuit_breaker']}")
    print(f"Jours bloques (CB)   : {res['n_days_blocked']}")

    print(f"\n--- PAR TIER ---")
    print(f"{'tier':<6} {'n':<6} {'wins':<6} {'WR%':<8} {'PnL_EUR':<12}")
    for t in ["S", "A", "B", "C"]:
        n, w, pnl = res["by_tier"][t]
        wr = w / n * 100 if n > 0 else 0
        print(f"{t:<6} {n:<6} {w:<6} {wr:<8.1f} {pnl:>+12.2f}")
    # tier D
    n, w, pnl = res["by_tier"]["D"]
    print(f"D     {n:<6} (SKIP - non pris)")

    # Stats lot cap
    if res["equity_curve"]:
        eq = res["equity_curve"]
        n_capped = sum(1 for e in eq if e.get("capped"))
        print(f"\nTrades plafonnes a 100 lots : {n_capped}/{len(eq)} ({n_capped/len(eq)*100:.1f}%)")

    # Equity curve : key milestones
    print(f"\n--- EQUITY CURVE (milestones) ---")
    if res["equity_curve"]:
        eq = res["equity_curve"]
        chunks = max(1, len(eq) // 20)
        for i in range(0, len(eq), chunks):
            e = eq[i]
            cap_mark = " [CAP]" if e.get("capped") else ""
            print(f"  Trade #{i+1:<5} {e['ts'].strftime('%Y-%m-%d %H:%M'):<18} {e['asset']:<10} "
                  f"tier {e['tier']} pnl_R={e['pnl_R']:+.2f} lots={e['lots']:>6.2f}{cap_mark} "
                  f"risk={e['risk_amount']:>8.1f}E pnl={e['pnl_euros']:>+9.2f}E -> capital {e['capital']:>11.2f}E")

    # Sauvegarde JSON
    out = {
        "params": {
            "capital_start": CAPITAL_START, "risk_base": RISK_BASE,
            "lot_max": LOT_MAX, "daily_stop_pct": DAILY_STOP_PCT,
            "liquidation": LIQUIDATION,
            "sim_start": str(SIM_START), "sim_end": str(SIM_END),
            "tier_risk": TIER_RISK,
        },
        "per_asset_n": per_asset_n,
        "results": {k: v for k, v in res.items() if k != "equity_curve"},
        "equity_curve": [
            {"ts": e["ts"].isoformat(), "asset": e["asset"], "tier": e["tier"],
             "pnl_R": e["pnl_R"], "lots": e.get("lots"), "capped": e.get("capped"),
             "risk_amount": e["risk_amount"],
             "pnl_euros": e["pnl_euros"], "capital": e["capital"]}
            for e in res["equity_curve"]
        ],
    }
    Path("/workspace/v22_simu_2026_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nDetails -> /workspace/v22_simu_2026_results.json")
    print(f"Temps total : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
