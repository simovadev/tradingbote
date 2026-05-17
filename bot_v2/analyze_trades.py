"""Analyse forensique des trades : LOSS + manqués.

Pour chaque OB candidat sur N jours :
1. Recolte toutes les donnees du pipeline (verdict, confluences, raison rejet).
2. Simule le trade (qu'il soit accepte ou rejete).
3. Pour les LOSS du bot : analyse selon la bible pourquoi ca a perdu.
4. Pour les REJETS : si le trade aurait ete WIN, c'est un TRADE MANQUE => analyse pourquoi rejete.

Sortie : rapport texte + JSON detaille.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from bot_v2.backtest import simulate_trade
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.market_phase import analyze_phase, has_displacement_at_validation
from bot_v2.config import primary_instruments
from bot_v2.data_loader import load
from bot_v2.pipeline import run_pipeline


def analyze_instrument(
    instrument: str,
    ltf_name: str = "M5",
    htf_name: str = "H1",
    days: int = 3,
) -> dict:
    """Analyse exhaustive de tous les setups de l'instrument."""
    # Pipeline avec TOUS les setups (min_score=0, min_quality=0)
    results = run_pipeline(
        instrument, ltf_name=ltf_name, htf_name=htf_name, htf2_name=None,
        days=days, min_score=0, min_quality=0,
    )

    df_ltf = load(instrument, ltf_name)
    mask = df_ltf.index >= (df_ltf.index.max() - pd.Timedelta(days=days))
    df_ltf = df_ltf[mask]

    # Simule TOUS les OB (acceptes ET rejetes) avec leur setup theorique
    # Pour les rejetes : on calcule entry/sl/tp comme si on les avait pris
    from bot_v2.concepts.htf_swings import collect_htf_swings
    from bot_v2.concepts.liquidity import find_swings
    from bot_v2.trade_setup import build_setup_from_ob

    htf_dfs = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs, swing_strength=3)
    swings_ltf = find_swings(df_ltf)

    analysis = {
        "instrument": instrument,
        "ltf": ltf_name,
        "days": days,
        "total_candidates": len(results),
        "trades_taken": [],          # trades acceptes + outcome
        "trades_rejected": [],       # rejetes + outcome si on les avait pris
        "missed_wins": [],           # rejetes mais qui auraient WIN
        "bad_taken_loss": [],        # acceptes mais LOSS
        "rejection_reasons": Counter(),
    }

    for r in results:
        # Calcule un setup theorique pour TOUS (memes rejetes)
        try:
            setup = build_setup_from_ob(
                df_ltf, r.ob, swings_ltf, instrument,
                htf_swings=htf_swings,
            )
        except Exception:
            setup = None

        if setup is None and r.trade_setup is not None:
            setup = r.trade_setup
        if setup is None:
            continue

        sim = simulate_trade(setup, df_ltf, r.ob.validation_index + 1)

        # Analyse phase au moment du trade
        phase_at_validation = analyze_phase(df_ltf, r.ob.validation_index, lookback=20)

        # Distance vers le TP en bougies (combien de temps pour atteindre TP/SL)
        bars_to_exit = None
        if sim.exit_index is not None and sim.fill_index is not None:
            bars_to_exit = sim.exit_index - sim.fill_index

        # Distance entry -> SL et entry -> TP en ATR
        from bot_v2.concepts.market_phase import has_displacement_at_validation
        prev_14 = df_ltf.iloc[max(0, r.ob.validation_index - 14):r.ob.validation_index]
        atr = float((prev_14["high"] - prev_14["low"]).mean()) if len(prev_14) > 0 else 1.0

        record = {
            "ts": r.ob.validation_ts.isoformat(),
            "direction": r.ob.direction,
            "verdict": r.verdict,
            "score": r.score,
            "confluences": r.confluences,
            "rejection_reason": r.rejection_reason,
            "entry": float(setup.entry_price),
            "sl": float(setup.stop_loss),
            "tp": float(setup.take_profit),
            "rr": float(setup.rr),
            "risk_atr": abs(setup.entry_price - setup.stop_loss) / atr if atr else 0,
            "reward_atr": abs(setup.take_profit - setup.entry_price) / atr if atr else 0,
            "outcome": sim.outcome,
            "bars_to_exit": bars_to_exit,
            "phase_at_validation": phase_at_validation.phase,
            "phase_reason": phase_at_validation.reason,
            "ob_high": float(r.ob.ob_high),
            "ob_low": float(r.ob.ob_low),
            "ob_group_size": r.ob.group_size,
            "tp_source": setup.tp_source,
        }

        if r.verdict == "TRADE":
            analysis["trades_taken"].append(record)
            if sim.outcome == "LOSS":
                analysis["bad_taken_loss"].append(record)
        else:
            analysis["trades_rejected"].append(record)
            analysis["rejection_reasons"][r.rejection_reason[:60]] += 1
            if sim.outcome == "WIN":
                analysis["missed_wins"].append(record)

    return analysis


def print_summary(a: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"ANALYSE {a['instrument']} {a['ltf']} sur {a['days']} jours")
    print(f"{'=' * 70}")
    print(f"  Candidats total      : {a['total_candidates']}")
    print(f"  Trades pris (bot)    : {len(a['trades_taken'])}")
    print(f"  Trades rejetes (bot) : {len(a['trades_rejected'])}")

    taken = a["trades_taken"]
    wins = [t for t in taken if t["outcome"] == "WIN"]
    losses = [t for t in taken if t["outcome"] == "LOSS"]
    no_fills = [t for t in taken if t["outcome"] == "NO_FILL"]
    print(f"    Wins               : {len(wins)}")
    print(f"    Losses             : {len(losses)}")
    print(f"    No fill            : {len(no_fills)}")
    print(f"  Trades MANQUES (rejetes mais WIN) : {len(a['missed_wins'])}")

    print(f"\n=== TOP raisons de rejet ===")
    for reason, n in a["rejection_reasons"].most_common(8):
        print(f"  {n:3d}  {reason}")

    print(f"\n=== TRADES MANQUES (rejetes mais auraient WIN) ===")
    for m in a["missed_wins"]:
        print(f"  {m['ts']} | {m['direction']:8s} | RR={m['rr']:.1f} "
              f"| score={m['score']:3d} | REJET: {m['rejection_reason'][:60]}")
        print(f"      Confluences: {', '.join(m['confluences'][:6])}")

    print(f"\n=== LOSS du bot (PRIS mais LOSS) ===")
    for l in a["bad_taken_loss"]:
        print(f"  {l['ts']} | {l['direction']:8s} | RR={l['rr']:.1f} "
              f"| score={l['score']:3d}")
        print(f"      Confluences: {', '.join(l['confluences'][:6])}")
        print(f"      Phase: {l['phase_at_validation']} | bars_to_exit={l['bars_to_exit']} "
              f"| risk_atr={l['risk_atr']:.1f}x | reward_atr={l['reward_atr']:.1f}x")


def diagnose_loss_pattern(losses: list[dict]) -> dict:
    """Cherche des patterns communs dans les LOSS."""
    if not losses:
        return {}
    return {
        "n_losses": len(losses),
        "avg_score": sum(l["score"] for l in losses) / len(losses),
        "avg_rr": sum(l["rr"] for l in losses) / len(losses),
        "avg_risk_atr": sum(l["risk_atr"] for l in losses) / len(losses),
        "avg_bars_to_exit": sum((l["bars_to_exit"] or 0) for l in losses) / len(losses),
        "phases": dict(Counter(l["phase_at_validation"] for l in losses)),
        "tp_sources": dict(Counter(l["tp_source"] for l in losses)),
        "directions": dict(Counter(l["direction"] for l in losses)),
        # Killzones les plus losantes
        "killzones_lossing": dict(Counter(
            next((c for c in l["confluences"] if c.startswith("killzone=")), "?")
            for l in losses
        )),
    }


def diagnose_missed_wins(missed: list[dict]) -> dict:
    """Cherche des patterns dans les trades manqués."""
    if not missed:
        return {}
    return {
        "n_missed": len(missed),
        "rejection_reasons": dict(Counter(m["rejection_reason"][:50] for m in missed)),
        "avg_rr_missed": sum(m["rr"] for m in missed) / len(missed),
        "directions": dict(Counter(m["direction"] for m in missed)),
    }


def main():
    instruments = sys.argv[1].split(",") if len(sys.argv) > 1 else ["XAUUSD", "NAS100"]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ltf = sys.argv[3] if len(sys.argv) > 3 else "M5"
    htf = sys.argv[4] if len(sys.argv) > 4 else "H1"

    all_results = {}
    for inst in instruments:
        try:
            a = analyze_instrument(inst, ltf_name=ltf, htf_name=htf, days=days)
            all_results[inst] = a
            print_summary(a)

            loss_diag = diagnose_loss_pattern(a["bad_taken_loss"])
            missed_diag = diagnose_missed_wins(a["missed_wins"])

            print(f"\n=== DIAGNOSTIC LOSS {inst} ===")
            for k, v in loss_diag.items():
                print(f"  {k:25s} : {v}")
            print(f"\n=== DIAGNOSTIC MANQUES {inst} ===")
            for k, v in missed_diag.items():
                print(f"  {k:25s} : {v}")
        except Exception as e:
            print(f"ERREUR {inst}: {e}")
            import traceback
            traceback.print_exc()

    # Sauve un JSON detaille
    out_path = Path("c:/Users/Shadow/TradingBot/bot_v2/analyze_output.json")
    # JSON sans Counter (convertit en dict)
    serialisable = {}
    for inst, a in all_results.items():
        serialisable[inst] = {
            **a,
            "rejection_reasons": dict(a["rejection_reasons"]),
        }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON detaille -> {out_path}")


if __name__ == "__main__":
    main()
