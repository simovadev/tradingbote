"""Dashboard ML — Multi-pages.

Pages :
- /                : Replay live sur mois OOS aleatoire
- /label_day       : Labeling par jour (Phase 1)
- /label_month     : Labeling par mois avec highlight v4 + dessin OB manuel (Phase 3)
- /explain_trade   : Debug d'un setup precis (Phase 4)

Usage :
    ./venv/Scripts/python.exe -m bot_v2.ml_dashboard
    -> http://127.0.0.1:8002
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from bot_v2.backtest import simulate_trade, simulate_trade_trailing
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.daily_bias import build_d1_from_h1
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.htf_swings import collect_htf_swings
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import SMT_PAIRS, get_param
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob
from bot_v2.trade_setup import compute_position_size, TradeSetup
from bot_v2 import ml_filter, ml_label_backend, manual_labels
from bot_v2.ml_label_backend import scan_period, get_candles, get_random_day
from bot_v2.manual_labels import ManualLabel, DrawnOB, save_label, save_drawn_ob, labels_summary
from bot_v2.ml_train_labels import train_v4
from bot_v2.drawn_obs_analyzer import analyze_drawn_obs


app = FastAPI(title="ML Dashboard — Trading Bot v3+")

OOS_START = pd.Timestamp("2025-06-11", tz="UTC")


@dataclass
class _ReplayState:
    paused: bool = False
    stop: bool = False


REPLAY = _ReplayState()


# ============================================================
# PAGE ROUTING
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + REPLAY_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/label_day", response_class=HTMLResponse)
def page_label_day() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + LABEL_DAY_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/label_month", response_class=HTMLResponse)
def page_label_month() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + LABEL_MONTH_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/explain_trade", response_class=HTMLResponse)
def page_explain() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + EXPLAIN_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/analyze", response_class=HTMLResponse)
def page_analyze() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + ANALYZE_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/label_fast", response_class=HTMLResponse)
def page_label_fast() -> HTMLResponse:
    return HTMLResponse(NAV_HTML + LABEL_FAST_HTML,
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/analyze_drawn_obs")
def api_analyze_drawn_obs():
    return analyze_drawn_obs("XAUUSD")


# ============================================================
# API : REPLAY LIVE (page /)
# ============================================================

@app.get("/api/random_period")
def random_period(days: int = Query(30), seed: int | None = Query(None),
                  instrument: str = Query("XAUUSD")) -> dict:
    if seed is not None:
        random.seed(seed)
    df = load(instrument, "M1")
    end_data = df.index[-1]
    available_days = (end_data - OOS_START).days - days
    if available_days <= 0:
        return {"error": "Fenetre OOS trop courte"}
    offset = random.randint(0, available_days)
    start = OOS_START + pd.Timedelta(days=offset)
    end = start + pd.Timedelta(days=days)
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
    }


@app.get("/api/replay_ml")
async def replay_ml(
    instrument: str = Query("XAUUSD"),
    start_ts: int = Query(...),
    end_ts: int = Query(...),
    speed: int = Query(5),
    threshold: float | None = Query(None),
):
    start = pd.Timestamp(start_ts, unit="s", tz="UTC")
    end = pd.Timestamp(end_ts, unit="s", tz="UTC")

    df_ltf_full = load(instrument, "M1")
    df_htf = load(instrument, "M15")
    try:
        df_htf2 = load(instrument, "H1")
    except Exception:
        df_htf2 = None
    try:
        df_d1 = load(instrument, "D1")
        if len(df_d1) < 10:
            raise FileNotFoundError
    except Exception:
        df_h1 = load(instrument, "H1")
        df_d1 = build_d1_from_h1(df_h1)

    htf_dfs_swings = {}
    for tf in ["H1", "H4", "D1"]:
        try:
            htf_dfs_swings[tf] = load(instrument, tf)
        except Exception:
            pass
    htf_swings = collect_htf_swings(htf_dfs_swings, swing_strength=3)

    mask = (df_ltf_full.index >= start) & (df_ltf_full.index <= end)
    df_ltf = df_ltf_full[mask]

    correlated_dfs = {}
    for corr_name, corr_type in SMT_PAIRS.get(instrument, []):
        try:
            df_c = load(corr_name, "M1")
            mask_c = (df_c.index >= start) & (df_c.index <= end)
            correlated_dfs[corr_name] = (df_c[mask_c], corr_type)
        except Exception:
            continue

    # === PRE-CALCUL MINIMAL (rapide, ~5-10 sec) ===
    sws = get_param(instrument, "swing_strength_m1", 2)
    obs = detect_order_blocks(df_ltf, swing_strength=sws, max_group_size=2)

    cache = {
        "swings_ltf": find_swings(df_ltf, strength=sws),
        "fvgs_ltf": detect_fvg(df_ltf),
        "breakers_ltf": detect_breakers(df_ltf),
        "obs_htf": detect_order_blocks(df_htf),
    }
    cache["structure_breaks"] = detect_structure_breaks(df_ltf, swings=cache["swings_ltf"])
    cache["htf_trend"] = detect_trend(cache["swings_ltf"], lookback=6)
    if df_htf2 is not None:
        cache["obs_htf2"] = detect_order_blocks(df_htf2)

    thr = threshold if threshold is not None else ml_filter.ML_THRESHOLDS.get(
        instrument, ml_filter.DEFAULT_THRESHOLD
    )

    # Index OB par validation_index (rapide a iterer)
    obs_by_index: dict[int, list] = {}
    for ob in obs:
        obs_by_index.setdefault(ob.validation_index, []).append(ob)

    # === MULTI-TF M5 (user 2026-05-16) : pre-calcul des setups M5 ===
    # On charge M5, on detecte OB+MSS confirmes, on filtre via le modele M5
    # et on indexe par minute pour emettre les setups M5 pendant le replay M1.
    m5_setups_by_minute: dict[int, list] = {}
    has_m5_model = ml_filter.load_model_for_tf("M5") is not None
    if has_m5_model:
        try:
            df_m5 = load(instrument, "M5")
            mask_m5 = (df_m5.index >= start) & (df_m5.index <= end)
            df_m5_w = df_m5[mask_m5]
            df_m5_htf = load(instrument, "H1")
            df_m5_htf2 = load(instrument, "H4")

            obs_m5 = detect_order_blocks(df_m5_w, swing_strength=2, max_group_size=2)
            swings_m5 = find_swings(df_m5_w, strength=2)
            structure_breaks_m5 = detect_structure_breaks(df_m5_w, swings=swings_m5)
            from bot_v2.concepts.mss_setup import detect_mss_setups, confirm_ob_with_mss
            mss_setups_m5 = detect_mss_setups(df_m5_w, structure_breaks=structure_breaks_m5, swings=swings_m5)
            obs_m5_confirmed = confirm_ob_with_mss(obs_m5, mss_setups_m5, window_bars=10)

            cache_m5 = {
                "swings_ltf": swings_m5,
                "fvgs_ltf": detect_fvg(df_m5_w),
                "breakers_ltf": detect_breakers(df_m5_w),
                "obs_htf": detect_order_blocks(df_m5_htf),
                "obs_htf2": detect_order_blocks(df_m5_htf2),
                "structure_breaks": structure_breaks_m5,
                "htf_trend": detect_trend(swings_m5, lookback=6),
            }

            for ob_m5 in obs_m5_confirmed:
                try:
                    r_m5 = evaluate_ob(
                        ob_m5, df_m5_w, df_m5_htf, df_d1, instrument,
                        ltf_name="M5", htf_name="H1",
                        df_htf2=df_m5_htf2, htf2_name="H4",
                        correlated_dfs={}, htf_swings=htf_swings,
                        df_h1=df_m5_htf, min_score=0, min_quality=0,
                        cache=cache_m5,
                    )
                except Exception:
                    continue
                if r_m5.verdict != "TRADE" or r_m5.trade_setup is None:
                    continue

                proba_m5 = ml_filter.predict_proba_for_tf(r_m5, ob_m5, instrument, tf="M5")
                if proba_m5 is None or proba_m5 < thr:
                    continue

                setup_m5 = r_m5.trade_setup
                event_m5 = {
                    "tf": "M5",
                    "validation_ts": ob_m5.validation_ts.isoformat(),
                    "validation_time": int(ob_m5.validation_ts.timestamp()),
                    "direction": ob_m5.direction,
                    "score": r_m5.score, "ml_proba": round(proba_m5, 3),
                    "killzone": r_m5.killzone_name,
                    "ob_high": float(ob_m5.ob_high), "ob_low": float(ob_m5.ob_low),
                    "entry": float(setup_m5.entry_price),
                    "sl": float(setup_m5.stop_loss),
                    "tp": float(setup_m5.take_profit),
                    "rr": float(setup_m5.rr),
                }
                # Simulate trade pour outcome
                # Defaults au cas ou (lots=0, exception, etc)
                event_m5["outcome"] = "PENDING"
                event_m5["pnl_usd"] = 0.0
                event_m5["realized_rr"] = 0.0
                try:
                    lots, risk_usd = compute_position_size(
                        setup_m5.entry_price, setup_m5.stop_loss, instrument,
                        balance=60.0, risk_pct=0.10,
                    )
                    if lots > 0:
                        sim_setup = TradeSetup(
                            instrument=instrument, direction=setup_m5.direction,
                            entry_price=setup_m5.entry_price, stop_loss=setup_m5.stop_loss,
                            take_profit=setup_m5.take_profit, rr=setup_m5.rr,
                            risk_points=setup_m5.risk_points, reward_points=setup_m5.reward_points,
                            risk_usd=risk_usd, reward_usd=risk_usd * setup_m5.rr,
                            position_size_lots=lots,
                            ob_validation_ts=setup_m5.ob_validation_ts,
                            tp_source=setup_m5.tp_source,
                        )
                        # Trailing retire (user 2026-05-16) : TP fixe simple
                        sim = simulate_trade(sim_setup, df_m5_w, ob_m5.validation_index + 1)
                        event_m5["outcome"] = sim.outcome
                        event_m5["pnl_usd"] = round(float(sim.pnl_usd), 2)
                        event_m5["realized_rr"] = round(sim.pnl_usd / max(risk_usd, 1e-9), 3)
                        if sim.fill_ts is not None:
                            event_m5["fill_time"] = int(sim.fill_ts.timestamp())
                        if sim.exit_ts is not None:
                            event_m5["exit_time"] = int(sim.exit_ts.timestamp())
                        if sim.exit_price is not None:
                            event_m5["exit_price"] = float(sim.exit_price)
                    else:
                        print(f"WARN M5: lots=0 pour setup {ob_m5.validation_ts}", flush=True)
                except Exception as e:
                    print(f"ERR simulate_trade M5: {e}", flush=True)
                    event_m5["outcome"] = "ERROR"

                # Index par minute (alignee sur M1 par troncation des secondes)
                minute_key = int(ob_m5.validation_ts.timestamp()) // 60 * 60
                m5_setups_by_minute.setdefault(minute_key, []).append(event_m5)
        except Exception as e:
            print(f"M5 pre-calcul erreur : {e}", flush=True)

    REPLAY.paused = False
    REPLAY.stop = False
    delay = max(speed, 1) / 1000.0

    # === STREAMING : evaluation DANS le stream, pas avant ===
    async def event_stream():
        # Ready immediat apres pre-calcul minimal
        n_m5_setups = sum(len(v) for v in m5_setups_by_minute.values())
        yield (
            f"event: ready\ndata: "
            f"{json.dumps({'total': len(df_ltf), 'n_obs_detected': len(obs), 'threshold': thr, 'n_m5_setups': n_m5_setups})}\n\n"
        )

        for i in range(len(df_ltf)):
            if REPLAY.stop:
                break
            while REPLAY.paused and not REPLAY.stop:
                await asyncio.sleep(0.1)
            if REPLAY.stop:
                break

            row = df_ltf.iloc[i]
            ts = df_ltf.index[i]
            candle = {
                "time": int(ts.timestamp()),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
            }
            yield f"event: candle\ndata: {json.dumps(candle)}\n\n"

            # Emettre les setups M5 qui se valident a cette minute
            minute_key = int(ts.timestamp()) // 60 * 60
            if minute_key in m5_setups_by_minute:
                for se_m5 in m5_setups_by_minute[minute_key]:
                    yield f"event: setup\ndata: {json.dumps(se_m5)}\n\n"

            # Si un OB est valide sur cette bougie, on l'evalue MAINTENANT (a la volee)
            if i in obs_by_index:
                for ob in obs_by_index[i]:
                    try:
                        r = evaluate_ob(
                            ob, df_ltf, df_htf, df_d1, instrument,
                            ltf_name="M1", htf_name="M15",
                            df_htf2=df_htf2, htf2_name="H1",
                            correlated_dfs=correlated_dfs,
                            htf_swings=htf_swings,
                            df_h1=df_htf2,
                            min_score=0, min_quality=0,
                            cache=cache,
                        )
                    except Exception:
                        continue
                    if r.verdict != "TRADE" or r.trade_setup is None:
                        continue

                    take, proba = ml_filter.should_take(r, ob, instrument)
                    if not take:
                        continue

                    setup = r.trade_setup
                    event: dict = {
                        "tf": "M1",
                        "validation_ts": ob.validation_ts.isoformat(),
                        "validation_time": int(ob.validation_ts.timestamp()),
                        "direction": ob.direction,
                        "score": r.score, "ml_proba": round(proba, 3),
                        "killzone": r.killzone_name,
                        "ob_high": float(ob.ob_high), "ob_low": float(ob.ob_low),
                        "entry": float(setup.entry_price),
                        "sl": float(setup.stop_loss),
                        "tp": float(setup.take_profit),
                        "rr": float(setup.rr),
                    }
                    # Defaults au cas ou : si lots == 0 ou exception
                    event["outcome"] = "PENDING"
                    event["realized_rr"] = 0.0
                    event["pnl_usd"] = 0.0
                    try:
                        lots, risk_usd = compute_position_size(
                            setup.entry_price, setup.stop_loss, instrument,
                            balance=60.0, risk_pct=0.10,
                        )
                        if lots > 0:
                            sim_setup = TradeSetup(
                                instrument=instrument, direction=setup.direction,
                                entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                                take_profit=setup.take_profit, rr=setup.rr,
                                risk_points=setup.risk_points, reward_points=setup.reward_points,
                                risk_usd=risk_usd, reward_usd=risk_usd * setup.rr,
                                position_size_lots=lots,
                                ob_validation_ts=setup.ob_validation_ts,
                                tp_source=setup.tp_source,
                            )
                            # Trailing retire (user 2026-05-16) : TP fixe simple
                            sim = simulate_trade(sim_setup, df_ltf, ob.validation_index + 1)
                            event["outcome"] = sim.outcome
                            event["pnl_usd"] = round(float(sim.pnl_usd), 2)
                            event["realized_rr"] = round(sim.pnl_usd / max(risk_usd, 1e-9), 3)
                            if sim.fill_ts is not None:
                                event["fill_time"] = int(sim.fill_ts.timestamp())
                            if sim.exit_ts is not None:
                                event["exit_time"] = int(sim.exit_ts.timestamp())
                            if sim.exit_price is not None:
                                event["exit_price"] = float(sim.exit_price)
                        else:
                            print(f"WARN: lots=0 pour setup {ob.validation_ts} (entry={setup.entry_price}, sl={setup.stop_loss})", flush=True)
                    except Exception as e:
                        print(f"ERR simulate_trade: {e}", flush=True)
                        event["outcome"] = "ERROR"

                    yield f"event: setup\ndata: {json.dumps(event)}\n\n"

            await asyncio.sleep(delay)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/replay/pause")
def replay_pause():
    REPLAY.paused = True
    return {"paused": True}


@app.post("/api/replay/resume")
def replay_resume():
    REPLAY.paused = False
    return {"paused": False}


@app.post("/api/replay/stop")
def replay_stop():
    REPLAY.stop = True
    return {"stopped": True}


# ============================================================
# API : LABEL_DAY
# ============================================================

@app.get("/api/label_day/random")
def api_label_day_random():
    start, end = get_random_day("XAUUSD")
    return _label_period_payload(start, end)


@app.get("/api/label_day/{date_str}")
def api_label_day_date(date_str: str):
    """date_str format: 2025-08-15"""
    start = pd.Timestamp(date_str, tz="UTC").normalize()
    end = start + pd.Timedelta(days=1)
    return _label_period_payload(start, end)


@app.get("/api/diagnose_day/{date_str}")
def api_diagnose_day(date_str: str):
    """Pourquoi seuls X OB passent ? Quel filtre rejette le plus ?"""
    start = pd.Timestamp(date_str, tz="UTC").normalize()
    end = start + pd.Timedelta(days=1)
    return ml_label_backend.diagnose_period("XAUUSD", start, end)


def _label_period_payload(start: pd.Timestamp, end: pd.Timestamp,
                          include_rejected: bool = True) -> dict:
    trades = scan_period("XAUUSD", start, end,
                         include_outcome=True, include_rejected=include_rejected)
    # Joindre labels existants + bug_category
    existing = {(lb.ts, lb.direction): lb for lb in manual_labels.load_labels()}
    trades_dicts = []
    for t in trades:
        d = asdict(t)
        lb = existing.get((t.ts, t.direction))
        d["existing_label"] = lb.label if lb else None
        d["bug_category"] = lb.bug_category if lb else None
        trades_dicts.append(d)
    candles = get_candles("XAUUSD", start, end)
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "candles": candles, "trades": trades_dicts,
        "summary": labels_summary(),
    }


# ============================================================
# API : LABEL_MONTH
# ============================================================

@app.get("/api/label_month/random")
def api_label_month_random():
    df = load("XAUUSD", "M1")
    end_data = df.index[-1]
    avail = (end_data - OOS_START).days - 30
    offset = random.randint(0, max(avail, 0))
    start = OOS_START + pd.Timedelta(days=offset)
    end = start + pd.Timedelta(days=30)
    return _label_period_payload(start, end)


@app.get("/api/label_month/{date_str}")
def api_label_month_date(date_str: str):
    """date_str format: 2025-08-15"""
    start = pd.Timestamp(date_str, tz="UTC").normalize()
    end = start + pd.Timedelta(days=30)
    return _label_period_payload(start, end)


# ============================================================
# API : SAVE LABEL & DRAWN OB
# ============================================================

class LabelIn(BaseModel):
    ts: str
    direction: str
    label: str  # "bug" | "pass"
    bug_category: str | None = None  # categorie si bug


@app.post("/api/save_label")
def api_save_label(body: LabelIn):
    if body.label not in ("bug", "pass"):
        return {"error": "label invalide"}
    lb = ManualLabel(
        ts=body.ts, direction=body.direction, label=body.label,
        labeled_at=datetime.now(timezone.utc).isoformat(),
        bug_category=body.bug_category if body.label == "bug" else None,
    )
    save_label(lb)
    return {"saved": True, "summary": labels_summary()}


@app.get("/api/bug_categories_stats")
def api_bug_categories_stats():
    """Stats des categories de bug pour identifier les patterns."""
    from collections import Counter
    raw = [d for d in manual_labels._load(manual_labels.LABELS_PATH)
           if d.get("label") == "bug"]
    counter = Counter(d.get("bug_category") or "non_categorise" for d in raw)
    return {
        "total_bugs": len(raw),
        "by_category": dict(counter.most_common()),
    }


class DrawnOBIn(BaseModel):
    start_ts: str
    end_ts: str
    ob_high: float
    ob_low: float
    direction: str


@app.post("/api/save_drawn_ob")
def api_save_drawn_ob(body: DrawnOBIn):
    if body.direction not in ("bullish", "bearish"):
        return {"error": "direction invalide"}
    ob = DrawnOB(
        start_ts=body.start_ts, end_ts=body.end_ts,
        ob_high=body.ob_high, ob_low=body.ob_low,
        direction=body.direction,
        drawn_at=datetime.now(timezone.utc).isoformat(),
    )
    save_drawn_ob(ob)
    return {"saved": True, "n_drawn": len(manual_labels.load_drawn_obs())}


# ============================================================
# API : TRAIN V4
# ============================================================

@app.post("/api/train_v4")
def api_train_v4():
    result = train_v4()
    return result


@app.get("/api/labels_summary")
def api_labels_summary():
    return labels_summary()


# ============================================================
# API : EXPLAIN (Phase 4)
# ============================================================

@app.get("/api/explain")
def api_explain(ts: str, direction: str):
    """Pourquoi Vizion (n')a (pas) detecte un setup a ts ? Quels params changer ?"""
    target_ts = pd.Timestamp(ts, tz="UTC")
    instrument = "XAUUSD"
    df_ltf = load(instrument, "M1")
    # Fenetre +- 1h
    win_start = target_ts - pd.Timedelta(hours=1)
    win_end = target_ts + pd.Timedelta(hours=1)
    mask = (df_ltf.index >= win_start) & (df_ltf.index <= win_end)
    df_w = df_ltf[mask]

    diagnostics = []
    # Test plusieurs swing_strength
    for sws in [1, 2, 3]:
        swings = find_swings(df_w, strength=sws)
        n_sw = len(swings)
        obs = detect_order_blocks(df_w, swing_strength=sws, max_group_size=5)
        n_ob = len(obs)
        matching_dir = sum(1 for ob in obs if ob.direction == direction)
        diagnostics.append({
            "swing_strength": sws,
            "n_swings": n_sw,
            "n_ob_total": n_ob,
            "n_ob_matching_direction": matching_dir,
        })

    candles = get_candles(instrument, win_start, win_end)
    return {
        "target_ts": ts, "direction": direction,
        "diagnostics": diagnostics,
        "candles": candles,
        "advice": (
            "Si n_ob augmente avec swing_strength plus bas => assouplir swing_strength_m1 dans config. "
            "Si toujours 0 OB matching => probablement pas de groupe inversee complete a cet endroit."
        ),
    }


# ============================================================
# HTML : NAV + PAGES
# ============================================================

NAV_HTML = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>ML Dashboard</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#c9d1d9;
        --muted:#8b949e; --green:#3fb950; --red:#f85149; --blue:#58a6ff; --yellow:#d29922; }
* { box-sizing: border-box; }
body { margin:0; padding:0; background:var(--bg); color:var(--text);
       font-family: -apple-system, "Segoe UI", sans-serif; font-size:14px; }
nav { background:var(--panel); border-bottom:1px solid var(--border); padding:10px 20px;
      display:flex; gap:8px; align-items:center; }
nav a { color:var(--muted); text-decoration:none; padding:6px 14px; border-radius:4px;
        font-size:13px; }
nav a:hover { background:#21262d; color:var(--text); }
nav a.active { background:var(--blue); color:white; }
nav .brand { font-weight:600; margin-right:20px; color:var(--text); font-size:15px; }
nav .spacer { flex:1; }
nav .summary { color:var(--muted); font-size:12px; }
button { cursor:pointer; background:#21262d; color:var(--text); border:1px solid var(--border);
         padding:6px 12px; border-radius:4px; font-size:13px; font-family:inherit; }
button:hover:not(:disabled) { background:#30363d; }
button:disabled { opacity:0.4; cursor:not-allowed; }
button.primary { background:var(--blue); color:white; border-color:var(--blue); }
button.primary:hover:not(:disabled) { background:#388bfd; }
button.danger { background:#6e2727; color:white; border-color:#6e2727; }
button.success { background:var(--green); color:white; border-color:var(--green); }
button.warn { background:var(--yellow); color:#0d1117; border-color:var(--yellow); }
select, input { background:#21262d; color:var(--text); border:1px solid var(--border);
                padding:6px 10px; border-radius:4px; font-size:13px; font-family:inherit; }
.muted { color:var(--muted); }
.empty { padding:40px; text-align:center; color:var(--muted); font-style:italic; }
</style>
</head><body>
<nav>
  <span class="brand">TradingBot v3+</span>
  <a href="/" id="nav-replay">Replay Live</a>
  <a href="/label_fast" id="nav-fast">🚀 Label Fast</a>
  <a href="/label_day" id="nav-day">Label Day</a>
  <a href="/label_month" id="nav-month">Label Month + Draw</a>
  <a href="/analyze" id="nav-analyze">Analyser mes OB</a>
  <a href="/explain_trade" id="nav-explain">Explain Trade</a>
  <span class="spacer"></span>
  <span class="summary" id="navSummary">...</span>
</nav>
<script>
(function() {
  const path = window.location.pathname;
  if (path === "/") document.getElementById("nav-replay").classList.add("active");
  else if (path.startsWith("/label_fast")) document.getElementById("nav-fast").classList.add("active");
  else if (path.startsWith("/label_day")) document.getElementById("nav-day").classList.add("active");
  else if (path.startsWith("/label_month")) document.getElementById("nav-month").classList.add("active");
  else if (path.startsWith("/analyze")) document.getElementById("nav-analyze").classList.add("active");
  else if (path.startsWith("/explain")) document.getElementById("nav-explain").classList.add("active");

  fetch("/api/labels_summary").then(r => r.json()).then(s => {
    document.getElementById("navSummary").textContent =
      `Labels : ${s.valid} valid | ${s.invalid} invalid | ${s.skip} skip (total ${s.total})`;
  });
})();
</script>
"""


REPLAY_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">Replay Live OOS — Mois aleatoire post 2025-06</h1>
  <div style="color:#8b949e; font-size:12px; margin-top:2px;">
    Le ML n'a JAMAIS vu ces donnees pendant l'entrainement.
  </div>
</header>

<div style="padding:12px 20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#161b22; border-bottom:1px solid #30363d;">
  <label class="muted">Actif</label>
  <select id="instrument">
    <option value="XAUUSD">XAUUSD (Or)</option>
    <option value="NAS100">NAS100 (Nasdaq)</option>
    <option value="GER40">GER40 (DAX 40)</option>
    <option value="BTCUSD">BTCUSD (Bitcoin)</option>
    <option value="EURUSD">EURUSD (EUR/USD)</option>
    <option value="GBPUSD">GBPUSD (GBP/USD)</option>
    <option value="AUDUSD">AUDUSD (AUD/USD)</option>
    <option value="USDJPY">USDJPY (USD/JPY)</option>
  </select>
  <label class="muted">Jours</label>
  <input id="days" type="number" min="7" max="60" value="30" style="width:60px">
  <label class="muted">Seuil ML</label>
  <input id="threshold" type="number" step="0.05" min="0" max="1" value="0.50" style="width:70px">
  <label class="muted">Vitesse</label>
  <select id="speed">
    <option value="50">x1 (50ms)</option>
    <option value="20">x5 (20ms)</option>
    <option value="5" selected>x20 (5ms)</option>
    <option value="2">x50 (2ms)</option>
    <option value="1">x100 (1ms)</option>
  </select>
  <button id="play" class="primary">Nouveau mois + Replay</button>
  <button id="pause" disabled>Pause</button>
  <button id="stop" class="danger" disabled>Stop</button>
  <span id="period" class="muted"></span>
  <span id="info" class="muted" style="margin-left:20px"></span>
</div>

<div id="chart" style="width: calc(100% - 40px); height: 55vh; margin: 0 20px"></div>

<div style="display:grid; grid-template-columns:repeat(7,1fr); gap:8px; padding:12px 20px; background:#161b22;">
  <div class="stat info"><div class="l">Bougies</div><div class="v" id="s_candles">0</div></div>
  <div class="stat info"><div class="l">Vizion</div><div class="v" id="s_vizion">0</div></div>
  <div class="stat warn"><div class="l">Pris ML</div><div class="v" id="s_taken">0</div></div>
  <div class="stat good"><div class="l">Wins</div><div class="v" id="s_wins">0</div></div>
  <div class="stat bad"><div class="l">Losses</div><div class="v" id="s_losses">0</div></div>
  <div class="stat"><div class="l">WR</div><div class="v" id="s_wr">-</div></div>
  <div class="stat"><div class="l">Balance (€)</div><div class="v" id="s_pnl">100.00€</div></div>
</div>

<style>
.stat { background:#0d1117; padding:10px 12px; border-radius:4px; border:1px solid #30363d; }
.stat .l { color:#8b949e; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; }
.stat .v { font-size:18px; font-weight:600; margin-top:2px; }
.stat.good .v { color:#3fb950; } .stat.bad .v { color:#f85149; }
.stat.info .v { color:#58a6ff; } .stat.warn .v { color:#d29922; }
</style>

<table style="width: calc(100% - 40px); margin: 0 20px 20px; border-collapse: collapse; background:#161b22; border:1px solid #30363d; border-radius:6px; overflow:hidden;">
  <thead><tr>
    <th>Time</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th><th>RR</th>
    <th>Score</th><th>ML</th><th>KZ</th><th>Outcome</th><th>PnL €</th>
  </tr></thead>
  <tbody id="trades_body"><tr><td colspan="11" class="empty">Aucun trade.</td></tr></tbody>
</table>

<style>
table th, table td { padding:6px 10px; text-align:left; border-bottom:1px solid #21262d;
                     font-size:12px; font-family:monospace; }
table th { background:#0d1117; color:#8b949e; text-transform:uppercase; font-size:10px;
           letter-spacing:0.5px; font-family:inherit; }
td.bullish { color:#3fb950; font-weight:600; } td.bearish { color:#f85149; font-weight:600; }
td.WIN { color:#3fb950; font-weight:600; } td.LOSS { color:#f85149; font-weight:600; }
td.NO_FILL { color:#8b949e; } td.pnl-pos { color:#3fb950; } td.pnl-neg { color:#f85149; }
</style>

<script>
// Money management compose (user 2026-05-16) :
// - Balance depart 100 EUR
// - Risk = 20% de la balance COURANTE par trade
// - Apres chaque WIN : balance += balance * 0.20 * rr
// - Apres chaque LOSS : balance -= balance * 0.20
const INITIAL_BALANCE = 100;
const RISK_PCT = 0.20;

// REGLE 1 (user 2026-05-16) : Cooldown 15 min toutes directions sur meme actif.
// Si un trade a ete pris dans les 15 dernieres minutes, on skip les nouveaux setups.
const COOLDOWN_SEC = 15 * 60;

let chart, candleSeries, eventSource = null, paused = false;
let stats = { candles:0, vizion:0, taken:0, wins:0, losses:0, no_fill:0, pnl:0, balance: INITIAL_BALANCE };
let tradeZones = [], pendingCandles = [], allMarkers = [];
// Bug fix #4 : cooldown separe par TF (M1 et M5 ne se bloquent pas mutuellement)
let lastTradeTimeByTF = { M1: 0, M5: 0 };
let nSkippedCooldown = 0;

function $(id) { return document.getElementById(id); }
function fmt(n, d) { return Number(n).toFixed(d === undefined ? 2 : d); }

function applyTradeToBalance(outcome, realized_rr) {
  // Calcule le PnL en fonction de la balance COURANTE et du RR REEL apres trailing.
  // Fix bug #3 (user 2026-05-16) : on utilise realized_rr (peut etre 0.8 si trailing
  // a coupe avant TP, ou 3.5 si gros run capture) au lieu du RR initial du setup.
  const risk_amount = stats.balance * RISK_PCT;
  if (outcome === 'WIN') {
    // realized_rr > 0 dans ce cas (peut etre 0.5 a 10+)
    const gain = risk_amount * realized_rr;
    stats.balance += gain;
    return gain;
  } else if (outcome === 'LOSS') {
    // realized_rr est negatif ou ~ -1 (peut etre -0.2 si trailing en breakeven negatif)
    const loss = risk_amount * Math.abs(realized_rr);
    stats.balance -= loss;
    return -loss;
  }
  return 0;  // NO_FILL ou PENDING
}

function updateStats() {
  $('s_candles').textContent = stats.candles;
  $('s_vizion').textContent = stats.vizion;
  $('s_taken').textContent = stats.taken;
  $('s_wins').textContent = stats.wins;
  $('s_losses').textContent = stats.losses;
  const closed = stats.wins + stats.losses;
  $('s_wr').textContent = closed > 0 ? fmt(stats.wins/closed*100, 1) + '%' : '-';
  const pnlEl = $('s_pnl');
  const pnlVsInit = stats.balance - INITIAL_BALANCE;
  const pctGain = (pnlVsInit / INITIAL_BALANCE * 100);
  pnlEl.innerHTML = `${fmt(stats.balance, 2)}€<br><span style="font-size:11px; color:${pnlVsInit>=0?'#3fb950':'#f85149'}">(${pnlVsInit>=0?'+':''}${fmt(pctGain,1)}%)</span>`;
  pnlEl.parentElement.classList.toggle('good', pnlVsInit > 0);
  pnlEl.parentElement.classList.toggle('bad', pnlVsInit < 0);
}

function drawTradeZone(s) {
  if (!s.entry || !s.sl || !s.tp) return;
  const startTime = s.fill_time || s.validation_time;
  const endTime = s.exit_time || (startTime + 60 * 60);
  let tpStrong = false, slStrong = false;
  if (s.outcome === 'WIN') tpStrong = true;
  else if (s.outcome === 'LOSS') slStrong = true;
  const tpFill = chart.addBaselineSeries({
    baseValue: { type:'price', price:s.entry },
    topFillColor1:`rgba(63,185,80,${tpStrong?0.55:0.20})`,
    topFillColor2:`rgba(63,185,80,${tpStrong?0.25:0.08})`,
    topLineColor:'rgba(63,185,80,0.7)',
    bottomFillColor1:`rgba(63,185,80,${tpStrong?0.55:0.20})`,
    bottomFillColor2:`rgba(63,185,80,${tpStrong?0.25:0.08})`,
    bottomLineColor:'rgba(63,185,80,0.7)',
    lineWidth:1, priceLineVisible:false, lastValueVisible:false,
  });
  tpFill.setData([{time:startTime, value:s.tp}, {time:endTime, value:s.tp}]);
  const slFill = chart.addBaselineSeries({
    baseValue: { type:'price', price:s.entry },
    topFillColor1:`rgba(248,81,73,${slStrong?0.55:0.20})`,
    topFillColor2:`rgba(248,81,73,${slStrong?0.25:0.08})`,
    topLineColor:'rgba(248,81,73,0.7)',
    bottomFillColor1:`rgba(248,81,73,${slStrong?0.55:0.20})`,
    bottomFillColor2:`rgba(248,81,73,${slStrong?0.25:0.08})`,
    bottomLineColor:'rgba(248,81,73,0.7)',
    lineWidth:1, priceLineVisible:false, lastValueVisible:false,
  });
  slFill.setData([{time:startTime, value:s.sl}, {time:endTime, value:s.sl}]);
  const entryColor = s.direction === 'bullish' ? '#3fb950' : '#f85149';
  const entryLine = chart.addLineSeries({
    color:entryColor, lineWidth:2,
    priceLineVisible:false, lastValueVisible:false,
  });
  entryLine.setData([{time:startTime, value:s.entry}, {time:endTime, value:s.entry}]);
  tradeZones.push({tpFill, slFill, entryLine});
  if (tradeZones.length > 10) {
    const old = tradeZones.shift();
    try{chart.removeSeries(old.tpFill);}catch(e){}
    try{chart.removeSeries(old.slFill);}catch(e){}
    try{chart.removeSeries(old.entryLine);}catch(e){}
  }
}

function clearAllZones() {
  for (const z of tradeZones) {
    try{chart.removeSeries(z.tpFill);}catch(e){}
    try{chart.removeSeries(z.slFill);}catch(e){}
    try{chart.removeSeries(z.entryLine);}catch(e){}
  }
  tradeZones = []; allMarkers = [];
}

function addTradeRow(s) {
  const tbody = $('trades_body');
  if (tbody.querySelector('.empty')) tbody.innerHTML = '';
  const pnlCls = s.pnl_usd > 0 ? 'pnl-pos' : (s.pnl_usd < 0 ? 'pnl-neg' : '');
  const time = s.validation_ts.split('+')[0].replace('T', ' ').substring(0, 16);
  const tfBadge = s.tf === 'M5'
    ? '<span style="background:#1c2a4a; color:#58a6ff; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:600;">M5</span>'
    : '<span style="background:#1f3a1d; color:#3fb950; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:600;">M1</span>';
  const tr = document.createElement('tr');
  tr.innerHTML = `<td class="muted">${time} ${tfBadge}</td>
    <td class="${s.direction}">${s.direction === 'bullish' ? 'BUY' : 'SELL'}</td>
    <td>${fmt(s.entry, 3)}</td><td>${fmt(s.sl, 3)}</td><td>${fmt(s.tp, 3)}</td>
    <td>${fmt(s.rr, 2)}</td><td>${s.score}</td><td>${fmt(s.ml_proba, 3)}</td>
    <td class="muted">${s.killzone || '-'}</td>
    <td class="${s.outcome || 'PENDING'}">${s.outcome || 'PENDING'}</td>
    <td class="${pnlCls}">${s.pnl_usd >= 0 ? '+' : ''}${fmt(s.pnl_usd || 0, 2)}</td>`;
  tbody.insertBefore(tr, tbody.firstChild);
}

async function startReplay() {
  stopReplay();
  $('play').disabled = true; $('pause').disabled = false; $('stop').disabled = false;
  stats = {candles:0, vizion:0, taken:0, wins:0, losses:0, no_fill:0, pnl:0, balance: INITIAL_BALANCE};
  lastTradeTimeByTF = { M1: 0, M5: 0 };
  nSkippedCooldown = 0;
  updateStats();
  candleSeries.setData([]); candleSeries.setMarkers([]); clearAllZones();
  $('trades_body').innerHTML = '<tr><td colspan="11" class="empty">Pioche mois OOS...</td></tr>';
  $('info').textContent = 'Pioche d\\'un mois...';
  const days = $('days').value;
  const instrument_pre = $('instrument').value;
  const period = await fetch(`/api/random_period?days=${days}&seed=${Date.now()}&instrument=${instrument_pre}`).then(r=>r.json());
  if (period.error) { $('info').textContent = 'Erreur: '+period.error; stopReplay(); return; }
  $('period').textContent = `${period.start.substring(0,10)} -> ${period.end.substring(0,10)}`;
  $('info').textContent = 'Pre-calcul (30-180s)...';
  const speed = $('speed').value;
  const threshold = $('threshold').value;
  paused = false;
  const instrument = $('instrument').value;
  eventSource = new EventSource(
    `/api/replay_ml?instrument=${instrument}&start_ts=${period.start_ts}&end_ts=${period.end_ts}&speed=${speed}&threshold=${threshold}`
  );
  eventSource.addEventListener('ready', e => {
    const d = JSON.parse(e.data);
    const m5info = d.n_m5_setups ? ` | M5: ${d.n_m5_setups} setups` : '';
    $('info').textContent = `${d.total} bougies | ${d.n_obs_detected} OB M1 detectes${m5info} | seuil ML ${d.threshold}`;
    $('trades_body').innerHTML = '<tr><td colspan="11" class="empty">Replay en cours, attente premier setup...</td></tr>';
  });
  eventSource.addEventListener('candle', e => {
    if (paused) { pendingCandles.push(JSON.parse(e.data)); return; }
    const c = JSON.parse(e.data);
    candleSeries.update(c); stats.candles++;
    if (stats.candles % 50 === 0) updateStats();
  });
  eventSource.addEventListener('setup', e => {
    const s = JSON.parse(e.data);
    stats.vizion++;
    // REGLE 1 : Cooldown 15 min PAR TF (bug fix #4 : M1 et M5 independants)
    const tfKey = s.tf || 'M1';
    const lastTime = lastTradeTimeByTF[tfKey] || 0;
    if (s.validation_time - lastTime < COOLDOWN_SEC) {
      nSkippedCooldown++;
      return;  // skip ce setup
    }
    lastTradeTimeByTF[tfKey] = s.validation_time;
    stats.taken++;
    // Money mgmt compose : on utilise le RR REEL apres trailing (bug fix #3)
    // realized_rr peut etre 0.5 a 10+ pour WIN, ou ~1 pour LOSS, fallback rr initial sinon.
    const rrToUse = (s.realized_rr !== undefined && s.realized_rr !== null) ? Math.abs(s.realized_rr) : (s.rr || 2);
    const real_pnl = applyTradeToBalance(s.outcome, rrToUse);
    s.pnl_usd = real_pnl;
    if (s.outcome === 'WIN') { stats.wins++; }
    else if (s.outcome === 'LOSS') { stats.losses++; }
    else if (s.outcome === 'NO_FILL') stats.no_fill++;
    // M5 = couleurs differentes (cyan/orange) pour distinguer
    const isM5 = s.tf === 'M5';
    let mkColor;
    if (isM5) {
      mkColor = s.direction === 'bullish' ? '#58a6ff' : '#d29922';
    } else {
      mkColor = s.direction === 'bullish' ? '#3fb950' : '#f85149';
    }
    const tfTag = isM5 ? '[M5] ' : '';
    allMarkers.push({
      time:s.validation_time,
      position: s.direction === 'bullish' ? 'belowBar' : 'aboveBar',
      color: mkColor,
      shape: s.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
      text: `${tfTag}${s.direction === 'bullish' ? 'BUY' : 'SELL'} ml=${s.ml_proba}`,
    });
    candleSeries.setMarkers(allMarkers);
    drawTradeZone(s); addTradeRow(s); updateStats();
  });
  eventSource.addEventListener('done', () => {
    $('info').textContent = 'Termine.'; stopReplay();
  });
  eventSource.addEventListener('error', () => {
    $('info').textContent = 'Connexion fermee.'; stopReplay();
  });
}

async function togglePause() {
  paused = !paused;
  $('pause').textContent = paused ? 'Reprendre' : 'Pause';
  if (paused) await fetch('/api/replay/pause', {method:'POST'});
  else {
    await fetch('/api/replay/resume', {method:'POST'});
    while (pendingCandles.length > 0) {
      candleSeries.update(pendingCandles.shift()); stats.candles++;
    }
    updateStats();
  }
}

async function stopReplay() {
  if (eventSource) { try{eventSource.close();}catch(e){} eventSource = null; }
  try { await fetch('/api/replay/stop', {method:'POST'}); } catch(e){}
  $('play').disabled = false; $('pause').disabled = true; $('stop').disabled = true;
  $('pause').textContent = 'Pause'; paused = false;
}

function init() {
  chart = LightweightCharts.createChart($('chart'), {
    layout:{background:{color:'#0d1117'}, textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#161b22'}, horzLines:{color:'#161b22'}},
    timeScale:{timeVisible:true, secondsVisible:false},
    rightPriceScale:{autoScale:true},
  });
  candleSeries = chart.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });
  $('play').addEventListener('click', startReplay);
  $('pause').addEventListener('click', togglePause);
  $('stop').addEventListener('click', stopReplay);
  updateStats();
}
init();
</script>
</body></html>
"""


# ============================================================
# LABEL_DAY HTML
# ============================================================

LABEL_DAY_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">Labeling — Jour entier</h1>
  <div class="muted" style="font-size:12px; margin-top:2px;">
    Tous les OB Vizion bruts du jour. <span style="color:#3fb950">VERT</span> = passe les filtres,
    <span style="color:#8b949e">GRIS</span> = rejete (tooltip = raison). Click pour valider/invalider.
    Mode "Dessiner mon OB" pour ajouter ce que Vizion rate.
  </div>
</header>

<div style="padding:12px 20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#161b22; border-bottom:1px solid #30363d;">
  <label class="muted">Date</label>
  <input id="date_pick" type="date" min="2025-06-11" max="2026-05-14" value="2025-08-15">
  <button id="load_day" class="primary">Charger ce jour</button>
  <button id="random_day">Jour aleatoire</button>
  <label class="muted" style="margin-left:20px">Mode</label>
  <select id="mode_day">
    <option value="label">Labeler OB existants</option>
    <option value="draw">Dessiner mon OB</option>
  </select>
  <select id="draw_dir_day" style="display:none">
    <option value="bullish">Bullish</option>
    <option value="bearish">Bearish</option>
  </select>
  <span id="period" class="muted"></span>
  <span id="info" class="muted" style="margin-left:20px"></span>
  <span class="spacer" style="flex:1"></span>
  <button id="diagnose">Pourquoi si peu d'OB ?</button>
  <button id="train_v4" class="success">Train v4 maintenant</button>
</div>

<div id="chart_day" style="width: calc(100% - 40px); height: 60vh; margin: 0 20px"></div>

<div id="diagnostic_panel" style="display:none; margin:12px 20px; padding:16px; background:#161b22; border:1px solid #d29922; border-radius:6px;">
  <h3 style="margin:0 0 10px 0; font-size:14px; color:#d29922;">Diagnostic : pourquoi seuls X OB passent ?</h3>
  <div id="diagnostic_content"></div>
</div>

<div style="padding:12px 20px;">
  <h3 style="margin:0 0 10px 0; font-size:14px;">Trades du jour - click pour labeller :</h3>
  <div id="trades_list"></div>
</div>

<style>
.trade-card { display:inline-block; background:#161b22; border:1px solid #30363d;
              border-radius:6px; padding:10px 14px; margin:6px; cursor:pointer;
              transition: all 0.15s; min-width:280px; }
.trade-card:hover { border-color:#58a6ff; }
.trade-card.valid { border-color:#3fb950; background:#0e2e15; }
.trade-card.invalid { border-color:#f85149; background:#3a1717; }
.trade-card.rejected { border-color:#444c56; background:#0d1117; opacity:0.7; }
.trade-card.rejected:hover { opacity:1; }
.rej-tag { background:#3a1717; color:#f0a8a3; padding:2px 6px; border-radius:3px;
           font-size:10px; margin-left:6px; cursor:help; }
.trade-card .ts { color:#8b949e; font-size:11px; font-family:monospace; }
.trade-card .dir { font-weight:600; }
.trade-card .dir.bullish { color:#3fb950; } .trade-card .dir.bearish { color:#f85149; }
.trade-card .meta { color:#8b949e; font-size:11px; margin-top:4px; }
.trade-card .actions { margin-top:8px; display:flex; gap:6px; }
.btn-mini { font-size:11px; padding:3px 8px; }
.outcome-mini { font-size:10px; padding:2px 6px; border-radius:3px; margin-left:6px; }
.outcome-WIN { background:#0e2e15; color:#3fb950; }
.outcome-LOSS { background:#3a1717; color:#f85149; }
.outcome-NO_FILL { background:#21262d; color:#8b949e; }
</style>

<script>
let chartD, candleD, currentTrades = [], obSeries = [], drawnSeries = [];
let drawingModeD = false, drawStateD = {step:0, point1:null};

function $(id) { return document.getElementById(id); }
function fmt(n,d) { return Number(n).toFixed(d === undefined ? 2 : d); }

function clearOBs() {
  for (const s of obSeries) { try{chartD.removeSeries(s);}catch(e){} }
  obSeries = [];
}

function drawTradeZoneD(t) {
  // Affiche entry/SL/TP comme dans replay live, mais sans outcome (pas simule pour rejetes)
  if (!t.entry || !t.stop_loss || !t.take_profit) return;
  const startT = new Date(t.ts).getTime() / 1000;
  const endT = startT + 60 * 60;  // 1h apres l'OB (defaut)

  // Couleurs : si rejete -> opacite reduite
  const opacityMult = t.verdict === 'REJECTED' ? 0.4 : 1.0;
  const labelOverride = t.existing_label;
  let tpAlpha = 0.20 * opacityMult, slAlpha = 0.20 * opacityMult;
  if (labelOverride === 'valid') { tpAlpha = 0.40; }
  else if (labelOverride === 'invalid') { slAlpha = 0.40; }

  // Outcome simule disponible si trade pris -> rendu plein
  if (t.outcome === 'WIN') { tpAlpha = 0.55; slAlpha = 0.08; }
  else if (t.outcome === 'LOSS') { slAlpha = 0.55; tpAlpha = 0.08; }

  // TP zone (vert)
  const tpFill = chartD.addBaselineSeries({
    baseValue: { type:'price', price: t.entry },
    topFillColor1: `rgba(63,185,80,${tpAlpha})`,
    topFillColor2: `rgba(63,185,80,${tpAlpha*0.4})`,
    topLineColor: `rgba(63,185,80,${0.7*opacityMult})`,
    bottomFillColor1: `rgba(63,185,80,${tpAlpha})`,
    bottomFillColor2: `rgba(63,185,80,${tpAlpha*0.4})`,
    bottomLineColor: `rgba(63,185,80,${0.7*opacityMult})`,
    lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });
  tpFill.setData([{time:startT, value:t.take_profit}, {time:endT, value:t.take_profit}]);
  obSeries.push(tpFill);

  // SL zone (rouge)
  const slFill = chartD.addBaselineSeries({
    baseValue: { type:'price', price: t.entry },
    topFillColor1: `rgba(248,81,73,${slAlpha})`,
    topFillColor2: `rgba(248,81,73,${slAlpha*0.4})`,
    topLineColor: `rgba(248,81,73,${0.7*opacityMult})`,
    bottomFillColor1: `rgba(248,81,73,${slAlpha})`,
    bottomFillColor2: `rgba(248,81,73,${slAlpha*0.4})`,
    bottomLineColor: `rgba(248,81,73,${0.7*opacityMult})`,
    lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });
  slFill.setData([{time:startT, value:t.stop_loss}, {time:endT, value:t.stop_loss}]);
  obSeries.push(slFill);

  // Ligne d'entry
  const entryColor = t.direction === 'bullish' ? `rgba(63,185,80,${opacityMult})` : `rgba(248,81,73,${opacityMult})`;
  const entryLine = chartD.addLineSeries({
    color: entryColor, lineWidth: 2,
    priceLineVisible: false, lastValueVisible: false,
  });
  entryLine.setData([{time:startT, value:t.entry}, {time:endT, value:t.entry}]);
  obSeries.push(entryLine);
}

function drawOBs(trades) {
  clearOBs();
  for (const t of trades) {
    drawTradeZoneD(t);
  }
  const markers = trades.map(t => {
    if (t.verdict === 'REJECTED') {
      return {
        time: new Date(t.ts).getTime() / 1000,
        position: t.direction === 'bullish' ? 'belowBar' : 'aboveBar',
        color: '#6e7681',
        shape: t.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
        text: `x ${t.rejection_bucket || 'rej'}`,
      };
    }
    return {
      time: new Date(t.ts).getTime() / 1000,
      position: t.direction === 'bullish' ? 'belowBar' : 'aboveBar',
      color: t.direction === 'bullish' ? '#3fb950' : '#f85149',
      shape: t.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
      text: `${t.direction === 'bullish' ? 'BUY' : 'SELL'} ml=${t.ml_proba}`,
    };
  });
  candleD.setMarkers(markers);
}

function drawManualRectD(time1, price1, time2, price2, direction) {
  const high = Math.max(price1, price2);
  const low = Math.min(price1, price2);
  const t1 = Math.min(time1, time2);
  const t2 = Math.max(time1, time2);
  const fillC = direction === 'bullish' ? 'rgba(217,255,0,0.25)' : 'rgba(255,140,0,0.25)';
  const lineC = direction === 'bullish' ? 'rgba(217,255,0,0.95)' : 'rgba(255,140,0,0.95)';
  const s = chartD.addBaselineSeries({
    baseValue: { type:'price', price: low },
    topFillColor1: fillC, topFillColor2: fillC, topLineColor: lineC,
    bottomFillColor1: fillC, bottomFillColor2: fillC, bottomLineColor: lineC,
    lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  });
  s.setData([{time:t1, value:high}, {time:t2, value:high}]);
  drawnSeries.push(s);
  return {high, low, t1, t2};
}

function renderTradesList(trades) {
  const list = $('trades_list');
  if (trades.length === 0) {
    list.innerHTML = '<div class="empty">Aucun OB Vizion brut sur ce jour.</div>';
    return;
  }
  // Separe acceptes / rejetes
  const accepted = trades.filter(t => t.verdict !== 'REJECTED');
  const rejected = trades.filter(t => t.verdict === 'REJECTED');

  const renderOne = (t, i) => {
    const time = t.ts.split('+')[0].replace('T', ' ').substring(11, 16);
    const cls = t.existing_label ? t.existing_label : (t.verdict === 'REJECTED' ? 'rejected' : '');
    const outcome = t.outcome ? `<span class="outcome-mini outcome-${t.outcome}">${t.outcome} ${t.pnl_usd >= 0 ? '+' : ''}${fmt(t.pnl_usd, 1)}$</span>` : '';
    const rejTag = t.verdict === 'REJECTED' ?
      `<span class="rej-tag" title="${t.rejection_reason || ''}">x ${t.rejection_bucket || 'rej'}</span>` : '';
    return `<div class="trade-card ${cls}" data-idx="${i}">
      <div class="ts">${time} - ${t.killzone || 'hors KZ'} ${rejTag}</div>
      <div class="dir ${t.direction}">${t.direction === 'bullish' ? 'BUY' : 'SELL'} @ ${fmt(t.entry, 3)}${outcome}</div>
      <div class="meta">Score: ${t.score}${t.verdict !== 'REJECTED' ? ' | RR: '+fmt(t.rr, 2)+' | ML: '+t.ml_proba : ''}${t.v4_proba !== null && t.v4_proba !== undefined ? ' | v4: '+t.v4_proba : ''}</div>
      <div class="actions">
        <button class="btn-mini success" onclick="labelTrade(${i}, 'valid')">Valide</button>
        <button class="btn-mini danger" onclick="labelTrade(${i}, 'invalid')">Invalide</button>
        <button class="btn-mini" onclick="labelTrade(${i}, 'skip')">Skip</button>
      </div>
    </div>`;
  };

  let html = '';
  if (accepted.length) {
    html += `<h4 style="margin:6px 0; font-size:13px; color:#3fb950;">${accepted.length} OB qui passent les filtres (vert) :</h4>`;
    html += accepted.map(t => renderOne(t, trades.indexOf(t))).join('');
  }
  if (rejected.length) {
    html += `<h4 style="margin:14px 0 6px; font-size:13px; color:#8b949e;">${rejected.length} OB rejetes (gris) - hover pour voir raison :</h4>`;
    html += rejected.map(t => renderOne(t, trades.indexOf(t))).join('');
  }
  list.innerHTML = html;
}

async function labelTrade(idx, label) {
  const t = currentTrades[idx];
  await fetch('/api/save_label', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ts: t.ts, direction: t.direction, label}),
  });
  t.existing_label = label;
  renderTradesList(currentTrades);
  fetch("/api/labels_summary").then(r=>r.json()).then(s => {
    document.getElementById("navSummary").textContent =
      `Labels : ${s.valid} valid | ${s.invalid} invalid | ${s.skip} skip (total ${s.total})`;
  });
}

async function loadDay(dateStr) {
  $('info').textContent = 'Chargement...';
  const url = dateStr === '__random__' ? '/api/label_day/random' : `/api/label_day/${dateStr}`;
  const data = await fetch(url).then(r => r.json());
  if (data.error) { $('info').textContent = 'Erreur: '+data.error; return; }
  $('period').textContent = `${data.start.substring(0,10)} -> ${data.end.substring(0,10)}`;
  $('info').textContent = `${data.trades.length} trades Vizion bruts, ${data.candles.length} bougies`;
  candleD.setData(data.candles);
  currentTrades = data.trades;
  drawOBs(data.trades);
  renderTradesList(data.trades);
}

async function trainV4() {
  $('info').textContent = 'Training v4...';
  const result = await fetch('/api/train_v4', {method:'POST'}).then(r=>r.json());
  if (result.error) { $('info').textContent = 'Erreur train : '+result.error; return; }
  $('info').textContent = `v4 entraine : ${result.n_labels} labels, AUC train=${fmt(result.train_auc, 3)}, CV=${result.cv_auc ? fmt(result.cv_auc, 3) : '-'}`;
}

function initDay() {
  chartD = LightweightCharts.createChart($('chart_day'), {
    layout:{background:{color:'#0d1117'}, textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#161b22'}, horzLines:{color:'#161b22'}},
    timeScale:{timeVisible:true, secondsVisible:false},
    rightPriceScale:{autoScale:true},
  });
  candleD = chartD.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });
  $('load_day').onclick = () => loadDay($('date_pick').value);
  $('random_day').onclick = () => loadDay('__random__');
  $('train_v4').onclick = trainV4;
  $('diagnose').onclick = diagnoseDay;

  $('mode_day').onchange = () => {
    drawingModeD = $('mode_day').value === 'draw';
    $('draw_dir_day').style.display = drawingModeD ? '' : 'none';
    drawStateD = {step:0, point1:null};
    $('info').textContent = drawingModeD
      ? 'Mode dessin actif - click 2 points sur le graphique (haut puis bas de TON OB)'
      : 'Mode label : click cards ou marker pour valider';
  };

  // Click handler chart : draw OB ou click marker pour cycle label
  chartD.subscribeClick(param => {
    if (!param.time || !param.point) return;
    if (drawingModeD) {
      const price = candleD.coordinateToPrice(param.point.y);
      if (drawStateD.step === 0) {
        drawStateD.point1 = {time: param.time, price};
        drawStateD.step = 1;
        $('info').textContent = '1er point pose. Click le 2eme point.';
      } else {
        const dir = $('draw_dir_day').value;
        const rect = drawManualRectD(drawStateD.point1.time, drawStateD.point1.price,
                                     param.time, price, dir);
        $('info').textContent = 'OB dessine. Saving...';
        fetch('/api/save_drawn_ob', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            start_ts: new Date(rect.t1 * 1000).toISOString(),
            end_ts: new Date(rect.t2 * 1000).toISOString(),
            ob_high: rect.high, ob_low: rect.low, direction: dir,
          }),
        }).then(r=>r.json()).then(res => {
          $('info').textContent = `OB sauve (total drawn: ${res.n_drawn}).`;
        });
        drawStateD = {step:0, point1:null};
      }
    }
  });

  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && drawingModeD) {
      drawStateD = {step:0, point1:null};
      $('info').textContent = 'Draw annule.';
    }
  });

  loadDay('__random__');
}

async function diagnoseDay() {
  const date = $('date_pick').value;
  $('diagnostic_panel').style.display = 'block';
  $('diagnostic_content').innerHTML = '<div class="muted">Analyse en cours (peut prendre 30s)...</div>';
  const data = await fetch(`/api/diagnose_day/${date}`).then(r => r.json());
  if (data.error) {
    $('diagnostic_content').innerHTML = '<div class="muted">Erreur: '+data.error+'</div>';
    return;
  }
  const total = data.n_obs_total;
  const ok = data.n_trade_ok;
  const rows = Object.entries(data.rejection_counts).map(([k, n]) => {
    const examples = (data.rejection_examples[k] || []).join(', ');
    const pct = (n / total * 100).toFixed(1);
    return `<tr>
      <td><code>${k}</code></td>
      <td style="text-align:right; color:#f85149;">${n}</td>
      <td style="text-align:right; color:#8b949e;">${pct}%</td>
      <td class="muted" style="font-size:11px">${examples}</td>
    </tr>`;
  }).join('');
  $('diagnostic_content').innerHTML = `
    <p style="margin:0 0 12px;">
      <strong>${total}</strong> OB detectes par Vizion (avant filtres).
      <strong style="color:#3fb950;">${ok}</strong> passent tous les filtres.
      <strong style="color:#f85149;">${total - ok}</strong> rejetes.
    </p>
    <table style="width:100%; border-collapse:collapse;">
      <thead><tr>
        <th>Filtre qui rejette</th><th style="text-align:right">N OB</th>
        <th style="text-align:right">%</th><th>Exemples</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="margin-top:12px; color:#8b949e; font-size:12px;">
      Le plus gros bucket = filtre a alleger en priorite. Mais attention : alleger = plus de faux signaux.
    </p>
  `;
}

initDay();
</script>
</body></html>
"""


# ============================================================
# LABEL_MONTH HTML
# ============================================================

LABEL_MONTH_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">Labeling — Mois entier + dessin OB manuel</h1>
  <div class="muted" style="font-size:12px; margin-top:2px;">
    OB Vizion bruts en gris. Highlight bleu = v4 pense que tu valideras. Dessine ta zone OB en cliquant 2 points.
  </div>
</header>

<div style="padding:12px 20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#161b22; border-bottom:1px solid #30363d;">
  <label class="muted">Mois (debut)</label>
  <input id="date_pick_m" type="date" min="2025-06-11" max="2026-04-15" value="2025-08-01">
  <button id="load_month" class="primary">Charger mois</button>
  <button id="random_month">Mois aleatoire</button>
  <label class="muted" style="margin-left:20px">Mode</label>
  <select id="mode">
    <option value="label">Labeler OB existants</option>
    <option value="draw">Dessiner mon OB</option>
  </select>
  <select id="draw_dir" style="display:none">
    <option value="bullish">Bullish</option>
    <option value="bearish">Bearish</option>
  </select>
  <span id="period_m" class="muted"></span>
  <span id="info_m" class="muted" style="margin-left:20px"></span>
</div>

<div id="chart_month" style="width: calc(100% - 40px); height: 65vh; margin: 0 20px"></div>

<div style="padding:12px 20px; color:#8b949e; font-size:12px;">
  <strong>Mode Label</strong> : click sur fleche pour cycler Valid → Invalid → Skip. <br>
  <strong>Mode Draw</strong> : click sur 2 points du graphique (haut puis bas du OB) pour dessiner. ESC pour annuler.
</div>

<script>
let chartM, candleM, currentTradesM = [], obSeriesM = [];
let drawingMode = false, drawState = {step: 0, point1: null};
let drawnRects = [];

function $(id) { return document.getElementById(id); }
function fmt(n,d) { return Number(n).toFixed(d===undefined?2:d); }

function clearOBsM() {
  for (const s of obSeriesM) { try{chartM.removeSeries(s);}catch(e){} }
  obSeriesM = [];
}

function colorForLabel(label) {
  if (label === 'valid') return {fill:'rgba(63,185,80,0.30)', line:'rgba(63,185,80,0.9)'};
  if (label === 'invalid') return {fill:'rgba(248,81,73,0.15)', line:'rgba(248,81,73,0.6)'};
  return null;
}

function drawOBsM(trades) {
  clearOBsM();
  for (const t of trades) {
    const startT = new Date(t.ob_start_ts).getTime() / 1000;
    const endT = new Date(t.ob_end_ts).getTime() / 1000 + 30 * 60;
    let fillC, lineC;
    if (t.existing_label) {
      const c = colorForLabel(t.existing_label) || {fill:'rgba(120,120,120,0.15)', line:'rgba(120,120,120,0.6)'};
      fillC = c.fill; lineC = c.line;
    } else if (t.v4_proba !== null && t.v4_proba >= 0.5) {
      fillC = 'rgba(88,166,255,0.25)'; lineC = 'rgba(88,166,255,0.9)';
    } else {
      fillC = t.direction === 'bullish' ? 'rgba(63,185,80,0.08)' : 'rgba(248,81,73,0.08)';
      lineC = t.direction === 'bullish' ? 'rgba(63,185,80,0.4)' : 'rgba(248,81,73,0.4)';
    }
    const s = chartM.addBaselineSeries({
      baseValue: { type:'price', price: t.ob_low },
      topFillColor1: fillC, topFillColor2: fillC, topLineColor: lineC,
      bottomFillColor1: fillC, bottomFillColor2: fillC, bottomLineColor: lineC,
      lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
    });
    s.setData([{time:startT, value:t.ob_high}, {time:endT, value:t.ob_high}]);
    obSeriesM.push(s);
  }
  const markers = trades.map(t => ({
    time: new Date(t.ts).getTime() / 1000,
    position: t.direction === 'bullish' ? 'belowBar' : 'aboveBar',
    color: t.v4_proba !== null && t.v4_proba >= 0.5 ? '#58a6ff'
           : (t.direction === 'bullish' ? '#3fb950' : '#f85149'),
    shape: t.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
    text: `${t.direction === 'bullish' ? 'BUY' : 'SELL'}${t.v4_proba !== null && t.v4_proba >= 0.5 ? ' BLEU' : ''}`,
  }));
  candleM.setMarkers(markers);
}

async function cycleLabel(t) {
  const order = ['valid', 'invalid', 'skip'];
  const cur = t.existing_label || '';
  const next = order[(order.indexOf(cur) + 1) % order.length];
  await fetch('/api/save_label', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ts: t.ts, direction: t.direction, label: next}),
  });
  t.existing_label = next;
  drawOBsM(currentTradesM);
  fetch("/api/labels_summary").then(r=>r.json()).then(s => {
    document.getElementById("navSummary").textContent =
      `Labels : ${s.valid} valid | ${s.invalid} invalid | ${s.skip} skip (total ${s.total})`;
  });
}

async function loadMonth(dateStr) {
  $('info_m').textContent = 'Chargement (peut etre lent)...';
  const url = dateStr === '__random__' ? '/api/label_month/random' : `/api/label_month/${dateStr}`;
  const data = await fetch(url).then(r=>r.json());
  if (data.error) { $('info_m').textContent = 'Erreur: '+data.error; return; }
  $('period_m').textContent = `${data.start.substring(0,10)} -> ${data.end.substring(0,10)}`;
  $('info_m').textContent = `${data.trades.length} trades Vizion bruts`;
  candleM.setData(data.candles);
  currentTradesM = data.trades;
  drawOBsM(data.trades);
}

function drawManualRect(time1, price1, time2, price2, direction) {
  const high = Math.max(price1, price2);
  const low = Math.min(price1, price2);
  const t1 = Math.min(time1, time2);
  const t2 = Math.max(time1, time2);
  const fillC = direction === 'bullish' ? 'rgba(217,255,0,0.25)' : 'rgba(255,140,0,0.25)';
  const lineC = direction === 'bullish' ? 'rgba(217,255,0,0.95)' : 'rgba(255,140,0,0.95)';
  const s = chartM.addBaselineSeries({
    baseValue: { type:'price', price: low },
    topFillColor1: fillC, topFillColor2: fillC, topLineColor: lineC,
    bottomFillColor1: fillC, bottomFillColor2: fillC, bottomLineColor: lineC,
    lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  });
  s.setData([{time:t1, value:high}, {time:t2, value:high}]);
  drawnRects.push(s);
  return {high, low, t1, t2};
}

function initMonth() {
  chartM = LightweightCharts.createChart($('chart_month'), {
    layout:{background:{color:'#0d1117'}, textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#161b22'}, horzLines:{color:'#161b22'}},
    timeScale:{timeVisible:true, secondsVisible:false},
    rightPriceScale:{autoScale:true},
  });
  candleM = chartM.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });

  $('load_month').onclick = () => loadMonth($('date_pick_m').value);
  $('random_month').onclick = () => loadMonth('__random__');
  $('mode').onchange = () => {
    drawingMode = $('mode').value === 'draw';
    $('draw_dir').style.display = drawingMode ? '' : 'none';
    drawState = {step:0, point1:null};
    $('info_m').textContent = drawingMode ? 'Click 2 points pour dessiner OB' : '';
  };

  // Click handler for label cycle ET draw
  chartM.subscribeClick(param => {
    if (!param.time || !param.point) return;
    if (!drawingMode) {
      // Mode label : trouver le trade le plus proche en temps
      const clickT = param.time;
      let closest = null, minDelta = Infinity;
      for (const t of currentTradesM) {
        const tT = new Date(t.ts).getTime() / 1000;
        const d = Math.abs(tT - clickT);
        if (d < minDelta) { minDelta = d; closest = t; }
      }
      if (closest && minDelta < 30 * 60) {
        cycleLabel(closest);
      }
    } else {
      // Mode draw : 2 points
      const price = candleM.coordinateToPrice(param.point.y);
      if (drawState.step === 0) {
        drawState.point1 = {time: param.time, price};
        drawState.step = 1;
        $('info_m').textContent = '1er point pose. Click le 2eme.';
      } else {
        const dir = $('draw_dir').value;
        const rect = drawManualRect(drawState.point1.time, drawState.point1.price,
                                     param.time, price, dir);
        $('info_m').textContent = 'OB dessine. Saving...';
        fetch('/api/save_drawn_ob', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            start_ts: new Date(rect.t1 * 1000).toISOString(),
            end_ts: new Date(rect.t2 * 1000).toISOString(),
            ob_high: rect.high, ob_low: rect.low, direction: dir,
          }),
        }).then(r=>r.json()).then(res => {
          $('info_m').textContent = `OB sauvegarde (total ${res.n_drawn}).`;
        });
        drawState = {step:0, point1:null};
      }
    }
  });

  // ESC pour cancel draw
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && drawingMode) {
      drawState = {step:0, point1:null};
      $('info_m').textContent = 'Draw annule.';
    }
  });

  loadMonth('__random__');
}
initMonth();
</script>
</body></html>
"""


# ============================================================
# EXPLAIN HTML
# ============================================================

EXPLAIN_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">Explain Trade — Diagnostic d'un setup</h1>
  <div class="muted" style="font-size:12px; margin-top:2px;">
    Donne timestamp + direction d'un setup que TU vois mais que Vizion rate. On debug pourquoi.
  </div>
</header>

<div style="padding:20px;">
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:20px;">
    <label class="muted">Timestamp (UTC)</label>
    <input id="ex_ts" type="datetime-local" value="2025-08-15T14:30" style="width:180px">
    <label class="muted">Direction</label>
    <select id="ex_dir">
      <option value="bullish">Bullish (BUY)</option>
      <option value="bearish">Bearish (SELL)</option>
    </select>
    <button id="ex_analyze" class="primary">Analyser</button>
  </div>

  <div id="ex_chart" style="width:100%; height:50vh; margin-bottom:20px;"></div>

  <div id="ex_result" style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:16px;">
    <div class="empty">Analyse non lancee.</div>
  </div>
</div>

<script>
let chartE, candleE;
function $(id) { return document.getElementById(id); }

async function analyzeTrade() {
  const ts = $('ex_ts').value + ':00';  // input datetime-local sans seconds
  const dir = $('ex_dir').value;
  $('ex_result').innerHTML = '<div class="empty">Analyse en cours...</div>';
  const url = `/api/explain?ts=${encodeURIComponent(ts)}&direction=${dir}`;
  const data = await fetch(url).then(r=>r.json());
  if (data.error) { $('ex_result').innerHTML = '<div class="empty">Erreur: '+data.error+'</div>'; return; }
  candleE.setData(data.candles);
  const rows = data.diagnostics.map(d => `
    <tr><td>${d.swing_strength}</td><td>${d.n_swings}</td>
        <td>${d.n_ob_total}</td><td>${d.n_ob_matching_direction}</td></tr>
  `).join('');
  $('ex_result').innerHTML = `
    <h3 style="margin:0 0 10px;">Diagnostic Vizion pour ${ts} ${dir}</h3>
    <table style="width:auto; border-collapse:collapse;">
      <thead><tr>
        <th>Swing strength</th><th>N swings</th><th>N OB total</th><th>N OB matching dir</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="margin-top:14px; color:#8b949e;">${data.advice}</p>
  `;
}

function initExplain() {
  chartE = LightweightCharts.createChart($('ex_chart'), {
    layout:{background:{color:'#0d1117'}, textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#161b22'}, horzLines:{color:'#161b22'}},
    timeScale:{timeVisible:true, secondsVisible:false},
  });
  candleE = chartE.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });
  $('ex_analyze').addEventListener('click', analyzeTrade);
}
initExplain();
</script>
</body></html>
"""


ANALYZE_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">Analyser mes OB dessines</h1>
  <div class="muted" style="font-size:12px; margin-top:2px;">
    Pour chaque zone que TU as dessinee sur /label_month, on identifie pourquoi Vizion l'a (ou non) prise.
    Le rapport propose des ajustements de config pour matcher TON style.
  </div>
</header>

<div style="padding:20px;">
  <button id="run_analyze" class="primary">Analyser mes OB dessines</button>
  <span id="analyze_info" class="muted" style="margin-left:20px"></span>

  <div id="summary" style="margin-top:20px;"></div>
  <div id="suggestions" style="margin-top:20px;"></div>
  <div id="rows_table" style="margin-top:20px;"></div>
</div>

<style>
.summary-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px; }
.summary-card { background:#161b22; border:1px solid #30363d; border-radius:6px; padding:14px; }
.summary-card .label { color:#8b949e; font-size:11px; text-transform:uppercase; }
.summary-card .value { font-size:24px; font-weight:600; margin-top:4px; }
.summary-card.info .value { color:#58a6ff; }
.summary-card.good .value { color:#3fb950; }
.summary-card.warn .value { color:#d29922; }
.summary-card.bad .value { color:#f85149; }
.bucket-row { background:#161b22; border:1px solid #30363d; border-radius:6px;
              padding:10px 14px; margin-bottom:6px; display:flex; align-items:center; gap:12px; }
.bucket-name { flex:1; font-family:monospace; }
.bucket-count { font-weight:600; color:#f85149; }
.suggestion-card { background:#161b22; border-left:3px solid #d29922; padding:14px;
                   margin-bottom:10px; border-radius:6px; }
.suggestion-card.high { border-left-color:#f85149; }
.suggestion-card.medium { border-left-color:#d29922; }
.suggestion-card.low { border-left-color:#58a6ff; }
.suggestion-card .prio { font-size:11px; text-transform:uppercase; font-weight:600; }
.suggestion-card.high .prio { color:#f85149; }
.suggestion-card.medium .prio { color:#d29922; }
.suggestion-card.low .prio { color:#58a6ff; }
.suggestion-card .issue { margin:6px 0; font-size:13px; }
.suggestion-card .action { color:#3fb950; font-size:13px; }
.suggestion-card code { background:#0d1117; padding:2px 6px; border-radius:3px; font-size:12px; }

.zones-table { width:100%; border-collapse:collapse; background:#161b22; border:1px solid #30363d;
               border-radius:6px; overflow:hidden; }
.zones-table th, .zones-table td { padding:7px 10px; border-bottom:1px solid #21262d;
                                   font-size:12px; text-align:left; font-family:monospace; }
.zones-table th { background:#0d1117; color:#8b949e; font-family:inherit;
                  text-transform:uppercase; font-size:10px; letter-spacing:0.5px; }
</style>

<script>
function $(id) { return document.getElementById(id); }

async function runAnalyze() {
  $('analyze_info').textContent = 'Analyse en cours (peut prendre 30-60s)...';
  $('summary').innerHTML = ''; $('suggestions').innerHTML = ''; $('rows_table').innerHTML = '';
  const data = await fetch('/api/analyze_drawn_obs').then(r => r.json());
  if (data.error) {
    $('analyze_info').textContent = data.error;
    return;
  }
  $('analyze_info').textContent = '';

  // Summary cards
  const ratio_detected = data.n_drawn > 0 ? Math.round(data.n_matched_vizion / data.n_drawn * 100) : 0;
  const ratio_accepted = data.n_drawn > 0 ? Math.round(data.n_currently_accepted / data.n_drawn * 100) : 0;
  $('summary').innerHTML = `
    <div class="summary-grid">
      <div class="summary-card info">
        <div class="label">Zones dessinees</div>
        <div class="value">${data.n_drawn}</div>
      </div>
      <div class="summary-card warn">
        <div class="label">Matche un OB Vizion</div>
        <div class="value">${data.n_matched_vizion} <span style="font-size:14px; color:#8b949e">(${ratio_detected}%)</span></div>
      </div>
      <div class="summary-card good">
        <div class="label">Pris par le bot actuel</div>
        <div class="value">${data.n_currently_accepted} <span style="font-size:14px; color:#8b949e">(${ratio_accepted}%)</span></div>
      </div>
      <div class="summary-card bad">
        <div class="label">Pas detecte du tout</div>
        <div class="value">${data.n_not_detected_by_vizion}</div>
      </div>
    </div>

    <h3 style="margin:10px 0 8px; font-size:14px;">Raisons de rejet :</h3>
    ${Object.entries(data.rejection_buckets).map(([k, n]) => `
      <div class="bucket-row">
        <div class="bucket-name">${k}</div>
        <div class="bucket-count">${n}</div>
      </div>
    `).join('')}

    <h3 style="margin:18px 0 8px; font-size:14px;">Distribution :</h3>
    <div style="display:flex; gap:30px; color:#8b949e; font-size:13px;">
      <div><strong>Sessions :</strong> ${Object.entries(data.session_distribution).map(([k,n]) => `${k}=${n}`).join(', ')}</div>
      <div><strong>Directions :</strong> ${Object.entries(data.direction_distribution).map(([k,n]) => `${k}=${n}`).join(', ')}</div>
    </div>
  `;

  // Suggestions
  if (data.suggestions && data.suggestions.length > 0) {
    $('suggestions').innerHTML = `
      <h3 style="margin:0 0 12px; font-size:14px;">Ajustements proposes (ordre de priorite) :</h3>
      ${data.suggestions.map(s => `
        <div class="suggestion-card ${s.priority}">
          <div class="prio">${s.priority}</div>
          <div class="issue">${s.issue}</div>
          <div class="action">${s.action}</div>
          <div style="margin-top:6px; font-size:12px; color:#8b949e;">
            <code>${s.param}</code> : <code>${s.current}</code> &rarr; <code>${s.suggested}</code>
          </div>
        </div>
      `).join('')}
    `;
  } else {
    $('suggestions').innerHTML = '<div class="muted" style="padding:20px; text-align:center;">Aucune suggestion (tous tes setups passent deja Vizion).</div>';
  }

  // Rows table
  const rows = data.rows.map(r => {
    const ts = r.start_ts.replace('T', ' ').substring(0, 16);
    const dirCls = r.direction === 'bullish' ? 'good' : 'bad';
    const verdict = r.pipeline_verdict || '-';
    const verdictCls = verdict === 'TRADE' ? 'good' : 'bad';
    return `
      <tr>
        <td>${ts}</td>
        <td style="color: ${dirCls === 'good' ? '#3fb950' : '#f85149'}; font-weight:600">${r.direction === 'bullish' ? 'BUY' : 'SELL'}</td>
        <td>${r.ob_low.toFixed(3)} - ${r.ob_high.toFixed(3)}</td>
        <td style="color: ${verdictCls === 'good' ? '#3fb950' : '#f85149'}">${verdict}</td>
        <td><code>${r.rejection_bucket || '-'}</code></td>
        <td>${r.killzone || '-'}</td>
        <td>${r.matched_ob_score !== null ? r.matched_ob_score : '-'}</td>
      </tr>
    `;
  }).join('');
  $('rows_table').innerHTML = `
    <h3 style="margin:0 0 8px; font-size:14px;">Detail par zone :</h3>
    <table class="zones-table">
      <thead><tr>
        <th>Time</th><th>Dir</th><th>Zone</th><th>Verdict</th>
        <th>Bucket</th><th>KZ</th><th>Score Vizion</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

$('run_analyze').addEventListener('click', runAnalyze);
</script>
</body></html>
"""


LABEL_FAST_HTML = """
<header style="padding:12px 20px; background:#161b22; border-bottom:1px solid #30363d;">
  <h1 style="margin:0; font-size:18px;">🚀 Label Fast — Trader Review</h1>
  <div class="muted" style="font-size:12px; margin-top:2px;">
    <kbd>P</kbd>=PASS (OK) · <kbd>B</kbd>=BUG puis <kbd>1-9</kbd> categorie · <kbd>←/→</kbd>=Nav · <kbd>D</kbd>=Dessin · <kbd>Esc</kbd>=Annule
  </div>
</header>

<div style="padding:12px 20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#161b22; border-bottom:1px solid #30363d;">
  <label class="muted">Date</label>
  <input id="lf_date" type="date" min="2025-06-11" max="2026-05-14" value="2025-08-15">
  <button id="lf_load" class="primary">Charger</button>
  <button id="lf_random">Jour aleatoire</button>
  <span class="spacer" style="flex:1"></span>
  <span id="lf_progress" style="font-size:14px; font-weight:600;"></span>
  <button id="lf_next_day" class="success" style="margin-left:10px">Jour suivant →</button>
</div>

<div id="lf_main" style="display:grid; grid-template-columns: 1fr 320px; gap:0; min-height:calc(100vh - 120px);">
  <div id="lf_chart_wrap" style="position:relative; background:#0d1117;">
    <div id="lf_chart" style="width:100%; height:100%;"></div>
    <div id="lf_overlay" style="position:absolute; bottom:20px; left:20px; right:20px;
         background: rgba(13,17,23,0.9); border:1px solid #30363d; border-radius:6px;
         padding:14px 18px; pointer-events:none; backdrop-filter: blur(6px);"></div>
  </div>
  <aside style="background:#161b22; border-left:1px solid #30363d; padding:20px;
                display:flex; flex-direction:column; gap:14px;">
    <div id="lf_info_panel">
      <div class="muted" style="font-size:11px;">Chargement...</div>
    </div>

    <div style="border-top:1px solid #30363d; padding-top:14px;">
      <h4 style="margin:0 0 10px; font-size:13px;">Decision</h4>
      <div style="display:flex; flex-direction:column; gap:8px;">
        <button id="lf_btn_bug" class="danger" style="padding:18px; font-size:14px;">
          <span style="display:block; font-weight:600;">B — BUG</span>
          <span style="display:block; font-size:11px; opacity:0.8;">le bot fait n'importe quoi ici</span>
        </button>
        <button id="lf_btn_pass" class="success" style="padding:18px; font-size:14px;">
          <span style="display:block; font-weight:600;">P — PASS</span>
          <span style="display:block; font-size:11px; opacity:0.8;">OK rien a dire</span>
        </button>
      </div>

      <!-- Sous-menu categorie de bug : cache par defaut, affiche quand B est cliquee -->
      <div id="lf_bug_menu" style="display:none; margin-top:10px; padding:10px;
           background:#1a0e0f; border:1px solid #6e2727; border-radius:6px;">
        <div style="font-size:12px; color:#f0a8a3; margin-bottom:8px; font-weight:600;">
          Pourquoi c'est un bug ? (1-9 ou clic)
        </div>
        <div style="display:flex; flex-direction:column; gap:4px;">
          <button class="bug-cat-btn" data-cat="pas_liquidite" data-key="1">
            <kbd>1</kbd> Pas de prise de liquidite avant
          </button>
          <button class="bug-cat-btn" data-cat="groupe_1_bougie" data-key="2">
            <kbd>2</kbd> 1 seule bougie (groupe trop petit)
          </button>
          <button class="bug-cat-btn" data-cat="pas_fvg_imbalance" data-key="3">
            <kbd>3</kbd> Pas de FVG / imbalance
          </button>
          <button class="bug-cat-btn" data-cat="sl_trop_loin" data-key="4">
            <kbd>4</kbd> SL trop loin (rr ecrase)
          </button>
          <button class="bug-cat-btn" data-cat="sl_trop_serre" data-key="5">
            <kbd>5</kbd> SL trop serre (stop hunt evident)
          </button>
          <button class="bug-cat-btn" data-cat="tp_trop_loin" data-key="6">
            <kbd>6</kbd> TP trop loin (jamais atteint)
          </button>
          <button class="bug-cat-btn" data-cat="contre_tendance" data-key="7">
            <kbd>7</kbd> Contre la tendance evidente
          </button>
          <button class="bug-cat-btn" data-cat="zone_pas_comblee" data-key="8">
            <kbd>8</kbd> Zone OB pas vraiment comblee
          </button>
          <button class="bug-cat-btn" data-cat="ob_inverse_breaker" data-key="9">
            <kbd>9</kbd> OB inverse (devrait etre un breaker)
          </button>
          <button class="bug-cat-btn" data-cat="pas_un_ob" data-key="0">
            <kbd>0</kbd> C'est meme pas un OB du tout
          </button>
          <button class="bug-cat-btn" data-cat="autre" data-key="a">
            <kbd>A</kbd> Autre (precise plus tard)
          </button>
          <button class="bug-cat-btn cancel-bug" data-cat="" data-key="Esc">
            <kbd>Esc</kbd> Annuler
          </button>
        </div>
      </div>
    </div>

    <div style="border-top:1px solid #30363d; padding-top:14px;">
      <h4 style="margin:0 0 8px; font-size:13px;">Navigation</h4>
      <div style="display:flex; gap:6px;">
        <button id="lf_btn_prev" style="flex:1;">← Prev</button>
        <button id="lf_btn_next" style="flex:1;">Next →</button>
      </div>
    </div>

    <div style="border-top:1px solid #30363d; padding-top:14px;">
      <h4 style="margin:0 0 8px; font-size:13px;">A la fin du jour</h4>
      <button id="lf_btn_draw" class="warn" style="width:100%; padding:10px;">
        D — Dessiner des OB manquants
      </button>
      <div id="lf_draw_panel" style="display:none; margin-top:10px;">
        <select id="lf_draw_dir" style="width:100%;">
          <option value="bullish">Bullish</option>
          <option value="bearish">Bearish</option>
        </select>
        <div class="muted" style="font-size:11px; margin-top:6px;">
          Click 2 points sur le chart. <kbd>Esc</kbd> pour annuler.
        </div>
        <div id="lf_draw_count" class="muted" style="font-size:11px; margin-top:6px;"></div>
      </div>
    </div>

    <div style="border-top:1px solid #30363d; padding-top:14px;">
      <h4 style="margin:0 0 8px; font-size:13px; display:flex; justify-content:space-between;">
        <span>Bugs - top categories</span>
        <button id="lf_refresh_stats" style="font-size:11px; padding:3px 8px;">↻</button>
      </h4>
      <div id="lf_bug_stats" class="muted" style="font-size:11px;">
        Aucun bug labellise.
      </div>
    </div>
  </aside>
</div>

<style>
kbd { background:#21262d; border:1px solid #444c56; padding:1px 6px; border-radius:3px;
      font-family:monospace; font-size:11px; color:#c9d1d9; }
#lf_overlay h2 { margin:0 0 6px; font-size:18px; }
#lf_overlay .dir-bull { color:#3fb950; }
#lf_overlay .dir-bear { color:#f85149; }
#lf_overlay .meta { color:#8b949e; font-size:12px; }
#lf_overlay .rej { color:#f0a8a3; font-size:11px; margin-top:4px; }
.info-row { display:flex; justify-content:space-between; padding:5px 0;
            border-bottom:1px solid #21262d; font-size:13px; }
.info-row .l { color:#8b949e; }
.info-row .v { font-weight:600; font-family:monospace; }
.info-row.valid .v { color:#3fb950; }
.info-row.invalid .v { color:#f85149; }
.flash-valid { animation: flashGreen 0.4s; }
.flash-invalid { animation: flashRed 0.4s; }
@keyframes flashGreen { 0%,100%{background:#0d1117;} 50%{background:#0e2e15;} }
@keyframes flashRed   { 0%,100%{background:#0d1117;} 50%{background:#3a1717;} }
.bug-cat-btn { background:#2d1818; color:#f0a8a3; border:1px solid #6e2727;
               padding:8px 10px; border-radius:4px; cursor:pointer; text-align:left;
               font-family:inherit; font-size:12px; display:flex; align-items:center; gap:8px; }
.bug-cat-btn:hover { background:#6e2727; color:white; }
.bug-cat-btn kbd { min-width:18px; text-align:center; }
.bug-cat-btn.cancel-bug { background:#21262d; color:#8b949e; border-color:#444c56; }
.bug-cat-btn.cancel-bug:hover { background:#30363d; }
</style>

<script>
let chartLF, candleLF, lfState = {
  trades: [], currentIdx: 0, candles: [],
  drawing: false, drawStep: 0, drawPoint1: null,
  obSeries: [], drawnRects: [],
};

function $(id) { return document.getElementById(id); }
function fmt(n,d) { return Number(n).toFixed(d===undefined?2:d); }

function clearOBSeries() {
  for (const s of lfState.obSeries) { try{chartLF.removeSeries(s);}catch(e){} }
  lfState.obSeries = [];
}

function focusOnCurrentOB() {
  // Zoom auto autour de l'OB courant : -1h avant, +1h apres
  const t = lfState.trades[lfState.currentIdx];
  if (!t) return;
  const center = new Date(t.ts).getTime() / 1000;
  const range = chartLF.timeScale();
  range.setVisibleRange({from: center - 3600, to: center + 3600});
}

function renderCurrent() {
  clearOBSeries();
  const idx = lfState.currentIdx;
  const total = lfState.trades.length;
  $('lf_progress').textContent = `OB ${idx + 1} / ${total}`;
  if (total === 0) {
    $('lf_info_panel').innerHTML = '<div class="muted">Aucun OB ce jour. Click "D" pour dessiner manuellement, ou jour suivant.</div>';
    $('lf_overlay').innerHTML = '<div class="muted">Aucun OB Vizion ce jour.</div>';
    return;
  }
  if (idx >= total) {
    $('lf_info_panel').innerHTML = `<div style="color:#3fb950; font-weight:600;">✓ Jour termine !</div>
      <div class="muted" style="margin-top:8px; font-size:12px;">Tu peux dessiner des OB que Vizion a rate (D), ou passer au jour suivant.</div>`;
    $('lf_overlay').innerHTML = '<h2 style="color:#3fb950;">Tous les OB labellises !</h2><div class="meta">Press D pour dessiner ou clic Jour suivant.</div>';
    return;
  }
  const t = lfState.trades[idx];
  const time = t.ts.split('+')[0].replace('T', ' ').substring(11, 16);
  const dirCls = t.direction === 'bullish' ? 'bull' : 'bear';
  const dirLbl = t.direction === 'bullish' ? 'BUY' : 'SELL';
  const rejInfo = t.verdict === 'REJECTED'
    ? `<div class="rej">x rejete par : ${t.rejection_bucket || 'inconnu'}</div>`
    : '';
  const outcomeInfo = t.outcome
    ? ` <span style="color:${t.outcome === 'WIN' ? '#3fb950' : '#f85149'}; font-size:12px;">[${t.outcome} ${t.pnl_usd >= 0 ? '+' : ''}${fmt(t.pnl_usd, 1)}$]</span>`
    : '';

  $('lf_overlay').innerHTML = `
    <h2 class="dir-${dirCls}">${dirLbl} @ ${fmt(t.entry, 3)} — ${time}${outcomeInfo}</h2>
    <div class="meta">
      ${t.killzone || 'hors KZ'} | Score: ${t.score} | RR: ${fmt(t.rr, 2)} | ML: ${t.ml_proba}
    </div>
    ${rejInfo}
  `;

  // Panel infos detaillees
  const lblClass = t.existing_label === 'pass' ? 'valid' : (t.existing_label === 'bug' ? 'invalid' : '');
  $('lf_info_panel').innerHTML = `
    <h4 style="margin:0 0 10px; font-size:13px;">OB #${idx + 1}/${total}</h4>
    <div class="info-row"><span class="l">Heure</span><span class="v">${time}</span></div>
    <div class="info-row"><span class="l">Direction</span><span class="v" style="color:${t.direction === 'bullish' ? '#3fb950' : '#f85149'}">${dirLbl}</span></div>
    <div class="info-row"><span class="l">Entry</span><span class="v">${fmt(t.entry, 3)}</span></div>
    <div class="info-row"><span class="l">SL</span><span class="v" style="color:#f85149">${fmt(t.stop_loss, 3)}</span></div>
    <div class="info-row"><span class="l">TP</span><span class="v" style="color:#3fb950">${fmt(t.take_profit, 3)}</span></div>
    <div class="info-row"><span class="l">RR</span><span class="v">${fmt(t.rr, 2)}</span></div>
    <div class="info-row"><span class="l">Score</span><span class="v">${t.score}</span></div>
    <div class="info-row"><span class="l">KZ</span><span class="v">${t.killzone || '-'}</span></div>
    <div class="info-row"><span class="l">ML proba</span><span class="v">${t.ml_proba}</span></div>
    ${t.outcome ? `<div class="info-row"><span class="l">Outcome</span><span class="v" style="color:${t.outcome === 'WIN' ? '#3fb950' : '#f85149'}">${t.outcome} ${t.pnl_usd >= 0 ? '+' : ''}${fmt(t.pnl_usd, 2)}$</span></div>` : ''}
    ${t.verdict === 'REJECTED' ? `<div class="info-row"><span class="l">Rejete par</span><span class="v" style="color:#f0a8a3; font-size:11px;">${t.rejection_bucket}</span></div>` : ''}
    ${t.existing_label ? `<div class="info-row ${lblClass}"><span class="l">Ton label</span><span class="v">${t.existing_label.toUpperCase()}${t.bug_category ? ' — ' + t.bug_category : ''}</span></div>` : ''}
  `;

  // Dessine SL/TP/Entry sur le chart
  const startT = new Date(t.ts).getTime() / 1000;
  const endT = startT + 60 * 60;
  let tpAlpha = 0.20, slAlpha = 0.20;
  if (t.outcome === 'WIN') { tpAlpha = 0.55; slAlpha = 0.08; }
  else if (t.outcome === 'LOSS') { slAlpha = 0.55; tpAlpha = 0.08; }
  else if (t.existing_label === 'valid') { tpAlpha = 0.45; }
  else if (t.existing_label === 'invalid') { slAlpha = 0.45; }

  const tpFill = chartLF.addBaselineSeries({
    baseValue: { type:'price', price: t.entry },
    topFillColor1: `rgba(63,185,80,${tpAlpha})`, topFillColor2: `rgba(63,185,80,${tpAlpha*0.4})`,
    topLineColor: 'rgba(63,185,80,0.8)',
    bottomFillColor1: `rgba(63,185,80,${tpAlpha})`, bottomFillColor2: `rgba(63,185,80,${tpAlpha*0.4})`,
    bottomLineColor: 'rgba(63,185,80,0.8)',
    lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });
  tpFill.setData([{time:startT, value:t.take_profit}, {time:endT, value:t.take_profit}]);
  lfState.obSeries.push(tpFill);

  const slFill = chartLF.addBaselineSeries({
    baseValue: { type:'price', price: t.entry },
    topFillColor1: `rgba(248,81,73,${slAlpha})`, topFillColor2: `rgba(248,81,73,${slAlpha*0.4})`,
    topLineColor: 'rgba(248,81,73,0.8)',
    bottomFillColor1: `rgba(248,81,73,${slAlpha})`, bottomFillColor2: `rgba(248,81,73,${slAlpha*0.4})`,
    bottomLineColor: 'rgba(248,81,73,0.8)',
    lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
  });
  slFill.setData([{time:startT, value:t.stop_loss}, {time:endT, value:t.stop_loss}]);
  lfState.obSeries.push(slFill);

  const entryColor = t.direction === 'bullish' ? '#3fb950' : '#f85149';
  const entryLine = chartLF.addLineSeries({
    color: entryColor, lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  });
  entryLine.setData([{time:startT, value:t.entry}, {time:endT, value:t.entry}]);
  lfState.obSeries.push(entryLine);

  candleLF.setMarkers([{
    time: startT,
    position: t.direction === 'bullish' ? 'belowBar' : 'aboveBar',
    color: entryColor,
    shape: t.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
    text: `${dirLbl} ${idx+1}/${total}`,
  }]);

  focusOnCurrentOB();
}

async function saveLabelAndNext(label, bugCategory = null) {
  const t = lfState.trades[lfState.currentIdx];
  if (!t) return;
  // Flash visuel
  const wrap = $('lf_main');
  wrap.classList.remove('flash-valid', 'flash-invalid');
  if (label === 'pass') wrap.classList.add('flash-valid');
  else if (label === 'bug') wrap.classList.add('flash-invalid');

  await fetch('/api/save_label', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      ts: t.ts, direction: t.direction, label,
      bug_category: bugCategory,
    }),
  });
  t.existing_label = label;
  t.bug_category = bugCategory;
  // Update nav summary
  fetch("/api/labels_summary").then(r=>r.json()).then(s => {
    document.getElementById("navSummary").textContent =
      `Reviews : ${s.bug || 0} BUG | ${s.pass || 0} PASS (total ${s.total})`;
  });
  // Refresh stats bugs (sidebar)
  if (label === 'bug') refreshBugStats();
  // Cache le menu si ouvert
  $('lf_bug_menu').style.display = 'none';
  lfState.bugMenuOpen = false;
  // Auto-next apres 150ms
  setTimeout(() => {
    if (lfState.currentIdx < lfState.trades.length) lfState.currentIdx++;
    renderCurrent();
  }, 150);
}

function showBugMenu() {
  $('lf_bug_menu').style.display = 'block';
  lfState.bugMenuOpen = true;
}

function hideBugMenu() {
  $('lf_bug_menu').style.display = 'none';
  lfState.bugMenuOpen = false;
}

async function labelCurrent(label) {
  if (label === 'pass') {
    await saveLabelAndNext('pass');
  } else if (label === 'bug') {
    // Affiche le sous-menu au lieu de sauver direct
    showBugMenu();
  }
}

function navPrev() {
  if (lfState.currentIdx > 0) {
    lfState.currentIdx--;
    renderCurrent();
  }
}

function navNext() {
  if (lfState.currentIdx < lfState.trades.length) {
    lfState.currentIdx++;
    renderCurrent();
  }
}

function toggleDrawMode() {
  lfState.drawing = !lfState.drawing;
  $('lf_draw_panel').style.display = lfState.drawing ? 'block' : 'none';
  if (lfState.drawing) {
    $('lf_btn_draw').textContent = 'D — Sortir mode dessin';
    $('lf_btn_draw').classList.remove('warn');
    $('lf_btn_draw').classList.add('primary');
  } else {
    $('lf_btn_draw').textContent = 'D — Dessiner des OB manquants';
    $('lf_btn_draw').classList.remove('primary');
    $('lf_btn_draw').classList.add('warn');
    lfState.drawStep = 0; lfState.drawPoint1 = null;
  }
}

async function loadDay(dateStr) {
  $('lf_progress').textContent = 'Chargement...';
  const url = dateStr === '__random__' ? '/api/label_day/random' : `/api/label_day/${dateStr}`;
  const data = await fetch(url).then(r=>r.json());
  if (data.error) { $('lf_progress').textContent = 'Erreur'; return; }
  lfState.trades = data.trades;
  lfState.currentIdx = 0;
  lfState.candles = data.candles;
  // Trier par timestamp
  lfState.trades.sort((a,b) => new Date(a.ts) - new Date(b.ts));
  candleLF.setData(data.candles);
  // Update date input
  if (data.start) $('lf_date').value = data.start.substring(0, 10);
  renderCurrent();
}

async function refreshBugStats() {
  const data = await fetch('/api/bug_categories_stats').then(r => r.json());
  const el = $('lf_bug_stats');
  if (!data.total_bugs) {
    el.textContent = 'Aucun bug labellise.';
    return;
  }
  const total = data.total_bugs;
  const rows = Object.entries(data.by_category).map(([cat, n]) => {
    const pct = Math.round(n / total * 100);
    return `<div style="display:flex; justify-content:space-between; padding:2px 0;">
      <span style="color:#f0a8a3;">${cat}</span>
      <span style="font-family:monospace;">${n} <span style="color:#8b949e;">(${pct}%)</span></span>
    </div>`;
  }).join('');
  el.innerHTML = `<div style="font-weight:600; color:#f0a8a3; margin-bottom:4px;">Total: ${total} bugs</div>` + rows;
}

async function nextDay() {
  // Jour suivant calendrier
  const current = $('lf_date').value;
  if (current) {
    const d = new Date(current);
    d.setDate(d.getDate() + 1);
    const next = d.toISOString().substring(0, 10);
    $('lf_date').value = next;
    loadDay(next);
  } else {
    loadDay('__random__');
  }
}

function initLF() {
  chartLF = LightweightCharts.createChart($('lf_chart'), {
    layout:{background:{color:'#0d1117'}, textColor:'#c9d1d9'},
    grid:{vertLines:{color:'#161b22'}, horzLines:{color:'#161b22'}},
    timeScale:{timeVisible:true, secondsVisible:false},
    rightPriceScale:{autoScale:true},
  });
  candleLF = chartLF.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });

  $('lf_load').onclick = () => loadDay($('lf_date').value);
  $('lf_random').onclick = () => loadDay('__random__');
  $('lf_next_day').onclick = nextDay;
  $('lf_btn_bug').onclick = () => labelCurrent('bug');
  $('lf_btn_pass').onclick = () => labelCurrent('pass');
  $('lf_btn_prev').onclick = navPrev;
  $('lf_btn_next').onclick = navNext;
  $('lf_btn_draw').onclick = toggleDrawMode;
  $('lf_refresh_stats').onclick = refreshBugStats;

  // Click handler chart pour dessin OB
  chartLF.subscribeClick(param => {
    if (!lfState.drawing || !param.time || !param.point) return;
    const price = candleLF.coordinateToPrice(param.point.y);
    if (lfState.drawStep === 0) {
      lfState.drawPoint1 = {time: param.time, price};
      lfState.drawStep = 1;
      $('lf_draw_count').textContent = '1er point pose. Click le 2eme.';
    } else {
      const dir = $('lf_draw_dir').value;
      const t1 = Math.min(lfState.drawPoint1.time, param.time);
      const t2 = Math.max(lfState.drawPoint1.time, param.time);
      const high = Math.max(lfState.drawPoint1.price, price);
      const low = Math.min(lfState.drawPoint1.price, price);
      const fillC = dir === 'bullish' ? 'rgba(217,255,0,0.30)' : 'rgba(255,140,0,0.30)';
      const lineC = dir === 'bullish' ? 'rgba(217,255,0,0.95)' : 'rgba(255,140,0,0.95)';
      const s = chartLF.addBaselineSeries({
        baseValue: { type:'price', price: low },
        topFillColor1: fillC, topFillColor2: fillC, topLineColor: lineC,
        bottomFillColor1: fillC, bottomFillColor2: fillC, bottomLineColor: lineC,
        lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
      });
      s.setData([{time:t1, value:high}, {time:t2, value:high}]);
      lfState.drawnRects.push(s);
      fetch('/api/save_drawn_ob', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          start_ts: new Date(t1 * 1000).toISOString(),
          end_ts: new Date(t2 * 1000).toISOString(),
          ob_high: high, ob_low: low, direction: dir,
        }),
      }).then(r=>r.json()).then(res => {
        $('lf_draw_count').textContent = `OB dessine et sauve (total ${res.n_drawn}).`;
      });
      lfState.drawStep = 0; lfState.drawPoint1 = null;
    }
  });

  // Mapping touche -> categorie de bug (ordre des boutons)
  const BUG_CAT_BY_KEY = {
    '1': 'pas_liquidite',
    '2': 'groupe_1_bougie',
    '3': 'pas_fvg_imbalance',
    '4': 'sl_trop_loin',
    '5': 'sl_trop_serre',
    '6': 'tp_trop_loin',
    '7': 'contre_tendance',
    '8': 'zone_pas_comblee',
    '9': 'ob_inverse_breaker',
    '0': 'pas_un_ob',
    'a': 'autre', 'A': 'autre',
  };

  // Raccourcis clavier
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    // Si menu bug ouvert : 1-9 = categorie, Esc = annule
    if (lfState.bugMenuOpen) {
      if (BUG_CAT_BY_KEY[e.key]) {
        e.preventDefault();
        saveLabelAndNext('bug', BUG_CAT_BY_KEY[e.key]);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        hideBugMenu();
      }
      return;
    }

    if (e.key === 'b' || e.key === 'B') { e.preventDefault(); labelCurrent('bug'); }
    else if (e.key === 'p' || e.key === 'P') { e.preventDefault(); labelCurrent('pass'); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); navPrev(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); navNext(); }
    else if (e.key === 'd' || e.key === 'D') { e.preventDefault(); toggleDrawMode(); }
    else if (e.key === 'Escape' && lfState.drawing) {
      lfState.drawStep = 0; lfState.drawPoint1 = null;
      $('lf_draw_count').textContent = 'Draw cancel.';
    }
  });

  // Click handler pour les boutons de categorie de bug
  document.querySelectorAll('.bug-cat-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const cat = btn.dataset.cat;
      if (!cat) {
        hideBugMenu();
        return;
      }
      saveLabelAndNext('bug', cat);
    });
  });

  loadDay($('lf_date').value);
  refreshBugStats();
}
initLF();
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    print("Dashboard ML : http://127.0.0.1:8002", flush=True)
    print("  /             : Replay Live OOS", flush=True)
    print("  /label_fast   : Labeling RAPIDE 1 OB par 1 OB (recommande)", flush=True)
    print("  /label_day    : Labeling par jour", flush=True)
    print("  /label_month  : Labeling par mois + dessin OB manuel", flush=True)
    print("  /analyze      : Analyse de TES OB dessines + suggestions config", flush=True)
    print("  /explain_trade: Explain setup", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
