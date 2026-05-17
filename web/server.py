"""Serveur FastAPI - mode entrainement humain."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select

from bot.training import next_training_trade
from config import HOST, PORT, INSTRUMENTS
from data.fetch import load_cached
from db.models import BotTuning, HumanLabeledTrade, SessionLocal, TrainingTrade, ValidatedSetup, init_db

WEB_ROOT = Path(__file__).parent
STATIC_DIR = WEB_ROOT / "static"

app = FastAPI(title="TradingBot Trainer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ====== Training endpoints ======

@app.get("/api/training/list")
def list_sessions() -> list[dict]:
    """Liste tous les trades de session training (ordre chrono de review)."""
    with SessionLocal() as s:
        rows = s.execute(
            select(TrainingTrade).order_by(TrainingTrade.session_index)
        ).scalars().all()
        return [{
            "id": r.id,
            "session_index": r.session_index,
            "sampled_day": r.sampled_day.isoformat(),
            "direction": r.direction,
            "score": r.score,
            "algo_verdict": r.algo_verdict,
            "would_trade": r.would_trade,
            "reviewed": r.reviewed_at is not None,
            "status": r.status,
        } for r in rows]


@app.post("/api/training/next")
def training_next() -> dict:
    """Pioche un nouveau jour aleatoire + scanne + retourne le meilleur trade."""
    record, day, skipped = next_training_trade()
    if record is None:
        raise HTTPException(404, f"Aucun trade trouve apres {skipped} jours testes")
    return _trade_to_dict(record, skipped=skipped)


@app.get("/api/training/{trade_id}")
def get_training_trade(trade_id: str) -> dict:
    """Recupere un trade specifique (pour navigation prev/next dans la liste)."""
    with SessionLocal() as s:
        r = s.get(TrainingTrade, trade_id)
        if r is None:
            raise HTTPException(404, "Trade introuvable")
        return _trade_to_dict(r)


class FeedbackIn(BaseModel):
    comment: str = ""
    user_drawings: dict[str, Any] = {}


@app.post("/api/training/{trade_id}/feedback")
def submit_feedback(trade_id: str, fb: FeedbackIn) -> dict:
    """Enregistre le feedback user pour un trade."""
    with SessionLocal() as s:
        r = s.get(TrainingTrade, trade_id)
        if r is None:
            raise HTTPException(404, "Trade introuvable")
        r.user_comment = fb.comment
        r.user_drawings = fb.user_drawings
        r.reviewed_at = datetime.utcnow()
        s.commit()
        return {"ok": True}


@app.get("/api/training/export/markdown", response_class=PlainTextResponse)
def export_markdown() -> str:
    """Exporte toutes les sessions reviewees en markdown (pour partager au dev)."""
    with SessionLocal() as s:
        rows = s.execute(
            select(TrainingTrade).order_by(TrainingTrade.session_index)
        ).scalars().all()

        out = ["# Training session log\n",
               f"_Genere le {datetime.utcnow().isoformat()}_\n",
               f"_Total trades : {len(rows)}_\n",
               f"_Reviewes : {sum(1 for r in rows if r.reviewed_at)}_\n\n",
               "---\n"]

        for r in rows:
            out.append(f"\n## Trade #{r.session_index} - {r.direction.upper()} {r.instrument}\n")
            out.append(f"- **Jour pioche** : {r.sampled_day.date()}\n")
            out.append(f"- **Entree** : {r.entry_time} @ {r.entry_price}\n")
            out.append(f"- **SL** : {r.stop_loss} | **TP** : {r.take_profit} | **RR** : {r.risk_reward:.2f}\n")
            out.append(f"- **Score bot** : {r.score}/100 ({r.algo_verdict})\n")
            out.append(f"- **Bot aurait trade** : {'OUI' if r.would_trade else 'NON'}\n")
            out.append(f"- **Outcome simule** : {r.status} (PnL: {r.pnl:.2f}$)\n")
            out.append(f"- **HTF strength** : {r.htf_strength} | **OB type** : {r.ob_type} | **Touch** : {r.touch_count}\n\n")

            sa = r.setup_analysis or {}
            if sa.get("hard_filters"):
                out.append("### Filtres durs\n")
                for hf in sa["hard_filters"]:
                    icon = "OK" if hf.get("passed") else "KO"
                    out.append(f"- [{icon}] {hf.get('label')} - _{hf.get('detail','')}_\n")

            if sa.get("factors"):
                out.append("\n### Facteurs vus par le bot\n")
                by_cat: dict[str, list] = {}
                for f in sa["factors"]:
                    by_cat.setdefault(f.get("category", "misc"), []).append(f)
                for cat, fs in by_cat.items():
                    out.append(f"\n**{cat}**\n")
                    for f in fs:
                        sign = "+" if f.get("applied_score", 0) >= 0 else ""
                        out.append(f"- [{f.get('status','?')}] {f.get('label')} ({sign}{f.get('applied_score',0)}) - _{f.get('detail','')}_\n")

            out.append("\n### Feedback user\n")
            if r.reviewed_at:
                out.append(f"_{r.reviewed_at}_\n\n")
                out.append(f"{r.user_comment or '_(pas de commentaire)_'}\n")
                if r.user_drawings:
                    counts = {
                        "rects": len(r.user_drawings.get("rects", [])),
                        "lines": len(r.user_drawings.get("lines", [])),
                        "arrows": len(r.user_drawings.get("arrows", [])),
                    }
                    out.append(f"\n_Dessins : {counts['rects']} zones, {counts['lines']} lignes, {counts['arrows']} fleches_\n")
                    out.append("```json\n")
                    import json
                    out.append(json.dumps(r.user_drawings, indent=2, default=str))
                    out.append("\n```\n")
            else:
                out.append("_(pas encore reviewé)_\n")

            out.append("\n---\n")

        return "".join(out)


# ====== Candles endpoint (lecture chart) ======

@app.get("/api/candles/{tf}")
def candles(tf: str, around: str | None = None, span: int | None = None) -> list[dict]:
    """Retourne les bougies d'un timeframe avec zoom adapte au TF.

    Args:
        tf: M1, M5, M15, M30, H1, H4.
        around: timestamp ISO -> on centre la vue.
        span: si fourni override, sinon span par defaut adapte au TF.
    """
    try:
        df = load_cached(tf)
    except FileNotFoundError:
        raise HTTPException(404, f"Pas de cache pour {tf}")

    # Span adapte par TF (avant / apres l'entree)
    # On affiche genereusement APRES l'entree pour voir comment ca s'est fini
    default_spans = {
        "M1":  (120, 80),
        "M5":  (100, 60),
        "M15": (80,  50),
        "M30": (70,  40),
        "H1":  (60,  40),
        "H4":  (50,  30),
    }
    before, after = default_spans.get(tf, (100, 60))
    if span is not None:
        before, after = span, span // 2

    if around:
        try:
            ts = pd.Timestamp(around)
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            raise HTTPException(400, "around invalide")
        try:
            loc = df.index.get_indexer([ts], method="nearest")[0]
        except Exception:
            raise HTTPException(404, "Timestamp hors range")
        start = max(0, loc - before)
        end = min(len(df), loc + after)
        df = df.iloc[start:end]
    else:
        df = df.iloc[-(before + after):]

    return [{
        "time": int(ts.timestamp()),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    } for ts, row in df.iterrows()]


# ====== Helpers ======

def _trade_to_dict(r: TrainingTrade, skipped: int = 0) -> dict:
    return {
        "id": r.id,
        "session_index": r.session_index,
        "sampled_day": r.sampled_day.isoformat(),
        "skipped_empty_days": skipped,
        "direction": r.direction,
        "instrument": r.instrument,
        "entry_time": r.entry_time.isoformat(),
        "entry_price": r.entry_price,
        "stop_loss": r.stop_loss,
        "take_profit": r.take_profit,
        "risk_reward": r.risk_reward,
        "lot_size": r.lot_size,
        "status": r.status,
        "exit_time": r.exit_time.isoformat() if r.exit_time else None,
        "exit_price": r.exit_price,
        "pnl": r.pnl,
        "score": r.score,
        "algo_verdict": r.algo_verdict,
        "would_trade": r.would_trade,
        "htf_strength": r.htf_strength,
        "ob_type": r.ob_type,
        "touch_count": r.touch_count,
        "notes": r.notes,
        "chart_overlays": r.chart_overlays or {},
        "setup_analysis": r.setup_analysis or {},
        "user_comment": r.user_comment or "",
        "user_drawings": r.user_drawings or {"rects": [], "lines": [], "arrows": []},
        "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
    }


# ====== MODE HUMAN LABELING ======

INSTRUMENT_ORDER = list(INSTRUMENTS.keys())


@app.get("/labeling")
def labeling_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "labeling.html")


@app.get("/api/labeling/available-days")
def labeling_available_days(instrument: str = "XAUUSD") -> list[str]:
    """Liste des jours dispos pour un actif (jours ouvres uniquement)."""
    try:
        df = load_cached(instrument, "M1")
    except FileNotFoundError:
        return []
    days = sorted({d.date() for d in df.index})
    return [str(d) for d in days if d.weekday() < 5]


def _score_sweep_wick_ob(ob, df, instrument):
    """Scoring normalise du sweep-wick OB (0-100), independant du prix de l'actif."""
    from bot.killzones import current_killzone
    import math

    score = 0
    # 1. Push impulsivity : ratio push / ATR
    if ob.push_end_index >= 14:
        recent = df.iloc[ob.push_end_index - 14:ob.push_end_index]
        atr = float((recent["high"] - recent["low"]).mean() or 1)
    else:
        atr = 1.0
    push_atr = ob.push_strength / atr if atr > 0 else 0
    # push_atr=2 -> 15pts, push_atr=5 -> 30pts
    score += min(30, int(push_atr * 7))

    # 2. Nb bougies du push (idealement 2-5)
    n_cdls = ob.push_end_index - ob.push_start_index + 1
    if 2 <= n_cdls <= 5:
        score += 15
    elif n_cdls <= 7:
        score += 8

    # 3. Killzone
    kz = current_killzone(ob.push_end_time)
    if kz:
        score += kz.weight   # entre 8 et 20
    # else: 0 (pas malus mais pas bonus)

    # 4. Distance sweep (plus c'est marque, mieux c'est)
    if ob.swept_swing is not None:
        if ob.direction == "bearish":
            sweep_dist = ob.zone_high - ob.swept_level
        else:
            sweep_dist = ob.swept_level - ob.zone_low
        # Normalise par ATR
        sweep_atr = abs(sweep_dist) / atr if atr > 0 else 0
        score += min(15, int(sweep_atr * 8))

    # 5. Day of week
    wd = ob.push_end_time.weekday()
    if wd in (1, 2, 3):    # mardi-mercredi-jeudi
        score += 5

    return min(100, max(0, score))


@app.get("/api/labeling/algo-trades")
def labeling_algo_trades(instrument: str, day: str) -> list[dict]:
    """Retourne tous les sweep-wick OBs detectes pour un jour/actif donne.

    Utilise le nouveau detecteur sweep_wick_ob qui capte les OB que le user trace.
    """
    from bot.detectors.swings import detect_swings
    from bot.detectors.sweep_wick_ob import detect_sweep_wick_obs

    try:
        target = pd.Timestamp(day).tz_localize("UTC")
    except Exception:
        raise HTTPException(400, "format jour invalide")

    try:
        df = load_cached(instrument, "M1")
    except FileNotFoundError:
        return []

    end = target + timedelta(days=1)
    sub = df.loc[target:end]
    if sub.empty:
        return []

    swings = detect_swings(sub, left=3, right=3)
    obs = detect_sweep_wick_obs(sub, swings)

    result = []
    for i, ob in enumerate(obs):
        # Construit entry/SL/TP suggeres
        entry = ob.zone_mid
        if ob.direction == "bullish":
            sl = ob.zone_low - ob.zone_size * 0.15
            # TP : prochaine liquidite haute visible
            tp = ob.zone_high + ob.zone_size * 2
        else:
            sl = ob.zone_high + ob.zone_size * 0.15
            tp = ob.zone_low - ob.zone_size * 2
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0

        result.append({
            "id": f"swo_{i}",
            "direction": ob.direction,
            "entry_price": round(entry, 4),
            "stop_loss": round(sl, 4),
            "take_profit": round(tp, 4),
            "risk_reward": round(rr, 2),
            "score": _score_sweep_wick_ob(ob, sub, instrument),
            "verdict": "sweep_wick_ob",
            "would_trade": True,
            "outcome": "pending",
            "ob_zone": {
                "high": ob.zone_high,
                "low": ob.zone_low,
                "from_time": ob.push_start_time.isoformat(),
            },
            "entry_time": ob.push_end_time.isoformat(),
        })
    return result


@app.get("/api/labeling/candle-at")
def labeling_candle_at(instrument: str, tf: str, timestamp: int) -> dict:
    """Retourne les details OHLC d'une bougie precise (closest match au timestamp Unix)."""
    try:
        df = load_cached(instrument, tf)
    except FileNotFoundError:
        raise HTTPException(404, f"Pas de cache pour {instrument} {tf}")

    ts = pd.Timestamp(timestamp, unit="s", tz="UTC")
    try:
        loc = df.index.get_indexer([ts], method="nearest")[0]
    except Exception:
        raise HTTPException(404, "Bougie hors range")

    row = df.iloc[loc]
    actual_time = df.index[loc]
    return {
        "time": int(actual_time.timestamp()),
        "iso": actual_time.isoformat(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
        "body_top": float(max(row["open"], row["close"])),
        "body_bottom": float(min(row["open"], row["close"])),
        "is_bullish": bool(row["close"] > row["open"]),
    }


@app.get("/api/labeling/random-day")
def labeling_random_day(instrument: str = "XAUUSD") -> dict:
    """Pioche un jour random (jour ouvre) pour un actif."""
    days = labeling_available_days(instrument)
    if not days:
        raise HTTPException(404, f"Pas de donnees pour {instrument}")
    idx = secrets.randbelow(len(days))
    return {"instrument": instrument, "day": days[idx], "all_days": days}


@app.get("/api/labeling/candles")
def labeling_candles(
    instrument: str,
    day: str,
    tf: str = "M1",
) -> list[dict]:
    """Bougies d'un actif pour une journee complete (24h) sur un TF donne.

    Pour H4/H1/M30/M15, on elargit la fenetre pour avoir du contexte.
    """
    try:
        df = load_cached(instrument, tf)
    except FileNotFoundError:
        raise HTTPException(404, f"Pas de cache pour {instrument} {tf}")

    try:
        target_day = pd.Timestamp(day).tz_localize("UTC").normalize()
    except Exception:
        raise HTTPException(400, "Format de jour invalide")

    # Fenetres elargies selon TF (plus de contexte pour les grands TFs)
    spans = {
        "M1":  (timedelta(hours=4), timedelta(hours=24)),   # 4h avant + 24h
        "M5":  (timedelta(hours=12), timedelta(hours=36)),  # plus large
        "M15": (timedelta(days=2),  timedelta(days=2)),
        "M30": (timedelta(days=3),  timedelta(days=2)),
        "H1":  (timedelta(days=10), timedelta(days=3)),
        "H4":  (timedelta(days=30), timedelta(days=7)),
    }
    before, after = spans.get(tf, (timedelta(hours=4), timedelta(hours=24)))
    start = target_day - before
    end = target_day + timedelta(days=1) + after

    df = df.loc[start:end]
    return [{
        "time": int(ts.timestamp()),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    } for ts, row in df.iterrows()]


class HumanTradeIn(BaseModel):
    instrument: str
    day: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str | None = None     # ISO timestamp de l'entry (optionnel mais important)
    ob_zone: dict[str, Any] = {}
    ob_candles: dict[str, Any] = {}
    extra_drawings: dict[str, Any] = {}
    reasoning: str = ""


@app.post("/api/labeling/save-trade")
def save_human_trade(payload: HumanTradeIn) -> dict:
    """Sauvegarde un trade label humain + calcul outcome auto."""
    if payload.direction not in ("bullish", "bearish"):
        raise HTTPException(400, "direction invalide")
    if payload.instrument not in INSTRUMENTS:
        raise HTTPException(400, "instrument inconnu")

    risk = abs(payload.entry_price - payload.stop_loss)
    reward = abs(payload.take_profit - payload.entry_price)
    rr = reward / risk if risk > 0 else 0.0

    # Outcome auto en regardant les bougies APRES le jour
    outcome, exit_time, exit_price = _compute_outcome(
        payload.instrument, payload.day, payload.direction,
        payload.entry_price, payload.stop_loss, payload.take_profit,
        payload.entry_time,
    )

    with SessionLocal() as s:
        rec = HumanLabeledTrade(
            instrument=payload.instrument,
            day=pd.Timestamp(payload.day).to_pydatetime().replace(tzinfo=None),
            direction=payload.direction,
            entry_price=payload.entry_price,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            risk_reward=rr,
            ob_zone=payload.ob_zone,
            ob_candles=payload.ob_candles,
            extra_drawings=payload.extra_drawings,
            reasoning=payload.reasoning,
            outcome=outcome,
            exit_time=exit_time.replace(tzinfo=None) if exit_time else None,
            exit_price=exit_price,
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return {
            "id": rec.id,
            "outcome": rec.outcome,
            "exit_time": rec.exit_time.isoformat() if rec.exit_time else None,
            "exit_price": rec.exit_price,
            "rr": rec.risk_reward,
        }


def _compute_outcome(instrument, day_str, direction, entry, sl, tp, entry_time_iso=None):
    """Verifie si SL ou TP touche en premier APRES l'entree (pas depuis minuit).

    Si entry_time fourni : on demarre la simulation a ce moment-la.
    Sinon : on essaie de trouver la 1ere bougie qui matche le prix d'entree.
    """
    try:
        df = load_cached(instrument, "M1")

        # Demarrage de la simulation : entry_time si fourni, sinon trouve auto
        if entry_time_iso:
            start_ts = pd.Timestamp(entry_time_iso)
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize("UTC")
        else:
            # Fallback : trouve la 1ere bougie du jour qui a passe par entry_price
            target_day = pd.Timestamp(day_str).tz_localize("UTC")
            day_end = target_day + timedelta(days=1)
            day_data = df.loc[target_day:day_end]
            start_ts = None
            for ts, row in day_data.iterrows():
                if row["low"] <= entry <= row["high"]:
                    start_ts = ts
                    break
            if start_ts is None:
                # Pas touche le jour-meme, demarre debut journee
                start_ts = target_day

        end = start_ts + timedelta(days=2)
        sub = df.loc[start_ts:end]
        # On skip la premiere bougie (celle de l'entree) car le prix l'a deja touchee
        for i, (ts, row) in enumerate(sub.iterrows()):
            if i == 0:
                continue
            if direction == "bullish":
                if row["low"] <= sl:
                    return "loss", ts, sl
                if row["high"] >= tp:
                    return "win", ts, tp
            else:
                if row["high"] >= sl:
                    return "loss", ts, sl
                if row["low"] <= tp:
                    return "win", ts, tp
    except Exception as e:
        print(f"_compute_outcome error: {e}")
    return "unknown", None, None


@app.get("/api/labeling/trades")
def list_human_trades(instrument: str | None = None, day: str | None = None) -> list[dict]:
    """Liste les trades labels (filtre optionnel par instrument/jour)."""
    with SessionLocal() as s:
        q = select(HumanLabeledTrade).order_by(HumanLabeledTrade.created_at.desc())
        if instrument:
            q = q.where(HumanLabeledTrade.instrument == instrument)
        if day:
            try:
                d = pd.Timestamp(day).to_pydatetime().replace(tzinfo=None)
                q = q.where(HumanLabeledTrade.day == d)
            except Exception:
                pass
        rows = s.execute(q).scalars().all()
        return [{
            "id": r.id,
            "instrument": r.instrument,
            "day": r.day.isoformat(),
            "direction": r.direction,
            "entry_price": r.entry_price,
            "stop_loss": r.stop_loss,
            "take_profit": r.take_profit,
            "risk_reward": r.risk_reward,
            "ob_zone": r.ob_zone,
            "ob_candles": r.ob_candles,
            "extra_drawings": r.extra_drawings,
            "reasoning": r.reasoning,
            "outcome": r.outcome,
            "exit_time": r.exit_time.isoformat() if r.exit_time else None,
            "exit_price": r.exit_price,
        } for r in rows]


@app.delete("/api/labeling/trades/{trade_id}")
def delete_human_trade(trade_id: int) -> dict:
    with SessionLocal() as s:
        rec = s.get(HumanLabeledTrade, trade_id)
        if not rec:
            raise HTTPException(404, "Trade introuvable")
        s.delete(rec)
        s.commit()
        return {"ok": True}


@app.get("/api/labeling/export/markdown", response_class=PlainTextResponse)
def export_labels_markdown() -> str:
    """Exporte tous les trades labels en markdown massif pour calibration."""
    with SessionLocal() as s:
        rows = s.execute(
            select(HumanLabeledTrade).order_by(
                HumanLabeledTrade.instrument, HumanLabeledTrade.day,
                HumanLabeledTrade.created_at
            )
        ).scalars().all()

    out = ["# Dataset humain labelise pour calibration\n",
           f"_Genere le {datetime.utcnow().isoformat()}_\n",
           f"_Total : {len(rows)} trades_\n\n"]

    # Stats globales
    by_outcome: dict[str, int] = {}
    by_instrument: dict[str, int] = {}
    for r in rows:
        by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        by_instrument[r.instrument] = by_instrument.get(r.instrument, 0) + 1

    out.append("## Stats\n\n")
    out.append("**Par outcome** : " + ", ".join(f"{k}={v}" for k, v in by_outcome.items()) + "\n\n")
    out.append("**Par actif** : " + ", ".join(f"{k}={v}" for k, v in by_instrument.items()) + "\n\n")
    out.append("---\n\n")

    current_inst = None
    current_day = None
    for r in rows:
        if r.instrument != current_inst:
            current_inst = r.instrument
            current_day = None
            out.append(f"\n# {r.instrument}\n\n")
        day_key = r.day.date()
        if day_key != current_day:
            current_day = day_key
            out.append(f"\n## {day_key} ({day_key.strftime('%A')})\n\n")

        out.append(f"### Trade #{r.id} - {r.direction.upper()}\n")
        out.append(f"- **Entry** : {r.entry_price}\n")
        out.append(f"- **SL** : {r.stop_loss}\n")
        out.append(f"- **TP** : {r.take_profit}\n")
        out.append(f"- **RR** : {r.risk_reward:.2f}\n")
        out.append(f"- **Outcome** : {r.outcome}")
        if r.exit_time:
            out.append(f" @ {r.exit_time} (prix {r.exit_price})\n")
        else:
            out.append("\n")
        if r.ob_zone:
            ob = r.ob_zone
            out.append(f"- **OB zone** : [{ob.get('low')} -> {ob.get('high')}]")
            if ob.get('t1'):
                out.append(f" entre {ob.get('t1')} et {ob.get('t2')}")
            out.append("\n")
        if r.reasoning:
            out.append(f"\n**Raisonnement :**\n> {r.reasoning}\n")
        out.append("\n")

    return "".join(out)


# ====== MODE VALIDATION RAPIDE (Y/N) ======

@app.get("/validate")
def validate_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "validate.html")


@app.get("/api/validate/next")
def validate_next(instrument: str | None = None) -> dict:
    """Pioche un setup aleatoire a valider.

    MODE REALISTE : le bot ne voit que les bougies jusqu'au moment du push
    + un buffer 5 bougies post pour confirmer le retour dans la zone.
    Le chart affiche ces bougies + un peu de futur masque pour reveal apres validation.
    """
    import secrets
    from datetime import datetime, timezone, timedelta
    from bot.detectors.swings import detect_swings
    from bot.detectors.sweep_wick_ob import detect_sweep_wick_obs
    from bot.retrain import load_current_params

    # On ne tire que les actifs tradables (pas les smt_only comme DXY/XAG/SPX/UKOIL)
    tradable = [k for k, v in INSTRUMENTS.items() if not v.get("smt_only")]
    inst = instrument or secrets.choice(tradable)
    if INSTRUMENTS.get(inst, {}).get("smt_only"):
        raise HTTPException(400, f"{inst} est SMT-only, pas tradable")
    params = load_current_params()
    DEFAULT_PARAMS_KZ = ["asia", "london", "ny", "london_close"]

    # Liste des already-validated
    with SessionLocal() as s:
        from sqlalchemy import select
        existing = s.execute(
            select(ValidatedSetup.instrument, ValidatedSetup.entry_time)
            .where(ValidatedSetup.instrument == inst)
        ).all()
        seen = {(e[0], e[1].replace(tzinfo=None)) for e in existing}

    try:
        df = load_cached(inst, "M1")
    except FileNotFoundError:
        raise HTTPException(404, f"Pas de donnees pour {inst}")

    days = sorted({d.date() for d in df.index if d.weekday() < 5})
    if not days:
        raise HTTPException(404, "Pas de jours dispo")

    # Essaye 50 fois de trouver un setup non-deja-valide
    for _ in range(50):
        day = days[secrets.randbelow(len(days))]
        day_ts = pd.Timestamp(day).tz_localize("UTC")
        end = day_ts + timedelta(days=1)
        sub = df.loc[day_ts:end]
        if sub.empty:
            continue
        swings = detect_swings(sub, left=3, right=3)
        obs = detect_sweep_wick_obs(sub, swings)
        if not obs:
            continue
        # Filtre ceux deja vus + filtres dynamiques (killzones / heures)
        from bot.killzones import current_killzone
        candidates = []
        for ob in obs:
            entry_dt = ob.push_end_time.replace(tzinfo=None) if hasattr(ob.push_end_time, 'replace') else ob.push_end_time
            key = (inst, entry_dt)
            if key in seen:
                continue

            # Filtre killzones
            kz = current_killzone(ob.push_end_time)
            kz_name = kz.name if kz else "none"
            allowed_kz = params.get("allowed_killzones", DEFAULT_PARAMS_KZ)
            if params.get("killzone_required") and kz is None:
                continue
            if kz_name != "none" and kz_name not in allowed_kz:
                continue

            # Filtre heures
            hour = ob.push_end_time.hour
            allowed_hours = params.get("allowed_hours", list(range(24)))
            if hour not in allowed_hours:
                continue

            # FILTRE STRICT : trade aligne avec le trend HTF
            from bot.detectors.trend import analyze_trend, trade_aligned_with_trend
            from bot.detectors.fvg import detect_fvgs, find_fvg_in_zone

            # PAS DE FILTRE TREND HTF DUR : nos setups (sweep + OB + comblement)
            # sont DES MSS par definition. Un MSS est un retournement contre la tendance,
            # donc rejeter "contre tendance" rejetterait justement les bons setups.
            # Le contexte HTF est CAPTURE par compute_alignment (score HTF visible).
            try:
                df_m15 = load_cached(inst, "M15")
                m15_until_trade = df_m15.loc[df_m15.index <= ob.push_end_time]
            except FileNotFoundError:
                m15_until_trade = None

            # FILTRE BONUS : FVG M15 dans la zone OB = confluence forte
            try:
                if 'df_m15' in dir():
                    fvgs_m15 = detect_fvgs(m15_until_trade.tail(100))
                    fvg_m15 = find_fvg_in_zone(fvgs_m15, ob.zone_high, ob.zone_low, ob.direction, len(m15_until_trade) - 1)
                    ob._has_fvg_m15 = fvg_m15 is not None and not fvg_m15.filled
            except Exception:
                ob._has_fvg_m15 = False

            # CONFIRMATION SMT : verifie divergence avec actifs correles
            from bot.detectors.smt import check_smt_divergence
            from config import SMT_PAIRS
            ob._smt_results = []
            ob._smt_confirmed = False
            for corr_inst, corr_type in SMT_PAIRS.get(inst, []):
                try:
                    corr_df = load_cached(corr_inst, "M1")
                    smt_res = check_smt_divergence(
                        sub, corr_df,
                        ob.push_end_time,
                        ob.direction,
                        correlation_type=corr_type,
                        lookback_minutes=60,
                    )
                    ob._smt_results.append({
                        "correlate": corr_inst,
                        "type": corr_type,
                        "has_divergence": smt_res.has_divergence,
                        "detail": smt_res.detail,
                    })
                    if smt_res.has_divergence:
                        ob._smt_confirmed = True
                except FileNotFoundError:
                    pass

            candidates.append(ob)
        if not candidates:
            continue
        # Pioche random parmi les candidates
        ob = candidates[secrets.randbelow(len(candidates))]

        # Calcul SL/TP : SL = sous/au-dessus OB, TP = prochaine liquidite MAJEURE
        entry = ob.zone_mid
        if ob.direction == "bullish":
            sl = ob.zone_low - ob.zone_size * 0.15
        else:
            sl = ob.zone_high + ob.zone_size * 0.15
        risk = abs(entry - sl)

        # TP intelligent : prochaine liquidite reelle (PDH/PDL/Asia/Session/Swing)
        from bot.detectors.liquidity_target import find_next_liquidity_target
        try:
            df_m15_for_tp = load_cached(inst, "M15")
        except FileNotFoundError:
            df_m15_for_tp = None
        # === RR adaptatif a la VOLATILITE (ATR) ===
        # ATR sur les 50 dernieres bougies M1 avant le push
        entry_idx_in_sub = sub.index.get_indexer([ob.push_end_time], method='nearest')[0]
        atr_window = sub.iloc[max(0, entry_idx_in_sub - 50):entry_idx_in_sub]
        if len(atr_window) >= 14:
            atr_now = float((atr_window["high"] - atr_window["low"]).mean() or 1)
        else:
            atr_now = entry * 0.001
        atr_pct = (atr_now / entry) * 100   # % de volatilite

        # Mapping volatilite -> RR max + multiplicateur push
        if atr_pct < 0.05:
            max_rr_dynamic = 2.0
            tp_mult = 1.0
            vol_label = "tres faible"
        elif atr_pct < 0.10:
            max_rr_dynamic = 2.5
            tp_mult = 1.3
            vol_label = "faible"
        elif atr_pct < 0.20:
            max_rr_dynamic = 3.0
            tp_mult = 1.6
            vol_label = "normale"
        elif atr_pct < 0.40:
            max_rr_dynamic = 4.0
            tp_mult = 2.0
            vol_label = "elevee"
        else:
            max_rr_dynamic = 5.0
            tp_mult = 2.5
            vol_label = "tres elevee"

        push_size = ob.push_strength if hasattr(ob, 'push_strength') else (ob.zone_high - ob.zone_low) * 3
        max_tp_dist = push_size * tp_mult

        # Distance max en % du prix : adaptee aussi (entre 0.3% et 1.5%)
        max_dist_pct = min(1.5, max(0.3, atr_pct * 5))

        target = find_next_liquidity_target(
            df_m1=sub,
            df_m15=df_m15_for_tp,
            entry_price=entry,
            entry_time=ob.push_end_time,
            direction=ob.direction,
            min_distance_pct=0.05,
            max_distance_pct=max_dist_pct,
        )
        if target is None:
            continue   # pas de liquidite cible -> pas de trade
        tp = target.price

        # Plafond TP par rapport a la taille du push (adapte volatilite)
        if abs(tp - entry) > max_tp_dist:
            if ob.direction == "bullish":
                tp = entry + max_tp_dist
            else:
                tp = entry - max_tp_dist
            target_source_str = f"{target.source} (cap {tp_mult}x push)"
        else:
            target_source_str = target.source

        # Sanity check : RR min 1.0, max = max_rr_dynamic (depend volatilite)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0
        if rr < 1.0:
            continue
        if rr > max_rr_dynamic:
            if ob.direction == "bullish":
                tp = entry + risk * max_rr_dynamic
            else:
                tp = entry - risk * max_rr_dynamic
            reward = abs(tp - entry)
            rr = max_rr_dynamic
            target_source_str = f"{target_source_str} (vol {vol_label}, RR={max_rr_dynamic})"

        # Stocke pour le payload
        ob._tp_source = target_source_str

        # ============ CALCUL ALIGNEMENT HTF (score 0-100) ============
        from bot.detectors.htf_alignment import compute_alignment
        try:
            df_h4_align = load_cached(inst, "H4")
        except FileNotFoundError:
            df_h4_align = None
        try:
            df_h1_align = load_cached(inst, "H1")
        except FileNotFoundError:
            df_h1_align = None
        try:
            df_m15_align = load_cached(inst, "M15")
        except FileNotFoundError:
            df_m15_align = None

        try:
            df_m1_align = load_cached(inst, "M1")
        except FileNotFoundError:
            df_m1_align = sub

        htf_align = compute_alignment(
            df_m1=df_m1_align,
            df_h1=df_h1_align,
            df_h4=df_h4_align,
            df_m15=df_m15_align,
            trade_direction=ob.direction,
            ob_high=ob.zone_high, ob_low=ob.zone_low,
            entry_price=entry, take_profit=tp,
            asof=ob.push_end_time,
        )
        ob._htf_alignment = htf_align.to_dict()
        # Definir entry_time_obj = moment du comblement de l'OB (= vrai entry ICT)
        entry_time_obj = ob.fill_time if hasattr(ob, 'fill_time') and ob.fill_time is not None else ob.push_end_time

        # Outcome simule a partir du vrai entry_time
        outcome, _, _ = _compute_outcome(
            inst, str(day), ob.direction, entry, sl, tp,
            entry_time_obj.isoformat() if hasattr(entry_time_obj, 'isoformat') else None
        )

        # Features pour ML
        features = _extract_features(ob, sub)

        # TOUTES les bougies du BLOC OB (pas juste une)
        push_candles = []
        if hasattr(ob, 'ob_block_start') and hasattr(ob, 'ob_block_end'):
            for idx in range(ob.ob_block_start, ob.ob_block_end + 1):
                if idx >= len(sub):
                    continue
                row = sub.iloc[idx]
                push_candles.append({
                    "time": int(sub.index[idx].timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
        elif hasattr(ob, 'ob_candle_index') and ob.ob_candle_index < len(sub):
            idx = ob.ob_candle_index
            row = sub.iloc[idx]
            push_candles.append({
                "time": int(sub.index[idx].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        # entry_time_obj deja defini plus haut (= ob.fill_time)
        entry_unix = int(entry_time_obj.timestamp())
        visible_until = entry_unix + 5 * 60   # 5 bougies apres entry pour confirmation visuelle

        return {
            "instrument": inst,
            "day": str(day),
            "direction": ob.direction,
            "entry_time": entry_time_obj.isoformat() if hasattr(entry_time_obj, 'isoformat') else str(entry_time_obj),
            "entry_unix": entry_unix,
            "visible_until_unix": visible_until,
            "entry_price": round(entry, 4),
            "stop_loss": round(sl, 4),
            "take_profit": round(tp, 4),
            "risk_reward": round(rr, 2),
            "ob_zone": {
                "high": ob.zone_high, "low": ob.zone_low,
                "from_time": ob.push_start_time.isoformat() if hasattr(ob.push_start_time, 'isoformat') else str(ob.push_start_time),
                "to_time": ob.push_end_time.isoformat() if hasattr(ob.push_end_time, 'isoformat') else str(ob.push_end_time),
            },
            "ob_candles": push_candles,
            "features": features,
            "outcome": outcome,
            "generation": params.get("__gen__", 1),
            "smt": {
                "confirmed": getattr(ob, '_smt_confirmed', False),
                "results": getattr(ob, '_smt_results', []),
            },
            "fvg_m15": getattr(ob, '_has_fvg_m15', False),
            "tp_source": getattr(ob, '_tp_source', None),
            "htf_alignment": getattr(ob, '_htf_alignment', None),
        }

    raise HTTPException(404, "Aucun setup non-valide trouve apres 50 essais")


def _extract_features(ob, df):
    """Features mesurables pour ML."""
    from bot.killzones import current_killzone
    n_cdls = ob.push_end_index - ob.push_start_index + 1
    if ob.push_end_index >= 14:
        recent = df.iloc[ob.push_end_index - 14:ob.push_end_index]
        atr = float((recent["high"] - recent["low"]).mean() or 1)
    else:
        atr = 1.0
    push_atr = ob.push_strength / atr
    kz = current_killzone(ob.push_end_time)

    sweep_dist = 0.0
    if ob.swept_swing is not None:
        if ob.direction == "bearish":
            sweep_dist = ob.zone_high - ob.swept_level
        else:
            sweep_dist = ob.swept_level - ob.zone_low

    return {
        "push_strength": float(ob.push_strength),
        "push_atr": float(push_atr),
        "n_candles": int(n_cdls),
        "zone_size": float(ob.zone_size),
        "sweep_dist": float(sweep_dist),
        "killzone": kz.name if kz else "none",
        "hour_utc": int(ob.push_end_time.hour),
        "weekday": int(ob.push_end_time.weekday()),
    }


class ValidateIn(BaseModel):
    instrument: str
    day: str
    direction: str
    entry_time: str
    entry_price: float           # VALEURS USER (apres edition si edite)
    stop_loss: float
    take_profit: float
    risk_reward: float
    ob_zone: dict[str, Any]
    features: dict[str, Any]
    outcome: str = "unknown"
    verdict: str   # "yes" | "no" | "skip" | "edited"
    original: dict[str, Any] = {}    # valeurs du bot originalement (si edite)
    comment: str = ""                 # commentaire libre user


@app.post("/api/validate/save")
def validate_save(payload: ValidateIn) -> dict:
    if payload.verdict not in ("yes", "no", "skip", "edited"):
        raise HTTPException(400, "verdict invalide")

    # Recalcule outcome si edite (les valeurs ont change)
    if payload.verdict == "edited" or payload.outcome == "recompute":
        outcome, _, _ = _compute_outcome(
            payload.instrument, payload.day, payload.direction,
            payload.entry_price, payload.stop_loss, payload.take_profit,
            payload.entry_time,
        )
    else:
        outcome = payload.outcome

    # R multiplier
    risk = abs(payload.entry_price - payload.stop_loss)
    reward = abs(payload.take_profit - payload.entry_price)
    if outcome == "win" and risk > 0:
        pnl_r = reward / risk
    elif outcome == "loss":
        pnl_r = -1.0
    else:
        pnl_r = 0.0

    from bot.retrain import current_generation
    gen = current_generation()

    with SessionLocal() as s:
        rec = ValidatedSetup(
            instrument=payload.instrument,
            day=pd.Timestamp(payload.day).to_pydatetime().replace(tzinfo=None),
            direction=payload.direction,
            entry_time=pd.Timestamp(payload.entry_time).to_pydatetime().replace(tzinfo=None),
            entry_price=payload.entry_price,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            risk_reward=payload.risk_reward,
            ob_zone=payload.ob_zone,
            features=payload.features,
            verdict=payload.verdict,
            original=payload.original,
            outcome=outcome,
            outcome_pnl_r=pnl_r,
            comment=payload.comment,
            generation=gen,
        )
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return {"id": rec.id, "verdict": rec.verdict, "outcome": rec.outcome,
                "pnl_r": pnl_r, "new_rr": (reward / risk) if risk > 0 else 0,
                "generation": gen}


@app.post("/api/validate/retrain")
def validate_retrain(batch_size: int = 30) -> dict:
    """Analyse les N dernieres validations + ajuste les filtres + incremente generation."""
    from bot.retrain import retrain_from_validations
    return retrain_from_validations(batch_size=batch_size)


@app.get("/api/validate/generation-status")
def validate_generation_status() -> dict:
    """Statut de la generation courante : combien de validations, params, historique."""
    from bot.retrain import current_generation, load_current_params, n_validations_in_generation, get_all_generations_stats
    gen = current_generation()
    n = n_validations_in_generation(gen)
    return {
        "current_generation": gen,
        "validations_in_current_gen": n,
        "validations_needed_for_retrain": 30,
        "current_params": load_current_params(),
        "history": get_all_generations_stats(),
    }


@app.get("/api/validate/stats")
def validate_stats() -> dict:
    from sqlalchemy import select, func
    with SessionLocal() as s:
        rows = s.execute(select(ValidatedSetup)).scalars().all()
        n = len(rows)
        yes_count = sum(1 for r in rows if r.verdict == "yes")
        no_count = sum(1 for r in rows if r.verdict == "no")
        skip_count = sum(1 for r in rows if r.verdict == "skip")

        # Win rate sur les YES uniquement
        yes_rows = [r for r in rows if r.verdict == "yes"]
        yes_wins = sum(1 for r in yes_rows if r.outcome == "win")
        yes_losses = sum(1 for r in yes_rows if r.outcome == "loss")
        yes_wr = (yes_wins / (yes_wins + yes_losses) * 100) if (yes_wins + yes_losses) else 0
        yes_expectancy = sum(r.outcome_pnl_r for r in yes_rows) / len(yes_rows) if yes_rows else 0

        # Win rate sur les NO (anti-validation)
        no_rows = [r for r in rows if r.verdict == "no"]
        no_wins = sum(1 for r in no_rows if r.outcome == "win")
        no_losses = sum(1 for r in no_rows if r.outcome == "loss")
        no_wr = (no_wins / (no_wins + no_losses) * 100) if (no_wins + no_losses) else 0

        # Par actif
        by_inst: dict = {}
        for r in rows:
            d = by_inst.setdefault(r.instrument, {"yes": 0, "no": 0, "skip": 0})
            d[r.verdict] = d.get(r.verdict, 0) + 1

        return {
            "total": n,
            "yes": yes_count,
            "no": no_count,
            "skip": skip_count,
            "yes_winrate": round(yes_wr, 1),
            "yes_expectancy_r": round(yes_expectancy, 2),
            "no_winrate": round(no_wr, 1),   # si > 50% = tu rejettes des trades qui auraient gagne
            "by_instrument": by_inst,
        }


def run() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
