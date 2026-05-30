"""v22_visualisation_29mai.py - Genere 6 graphes HTML interactifs.

Pour chaque actif on affiche :
- Chandeliers M5 du 29/05/2026
- OBs detectes : zones rectangulaires colorees (bullish=vert, bearish=rouge)
- OBs qui passent ta regle (D1+H1+PD) : encadres en or
- Trades simules : flag entry + ligne SL + ligne TP
- Verdict final : nb OBs, nb trades, WR, PnL
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from bot_v2.concepts.safe import (
    find_obs_simple, daily_bias_safe, htf_value_at_t, atr_safe,
)

DATA = ROOT / "data_vantage"
OUT_DIR = ROOT / "v22_html"
OUT_DIR.mkdir(exist_ok=True)

ASSETS_MAP = {
    "EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "NAS100": "NAS100",
    "GER40":  "GER40",  "BTCUSD": "BTCUSD", "USOUSD": "CL-OIL",
}
SPREADS = {
    "EURUSD": 0.00016, "XAUUSD": 0.14, "NAS100": 1.0,
    "GER40":  0.8,     "BTCUSD": 6.0,  "USOUSD": 0.06,
}

TARGET_DAY = pd.Timestamp("2026-05-29", tz="UTC")
DAY_END = TARGET_DAY + pd.Timedelta(days=1)

SL_BUFFER_ATR = 0.1; TP_RR = 2.0; PARTIAL_R = 1.0
MAX_FILL_BARS = 20; MAX_HOLD_BARS = 50


def get_filters_for_ob(df_m5, df_h1, df_d1, ob):
    val_ts = ob.validation_ts; val_idx = ob.validation_index
    d1 = daily_bias_safe(df_d1, val_ts)
    d1_a = False
    if d1["ok"]:
        d1_h = d1["bias"] == "haussier"
        d1_a = (ob.direction == "bullish" and d1_h) or (ob.direction == "bearish" and not d1_h)
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_a = False
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_old = float(df_h1["close"].iloc[pos])
            h1_h = h1_close > h1_old
            h1_a = (ob.direction == "bullish" and h1_h) or (ob.direction == "bearish" and not h1_h)
    pd_a = False
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1]); pdl = float(df_d1_past["low"].iloc[-1])
            mid = (pdh + pdl) / 2; price = float(df_m5["close"].iloc[val_idx])
            pd_a = (ob.direction == "bullish" and price < mid) or (ob.direction == "bearish" and price > mid)
    return d1_a, h1_a, pd_a


def simulate_ob_trade_full(df_m5, ob, atr, spread=0.0):
    """Retourne dict avec status, entry_ts, exit_ts, sl, tp_1r, tp_2r, pnl."""
    val_idx = ob.validation_index
    if val_idx + 1 >= len(df_m5): return None
    hs = spread / 2
    if ob.direction == "bullish":
        sl = ob.ob_low - SL_BUFFER_ATR * atr; entry = ob.ob_high
        entry_eff = entry + hs; sl_eff = sl - hs
    else:
        sl = ob.ob_high + SL_BUFFER_ATR * atr; entry = ob.ob_low
        entry_eff = entry - hs; sl_eff = sl + hs
    if abs(entry_eff - sl_eff) < 1e-9: return None
    risk = abs(entry_eff - sl_eff)
    if ob.direction == "bullish":
        tp_1r = entry_eff + PARTIAL_R * risk; tp_2r = entry_eff + TP_RR * risk
    else:
        tp_1r = entry_eff - PARTIAL_R * risk; tp_2r = entry_eff - TP_RR * risk
    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_FILL_BARS]
    if len(fh) == 0: return None
    entry_idx_rel = None
    for i in range(len(fh)):
        if ob.direction == "bullish":
            if fl[i] <= entry:
                if fl[i] <= sl: return {"status": "INVALIDATED", "entry_ts": None, "exit_ts": None, "pnl": None,
                                          "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
                entry_idx_rel = i; break
        else:
            if fh[i] >= entry:
                if fh[i] >= sl: return {"status": "INVALIDATED", "entry_ts": None, "exit_ts": None, "pnl": None,
                                          "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
                entry_idx_rel = i; break
    if entry_idx_rel is None:
        return {"status": "NO_FILL", "entry_ts": None, "exit_ts": None, "pnl": None,
                "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
    entry_global = val_idx + 1 + entry_idx_rel
    entry_ts = df_m5.index[entry_global]
    fe = entry_global + 1
    en = min(fe + MAX_HOLD_BARS, len(df_m5))
    pfh = df_m5["high"].values[fe:en]; pfl = df_m5["low"].values[fe:en]
    half_locked = False; stop = sl_eff
    for i in range(len(pfh)):
        if ob.direction == "bullish":
            hit_sl = pfl[i] <= stop; hit_1r = pfh[i] >= tp_1r; hit_2r = pfh[i] >= tp_2r
        else:
            hit_sl = pfh[i] >= stop; hit_1r = pfl[i] <= tp_1r; hit_2r = pfl[i] <= tp_2r
        if hit_sl and hit_2r:
            if half_locked: return {"status": "WIN_BE", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": 0.5 * PARTIAL_R, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
            return {"status": "LOSS", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": -1.0, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
        if hit_sl:
            if half_locked: return {"status": "WIN_BE", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": 0.5 * PARTIAL_R, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
            return {"status": "LOSS", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": -1.0, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
        if not half_locked and hit_1r:
            half_locked = True; stop = entry_eff
            if hit_2r: return {"status": "WIN_FULL", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": 0.5 * PARTIAL_R + 0.5 * TP_RR, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
        elif half_locked and hit_2r:
            return {"status": "WIN_FULL", "entry_ts": entry_ts, "exit_ts": df_m5.index[fe+i], "pnl": 0.5 * PARTIAL_R + 0.5 * TP_RR, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
    if half_locked: return {"status": "TIME_BE", "entry_ts": entry_ts, "exit_ts": df_m5.index[en-1], "pnl": 0.5 * PARTIAL_R, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}
    return {"status": "TIME_FLAT", "entry_ts": entry_ts, "exit_ts": df_m5.index[en-1], "pnl": 0.0, "sl": sl, "tp_1r": tp_1r, "tp_2r": tp_2r, "entry": entry}


def generate_html(asset_name, df_m5_day, obs_data, trades_data, summary):
    """Genere un HTML avec lightweight charts."""
    # Convertit candles en JSON serializable
    candles = []
    for ts, row in df_m5_day.iterrows():
        candles.append({
            "time": int(ts.timestamp()),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
        })

    # OBs JSON (zones rectangulaires + couleurs selon filtres)
    obs_json = []
    for ob_d in obs_data:
        obs_json.append({
            "ts_form": int(ob_d["formation_ts"].timestamp()),
            "ts_val": int(ob_d["validation_ts"].timestamp()),
            "ob_low": ob_d["ob_low"], "ob_high": ob_d["ob_high"],
            "direction": ob_d["direction"],
            "d1_a": ob_d["d1_a"], "h1_a": ob_d["h1_a"], "pd_a": ob_d["pd_a"],
            "passes_user_rule": ob_d["d1_a"] and ob_d["h1_a"] and ob_d["pd_a"],
        })

    # Trades JSON
    trades_json = []
    for t in trades_data:
        trades_json.append({
            "entry_ts": int(t["entry_ts"].timestamp()) if t["entry_ts"] else None,
            "exit_ts": int(t["exit_ts"].timestamp()) if t["exit_ts"] else None,
            "ob_form_ts": int(t["ob_form_ts"].timestamp()),
            "entry": t["entry"], "sl": t["sl"], "tp_1r": t["tp_1r"], "tp_2r": t["tp_2r"],
            "status": t["status"], "pnl": t["pnl"], "direction": t["direction"],
        })

    import json
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>V22 {asset_name} - 29/05/2026</title>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
body {{ font-family: 'Consolas', monospace; background: #0a0e16; color: #d4d8e0; padding: 16px; margin: 0; }}
.header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
h1 {{ color: #e8b339; margin: 0; font-size: 20px; }}
.summary {{ background: #131922; padding: 12px; border-radius: 6px; margin-bottom: 12px; display: flex; gap: 24px; flex-wrap: wrap; font-size: 14px; }}
.summary div {{ display: flex; align-items: center; gap: 6px; }}
.summary span.label {{ color: #8a93a8; }}
.summary span.value {{ color: #d4d8e0; font-weight: bold; }}
.summary span.pos {{ color: #26d97f; }}
.summary span.neg {{ color: #ef5350; }}
#chart {{ height: 720px; width: 100%; background: #131922; border-radius: 6px; }}
.legend {{ margin-top: 10px; display: flex; gap: 18px; font-size: 12px; flex-wrap: wrap; }}
.legend .item {{ display: flex; align-items: center; gap: 6px; }}
.box {{ width: 14px; height: 14px; border: 1px solid; }}
.box-ob-buy {{ background: rgba(38, 217, 127, 0.2); border-color: #26d97f; }}
.box-ob-sell {{ background: rgba(239, 83, 80, 0.2); border-color: #ef5350; }}
.box-passes {{ background: rgba(232, 179, 57, 0.3); border-color: #e8b339; border-width: 2px; }}
</style>
</head><body>

<div class="header">
<h1>V22 - {asset_name} - 29/05/2026 (jour complet)</h1>
<div style="font-size: 12px; color: #8a93a8;">Auto-genere par v22_visualisation_29mai.py</div>
</div>

<div class="summary">
<div><span class="label">OBs detectes :</span> <span class="value">{summary['n_obs']}</span></div>
<div><span class="label">Passent ta regle :</span> <span class="value">{summary['n_passes']}</span></div>
<div><span class="label">Trades pris :</span> <span class="value">{summary['n_filled']}</span></div>
<div><span class="label">WIN :</span> <span class="value pos">{summary['wins']}</span></div>
<div><span class="label">LOSS :</span> <span class="value neg">{summary['losses']}</span></div>
<div><span class="label">No-fill / Invalides :</span> <span class="value">{summary['n_skipped']}</span></div>
<div><span class="label">PnL :</span> <span class="value {'pos' if summary['pnl'] >= 0 else 'neg'}">{summary['pnl']:+.2f}R</span></div>
</div>

<div id="chart"></div>

<div class="legend">
<div class="item"><div class="box box-ob-sell"></div>Zone ROUGE (entry -> SL = risque)</div>
<div class="item"><div class="box box-ob-buy"></div>Zone VERTE (entry -> TP = gain)</div>
<div class="item"><span style="color:#e8b339">--- Entry (ligne or)</span></div>
<div class="item"><span style="color:#ff9800">... TP 1R (orange pointille)</span></div>
<div class="item">Fleche or = LONG/SHORT | Cercle = WIN/LOSS</div>
</div>

<script>
const candles = {json.dumps(candles)};
const obs = {json.dumps(obs_json)};
const trades = {json.dumps(trades_json)};

const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#131922' }}, textColor: '#d4d8e0' }},
  grid: {{ vertLines: {{ color: '#1d2330' }}, horzLines: {{ color: '#1d2330' }} }},
  timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#2a3140' }},
  rightPriceScale: {{ borderColor: '#2a3140' }},
  width: document.getElementById('chart').clientWidth,
  height: 720,
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26d97f', downColor: '#ef5350',
  borderUpColor: '#26d97f', borderDownColor: '#ef5350',
  wickUpColor: '#26d97f', wickDownColor: '#ef5350',
}});
candleSeries.setData(candles);

// === ON N'AFFICHE QUE LES TRADES PRIS ===
// On ignore les 54 OBs detectes qui rendaient le graphe illisible.
// Pour chaque trade :
//   - OB zone (dore translucide) entre ob_form_ts et exit_ts
//   - SL (rouge plein) entre entry_ts et exit_ts
//   - TP 1R (orange pointille)
//   - TP 2R (vert plein)
//   - Marker entry LONG/SHORT avec fleche doree
//   - Marker exit WIN/LOSS avec cercle
const allMarkers = [];

trades.forEach((t) => {{
  if (!t.entry_ts) return;
  const isWin = t.status.startsWith('WIN');
  const isLoss = t.status === 'LOSS';
  const tradeColor = isWin ? '#26d97f' : (isLoss ? '#ef5350' : '#888');
  const exitTs = t.exit_ts || (t.entry_ts + 3600);
  const direction = t.direction;

  // === ZONE ROUGE : entry <-> SL (zone de RISQUE) ===
  // On utilise 2 area series : une qui va jusqu'a entry (base), une qui va jusqu'a SL.
  // En combinant baseline area entre top et bottom, on simule un rectangle.
  // Plus simple : on utilise une AreaSeries avec topColor/bottomColor transparents et lineColor invisible,
  // mais ce n'est pas natif. On va plutot utiliser BaselineSeries qui colore au-dessus/dessous d'une baseline.
  // Pour vraiment dessiner un rectangle, on utilise un trick : 2 LineSeries (top=entry, bottom=sl) puis
  // une AreaSeries avec lineColor=transparent et topColor=red.

  // Methode : on cree une AreaSeries avec topLineColor invisible.
  // setData de l'area = top (entry) et on baseline = sl via 'baseLineStyle'.
  // En pratique, lightweight-charts ne supporte pas un fond entre 2 lignes arbitraires.
  // On va donc utiliser une Histogram series transparente positionnee sur entry,
  // ou plus simple : 2 areas, l'une bornee a entry, l'autre a sl.

  // SOLUTION SIMPLE QUI MARCHE : on cree un AreaSeries dont la 'baseline'
  // est implicitement 0. On le borne via priceScale customisee. Trop complexe.

  // ZONES REMPLIES via BaselineSeries (baseline = entry, top remplit en vert TP, bottom remplit en rouge SL)
  // Une BaselineSeries colore au-dessus de baseValue en topColor et en-dessous en bottomColor.
  // Pour bullish : baseline=entry. Above (vers TP) = vert. Below (vers SL) = rouge.
  // Pour bearish : baseline=entry. Above (vers SL) = rouge. Below (vers TP) = vert.
  const topColor = direction === 'bullish' ? 'rgba(38, 217, 127, 0.20)' : 'rgba(239, 83, 80, 0.20)';
  const bottomColor = direction === 'bullish' ? 'rgba(239, 83, 80, 0.20)' : 'rgba(38, 217, 127, 0.20)';

  // On dessine une ligne au prix de TP2R (cote top) puis une au prix SL (cote bottom)
  // pour que la BaselineSeries ait des donnees a colorier des deux cotes.
  // Strategie : 3 baseline series, une qui va a TP2R, une qui va a SL, l'autre au prix moyen.
  // En realite plus simple : une AreaSeries entre entry et TP2R (vert), une autre entre entry et SL (rouge).
  // L'AreaSeries colore l'aire entre la ligne et la baseline (=0 par defaut).
  // On peut detourner avec baseValue customisee via 'autoscaleInfoProvider'.

  // SOLUTION SIMPLE : utiliser 2 BaselineSeries.
  // BaselineSeries(baseValue = entry, data = [{time, value: TP_2R}, {time:exit, value:TP_2R}])
  // -> colore au-dessus de entry vers TP_2R en topFillColor.
  const gainSeries = chart.addBaselineSeries({{
    baseValue: {{ type: 'price', price: t.entry }},
    topFillColor1: topColor,
    topFillColor2: topColor,
    topLineColor: 'rgba(0,0,0,0)',
    bottomFillColor1: 'rgba(0,0,0,0)',
    bottomFillColor2: 'rgba(0,0,0,0)',
    bottomLineColor: 'rgba(0,0,0,0)',
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false,
  }});
  gainSeries.setData([
    {{ time: t.entry_ts, value: t.tp_2r }},
    {{ time: exitTs, value: t.tp_2r }}
  ]);

  const riskSeries = chart.addBaselineSeries({{
    baseValue: {{ type: 'price', price: t.entry }},
    topFillColor1: 'rgba(0,0,0,0)',
    topFillColor2: 'rgba(0,0,0,0)',
    topLineColor: 'rgba(0,0,0,0)',
    bottomFillColor1: bottomColor,
    bottomFillColor2: bottomColor,
    bottomLineColor: 'rgba(0,0,0,0)',
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false,
  }});
  riskSeries.setData([
    {{ time: t.entry_ts, value: t.sl }},
    {{ time: exitTs, value: t.sl }}
  ]);

  // BORDURES : lignes nettes entry / SL / TP1R / TP2R
  const entryLine = chart.addLineSeries({{
    color: '#e8b339', lineWidth: 2, lineStyle: 0,
    priceLineVisible: false, lastValueVisible: false,
  }});
  entryLine.setData([
    {{ time: t.entry_ts, value: t.entry }},
    {{ time: exitTs, value: t.entry }}
  ]);

  const slLine = chart.addLineSeries({{
    color: '#ef5350', lineWidth: 2, lineStyle: 0,
    priceLineVisible: false, lastValueVisible: false,
  }});
  slLine.setData([
    {{ time: t.entry_ts, value: t.sl }},
    {{ time: exitTs, value: t.sl }}
  ]);

  const tp1Line = chart.addLineSeries({{
    color: '#ff9800', lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false,
  }});
  tp1Line.setData([
    {{ time: t.entry_ts, value: t.tp_1r }},
    {{ time: exitTs, value: t.tp_1r }}
  ]);

  const tp2Line = chart.addLineSeries({{
    color: '#26d97f', lineWidth: 2, lineStyle: 0,
    priceLineVisible: false, lastValueVisible: false,
  }});
  tp2Line.setData([
    {{ time: t.entry_ts, value: t.tp_2r }},
    {{ time: exitTs, value: t.tp_2r }}
  ]);

  // Markers entry + exit
  allMarkers.push({{
    time: t.entry_ts,
    position: direction === 'bullish' ? 'belowBar' : 'aboveBar',
    color: '#e8b339',
    shape: direction === 'bullish' ? 'arrowUp' : 'arrowDown',
    text: (direction === 'bullish' ? 'LONG' : 'SHORT'),
    size: 2,
  }});
  allMarkers.push({{
    time: exitTs,
    position: direction === 'bullish' ? 'aboveBar' : 'belowBar',
    color: tradeColor,
    shape: 'circle',
    text: t.status + ' ' + (t.pnl !== null ? t.pnl.toFixed(2) + 'R' : ''),
    size: 2,
  }});
}});

// Lightweight Charts exige des markers tries par time croissant
allMarkers.sort(function(a, b) {{ return a.time - b.time; }});
if (allMarkers.length > 0) {{
  candleSeries.setMarkers(allMarkers);
}}

chart.timeScale().fitContent();

window.addEventListener('resize', () => {{
  chart.applyOptions({{ width: document.getElementById('chart').clientWidth }});
}});
</script>

</body></html>
"""
    return html


def process_asset(user_name, file_name):
    print(f"\n=== {user_name} ===")
    sp = SPREADS[user_name]

    # Charge data : on prend M5 sur 7 jours autour (pour avoir le contexte AVANT le 29/05)
    df_m5_full = pd.read_parquet(DATA / f"{file_name}_M5.parquet")[["open","high","low","close"]]
    df_h1_full = pd.read_parquet(DATA / f"{file_name}_H1.parquet")[["open","high","low","close"]]
    df_d1_full = pd.read_parquet(DATA / f"{file_name}_D1.parquet")[["open","high","low","close"]]

    # Periode de contexte : 5 jours avant le 29/05 pour avoir le contexte M5 + 29/05
    ctx_start = TARGET_DAY - pd.Timedelta(days=5)
    df_m5 = df_m5_full[(df_m5_full.index >= ctx_start) & (df_m5_full.index < DAY_END)]
    df_h1 = df_h1_full[(df_h1_full.index >= ctx_start) & (df_h1_full.index < DAY_END)]
    df_d1 = df_d1_full[(df_d1_full.index >= TARGET_DAY - pd.Timedelta(days=30)) & (df_d1_full.index < DAY_END)]
    print(f"  M5 contexte : {len(df_m5)} bougies | H1 : {len(df_h1)} | D1 : {len(df_d1)}")

    # Detect OBs sur tout le contexte, on filtre ceux du 29/05
    obs = find_obs_simple(df_m5, as_of_index=len(df_m5) - 1)
    obs_day = [ob for ob in obs if TARGET_DAY <= ob.validation_ts < DAY_END]
    print(f"  OBs detectes sur 29/05 : {len(obs_day)}")

    # Pour chaque OB : filters + simulation
    obs_data = []
    trades_data = []
    n_passes = 0; n_filled = 0; wins = 0; losses = 0; pnl_total = 0
    n_skipped = 0
    for ob in obs_day:
        d1a, h1a, pda = get_filters_for_ob(df_m5, df_h1, df_d1, ob)
        passes = d1a and h1a and pda
        if passes: n_passes += 1
        ob_d = {
            "formation_ts": ob.formation_ts, "validation_ts": ob.validation_ts,
            "ob_low": ob.ob_low, "ob_high": ob.ob_high,
            "direction": ob.direction,
            "d1_a": d1a, "h1_a": h1a, "pd_a": pda,
        }
        obs_data.append(ob_d)

        # Ne simule que les OBs qui passent ta regle
        if passes:
            atr = atr_safe(df_m5, ob.validation_index)
            if atr <= 0:
                n_skipped += 1; continue
            res = simulate_ob_trade_full(df_m5, ob, atr, spread=sp)
            if res is None:
                n_skipped += 1; continue
            res["direction"] = ob.direction
            res["ob_form_ts"] = ob.formation_ts
            trades_data.append(res)
            if res["status"].startswith("WIN"): wins += 1; n_filled += 1
            elif res["status"] == "LOSS": losses += 1; n_filled += 1
            else: n_skipped += 1
            if res["pnl"] is not None: pnl_total += res["pnl"]

    # Filtre les bougies du 29/05 pour le graphe
    df_29 = df_m5[(df_m5.index >= TARGET_DAY) & (df_m5.index < DAY_END)]
    print(f"  Bougies M5 29/05 : {len(df_29)}")
    if len(df_29) == 0:
        print(f"  Pas de donnees pour le 29/05, skip"); return

    summary = {
        "n_obs": len(obs_day), "n_passes": n_passes, "n_filled": n_filled,
        "wins": wins, "losses": losses, "n_skipped": n_skipped, "pnl": pnl_total,
    }
    html = generate_html(user_name, df_29, obs_data, trades_data, summary)
    out_path = OUT_DIR / f"v22_{user_name}_29mai2026.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML : {out_path}")
    print(f"  Resume : {n_passes}/{len(obs_day)} passent | {n_filled} filles | {wins}W/{losses}L | PnL {pnl_total:+.2f}R")


def main():
    print("=" * 70)
    print("V22 - VISUALISATION 29/05/2026 - 6 actifs")
    print("=" * 70)
    for user_name, file_name in ASSETS_MAP.items():
        process_asset(user_name, file_name)

    # Index page
    index_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>V22 - 29/05/2026</title>
<style>body{font-family:monospace;background:#0a0e16;color:#d4d8e0;padding:20px}
h1{color:#e8b339}a{color:#26d97f;display:block;margin:8px 0;font-size:16px}
a:hover{color:#e8b339}</style></head><body>
<h1>V22 - Visualisation 29/05/2026 - 6 actifs</h1>
<p>Clique sur un actif pour voir le chart M5 du jour avec OBs detectes, filtres applique, trades simules.</p>
"""
    for user_name in ASSETS_MAP.keys():
        index_html += f'<a href="v22_{user_name}_29mai2026.html">{user_name} - 29/05/2026</a>\n'
    index_html += "</body></html>"
    (OUT_DIR / "index.html").write_text(index_html, encoding="utf-8")

    print(f"\n=== INDEX : {OUT_DIR / 'index.html'} ===")


if __name__ == "__main__":
    main()
