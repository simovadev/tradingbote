"""Genere un HTML interactif (lightweight-charts TradingView) pour analyser les trades.

Pour chaque trade du backtest, on affiche :
- Candlestick M1 sur une fenetre [ob_ts - 30min ... exit_ts + 10min]
- Zone OB (rectangle ob_low -> ob_high)
- Marker entry (point d'entree LIMIT/MARKET)
- Lignes SL et TP
- Marker fill (quand le LIMIT a fille)
- Marker exit (WIN/LOSS/BE)

Permet de naviguer trade par trade pour COMPRENDRE :
- Pourquoi un OB n'a pas pullback (mauvais OB)
- Pourquoi un LIMIT a NO_FILL alors que le prix etait proche
- Pourquoi un setup WIN/LOSS

Usage :
    python build_trade_viewer.py <csv_trades> <output_html>

Le CSV doit contenir : instrument, date, ob_ts, placed_ts, entry, sl, tp,
                       fill_ts, exit_ts, outcome, pnl_r, direction

Lit les bougies M1 depuis data_vantage/<ASSET>_M1.parquet.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "c:/Users/Shadow/TradingBot")

import pandas as pd

DATA_DIR = "c:/Users/Shadow/TradingBot/data_vantage"


def load_m1_window(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Charge les bougies M1 de l'actif dans la fenetre."""
    path = f"{DATA_DIR}/{asset}_M1.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df[(df.index >= start) & (df.index <= end)]
    return df


def candles_to_json(df: pd.DataFrame) -> list:
    """Convertit les bougies en format lightweight-charts."""
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
    return out


def build_html(trades: list[dict], output_path: str):
    """Genere le HTML avec lightweight-charts + nav trade par trade."""
    trades_json = json.dumps(trades)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Trade Viewer - Backtest V12</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body {{ font-family: monospace; background: #1e1e1e; color: #ddd; margin: 0; padding: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  .nav {{ background: #2a2a2a; padding: 12px; border-radius: 6px; margin-bottom: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .nav button {{ background: #444; color: #ddd; border: 0; padding: 8px 14px; cursor: pointer; border-radius: 4px; font-family: monospace; }}
  .nav button:hover {{ background: #555; }}
  .nav input {{ background: #222; color: #ddd; border: 1px solid #444; padding: 6px; width: 80px; }}
  .info {{ background: #2a2a2a; padding: 12px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; line-height: 1.6; }}
  .info span {{ display: inline-block; min-width: 120px; color: #888; }}
  .win {{ color: #4ade80; font-weight: bold; }}
  .loss {{ color: #f87171; font-weight: bold; }}
  .be {{ color: #fbbf24; font-weight: bold; }}
  .nofill {{ color: #60a5fa; font-weight: bold; }}
  .invalid {{ color: #c084fc; font-weight: bold; }}
  #chart {{ width: 100%; height: 600px; background: #1e1e1e; border-radius: 6px; }}
  h1 {{ color: #f3f4f6; }}
  .filters {{ background: #2a2a2a; padding: 12px; border-radius: 6px; margin-bottom: 12px; }}
  .filters label {{ margin-right: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Trade Viewer V12 — Backtest tick par tick</h1>

  <div class="filters">
    <label>Filtre outcome :</label>
    <select id="filterOutcome">
      <option value="">Tous</option>
      <option value="WIN">WIN seulement</option>
      <option value="LOSS">LOSS seulement</option>
      <option value="BE">BE seulement</option>
      <option value="NO_FILL">NO_FILL seulement</option>
      <option value="INVALID_PRICE">INVALID_PRICE seulement</option>
    </select>
    <label>Actif :</label>
    <select id="filterAsset"><option value="">Tous</option></select>
  </div>

  <div class="nav">
    <button id="prev">◄ Precedent</button>
    <span id="counter">0 / 0</span>
    <button id="next">Suivant ►</button>
    <input id="jumpInput" type="number" min="1" placeholder="N°">
    <button id="jumpBtn">Aller</button>
    <span style="margin-left: 20px; color: #888;">|</span>
    <button id="onlyLosses">LOSS uniquement</button>
    <button id="onlyWins">WIN uniquement</button>
    <button id="all">Tous</button>
  </div>

  <div class="info" id="info"></div>

  <div id="chart"></div>
</div>

<script>
const TRADES = {trades_json};
let currentIdx = 0;
let filteredTrades = TRADES.slice();

// Init chart
const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#1e1e1e' }}, textColor: '#ddd' }},
  grid: {{ vertLines: {{ color: '#333' }}, horzLines: {{ color: '#333' }} }},
  timeScale: {{ timeVisible: true, secondsVisible: false }},
  rightPriceScale: {{ borderColor: '#555' }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderUpColor: '#26a69a', borderDownColor: '#ef5350',
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});

// Lines pour entry, SL, TP
let entryLine = null, slLine = null, tpLine = null;

function classByOutcome(o) {{
  return {{WIN: 'win', LOSS: 'loss', BE: 'be', NO_FILL: 'nofill', INVALID_PRICE: 'invalid'}}[o] || '';
}}

function showTrade(idx) {{
  if (filteredTrades.length === 0) {{
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('info').innerHTML = '<i>Aucun trade ne correspond au filtre</i>';
    return;
  }}
  if (idx < 0) idx = 0;
  if (idx >= filteredTrades.length) idx = filteredTrades.length - 1;
  currentIdx = idx;
  const t = filteredTrades[idx];
  document.getElementById('counter').textContent = `${{idx + 1}} / ${{filteredTrades.length}}`;

  // Set candles
  candleSeries.setData(t.candles || []);

  // Remove old lines
  if (entryLine) candleSeries.removePriceLine(entryLine);
  if (slLine) candleSeries.removePriceLine(slLine);
  if (tpLine) candleSeries.removePriceLine(tpLine);

  // Add entry, SL, TP lines
  entryLine = candleSeries.createPriceLine({{
    price: t.entry, color: '#3b82f6', lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title: 'Entry'
  }});
  slLine = candleSeries.createPriceLine({{
    price: t.sl, color: '#f87171', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'SL'
  }});
  tpLine = candleSeries.createPriceLine({{
    price: t.tp, color: '#4ade80', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'TP'
  }});

  // Markers (placement, fill, exit)
  const markers = [];
  if (t.placed_time) markers.push({{
    time: t.placed_time, position: 'aboveBar', color: '#3b82f6', shape: 'circle', text: 'P'
  }});
  if (t.fill_time) markers.push({{
    time: t.fill_time, position: t.direction === 'bullish' ? 'belowBar' : 'aboveBar',
    color: '#fbbf24', shape: 'arrowUp', text: 'FILL'
  }});
  if (t.exit_time) markers.push({{
    time: t.exit_time, position: 'belowBar',
    color: t.outcome === 'WIN' ? '#4ade80' : (t.outcome === 'BE' ? '#fbbf24' : '#f87171'),
    shape: t.outcome === 'WIN' ? 'arrowUp' : 'arrowDown',
    text: t.outcome
  }});
  candleSeries.setMarkers(markers);

  // Info panel
  const cls = classByOutcome(t.outcome);
  document.getElementById('info').innerHTML = `
    <div><span>Actif :</span> <b>${{t.instrument}}</b> ${{t.date}} | direction <b>${{t.direction}}</b></div>
    <div><span>OB ts :</span> ${{t.ob_ts}}</div>
    <div><span>Placement :</span> ${{t.placed_ts}}</div>
    <div><span>Entry :</span> ${{t.entry}} | <span>SL :</span> ${{t.sl}} | <span>TP :</span> ${{t.tp}} | <span>RR :</span> ${{t.rr}}</div>
    <div><span>ML proba :</span> ${{t.ml_proba || 'N/A'}}</div>
    <div><span>Outcome :</span> <span class="${{cls}}">${{t.outcome}}</span> ${{t.pnl_r !== undefined ? `| <span>PnL :</span> <b class="${{cls}}">${{t.pnl_r}}R</b>` : ''}}</div>
    ${{t.fill_ts ? `<div><span>Fill :</span> ${{t.fill_ts}}</div>` : ''}}
    ${{t.exit_ts ? `<div><span>Exit :</span> ${{t.exit_ts}}</div>` : ''}}
  `;

  // === Auto-zoom force sur la nouvelle plage ===
  // 1. Reset time scale pour voir toutes les bougies
  chart.timeScale().fitContent();
  // 2. Force le price scale a se reajuster (autoscale = true)
  candleSeries.priceScale().applyOptions({{ autoScale: true }});
  // 3. Re-fit apres un petit delay (pour laisser le temps a setData de propager)
  setTimeout(() => {{
    chart.timeScale().fitContent();
    candleSeries.priceScale().applyOptions({{ autoScale: true }});
  }}, 50);
}}

function applyFilters() {{
  const o = document.getElementById('filterOutcome').value;
  const a = document.getElementById('filterAsset').value;
  filteredTrades = TRADES.filter(t => {{
    if (o && t.outcome !== o) return false;
    if (a && t.instrument !== a) return false;
    return true;
  }});
  showTrade(0);
}}

// Init assets dropdown
const assets = [...new Set(TRADES.map(t => t.instrument))].sort();
const assetSelect = document.getElementById('filterAsset');
assets.forEach(a => {{
  const opt = document.createElement('option');
  opt.value = a; opt.textContent = a;
  assetSelect.appendChild(opt);
}});

document.getElementById('prev').addEventListener('click', () => showTrade(currentIdx - 1));
document.getElementById('next').addEventListener('click', () => showTrade(currentIdx + 1));
document.getElementById('jumpBtn').addEventListener('click', () => {{
  const n = parseInt(document.getElementById('jumpInput').value) - 1;
  if (!isNaN(n)) showTrade(n);
}});
document.getElementById('onlyLosses').addEventListener('click', () => {{
  document.getElementById('filterOutcome').value = 'LOSS';
  applyFilters();
}});
document.getElementById('onlyWins').addEventListener('click', () => {{
  document.getElementById('filterOutcome').value = 'WIN';
  applyFilters();
}});
document.getElementById('all').addEventListener('click', () => {{
  document.getElementById('filterOutcome').value = '';
  document.getElementById('filterAsset').value = '';
  applyFilters();
}});
document.getElementById('filterOutcome').addEventListener('change', applyFilters);
document.getElementById('filterAsset').addEventListener('change', applyFilters);

// Keyboard nav
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') showTrade(currentIdx - 1);
  if (e.key === 'ArrowRight') showTrade(currentIdx + 1);
}});

// Init
showTrade(0);
</script>
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"OK: HTML genere {output_path} ({len(trades)} trades)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV de trades (bt_diag_*.csv ou bt_tick_*_trades.csv)")
    p.add_argument("--output", default="c:/Users/Shadow/TradingBot/trade_viewer.html")
    p.add_argument("--max", type=int, default=200, help="Max trades a inclure (default 200)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["ob_ts"] = pd.to_datetime(df["ob_ts"])
    df["placed_ts"] = pd.to_datetime(df["placed_ts"])
    if "fill_ts" in df.columns:
        df["fill_ts"] = pd.to_datetime(df["fill_ts"], errors="coerce")
    if "exit_ts" in df.columns:
        df["exit_ts"] = pd.to_datetime(df["exit_ts"], errors="coerce")

    # Limite aux N premiers (sinon HTML trop lourd)
    if len(df) > args.max:
        print(f"WARN: {len(df)} trades > {args.max}, on garde les {args.max} premiers")
        df = df.head(args.max)

    trades = []
    for _, r in df.iterrows():
        asset = r["instrument"]
        ob_ts = r["ob_ts"]
        placed_ts = r["placed_ts"]
        # Fenetre : ob_ts - 30min -> exit_ts + 10min (ou placed_ts + 60min si pas d'exit)
        win_start = ob_ts - pd.Timedelta(minutes=30)
        if pd.notna(r.get("exit_ts")):
            win_end = r["exit_ts"] + pd.Timedelta(minutes=10)
        else:
            win_end = placed_ts + pd.Timedelta(minutes=90)

        candles_df = load_m1_window(asset, win_start, win_end)
        if len(candles_df) == 0:
            continue

        trades.append({
            "instrument": asset,
            "date": str(r.get("date", "")),
            "direction": r["direction"],
            "ob_ts": str(ob_ts),
            "placed_ts": str(placed_ts),
            "placed_time": int(placed_ts.timestamp()),
            "fill_ts": str(r.get("fill_ts", "")),
            "fill_time": int(r["fill_ts"].timestamp()) if pd.notna(r.get("fill_ts")) else None,
            "exit_ts": str(r.get("exit_ts", "")),
            "exit_time": int(r["exit_ts"].timestamp()) if pd.notna(r.get("exit_ts")) else None,
            "entry": float(r["entry"]),
            "sl": float(r["sl"]),
            "tp": float(r["tp"]),
            "rr": float(r.get("rr", 1.5)),
            "ml_proba": float(r.get("ml", r.get("ml_proba", 0))) if pd.notna(r.get("ml", r.get("ml_proba", None))) else None,
            "outcome": r["outcome"],
            "pnl_r": float(r["pnl_r"]) if pd.notna(r.get("pnl_r")) else None,
            "candles": candles_to_json(candles_df),
        })

    build_html(trades, args.output)
    print(f"Ouvre dans navigateur : file:///{args.output.replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
