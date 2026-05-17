"""Serveur web bot_v2 — visualisation des setups Vizion + REPLAY LIVE.

Le replay permet de voir le bot detecter les setups en temps reel, bougie par bougie,
comme si le marche tournait. C'est l'outil ideal pour valider la logique vs ton trading.

Endpoints :
- GET /                                  : page HTML
- GET /api/instruments                    : liste actifs primaires
- GET /api/candles?instrument=X&tf=Y&days=Z         : OHLC (mode statique)
- GET /api/setups?instrument=X&days=Y               : setups detectes (statique)
- GET /api/replay?instrument=X&tf=Y&days=Z&speed=N  : Server-Sent-Events stream live
- GET /api/backtest?instrument=X&days=Y             : stats

Usage :
    ./venv/Scripts/python.exe -m bot_v2.web_server
    -> http://127.0.0.1:8001
"""
from __future__ import annotations

import asyncio
import json
import time

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from bot_v2.backtest import backtest_instrument, simulate_trade
from bot_v2.concepts.daily_bias import build_d1_from_h1, compute_daily_bias
from bot_v2.concepts.killzones import killzone_at
from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.config import HOST, INSTRUMENTS
from bot_v2.data_loader import load
from bot_v2.pipeline import evaluate_ob, run_pipeline


app = FastAPI(title="Bot V2 — Vizion")


# ============ ETAT GLOBAL REPLAY ============
# Un seul replay actif a la fois suffit (un seul user local).
class ReplayState:
    paused: bool = False
    stop: bool = False

REPLAY = ReplayState()


# ============ HTML ============
HTML = r"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Bot V2 — Vizion Replay</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; background: #0d1117; color: #c9d1d9; }
    header { padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d;
             display: flex; align-items: center; justify-content: space-between; }
    header h1 { margin: 0; font-size: 18px; }
    .toolbar { padding: 12px 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
               background: #161b22; border-bottom: 1px solid #30363d; }
    select, button, input { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                            padding: 6px 12px; border-radius: 4px; font-size: 13px; }
    button { cursor: pointer; }
    button:hover { background: #30363d; }
    button.primary { background: #238636; color: white; border-color: #238636; }
    button.primary:hover { background: #2ea043; }
    button.danger { background: #6e2727; color: white; border-color: #6e2727; }
    button.danger:hover { background: #8a3434; }
    #chart { width: calc(100% - 40px); height: 60vh; margin: 0 20px; }
    #info { padding: 10px 20px; font-size: 13px; min-height: 30px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; margin-right: 6px;
             font-size: 11px; font-weight: bold; }
    .badge.bull { background: #1f4429; color: #3fb950; }
    .badge.bear { background: #4a1f1f; color: #f85149; }
    .badge.reject { background: #2d2d2d; color: #8b949e; }
    .badge.score { background: #1d3050; color: #58a6ff; }
    #log { padding: 0 20px; max-height: 30vh; overflow: auto; font-size: 12px; font-family: monospace; }
    .row { padding: 4px 0; border-bottom: 1px solid #21262d; }
    .row.trade { color: #3fb950; }
    .row.reject { color: #6e7681; }
    .label { color: #8b949e; }
    .val { font-weight: bold; color: #c9d1d9; }
    #stats { padding: 10px 20px; font-size: 12px; color: #8b949e; }
    #stats span { margin-right: 16px; }
  </style>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  <header>
    <h1>Bot V2 — Vizion Replay Live</h1>
    <span id="info"></span>
  </header>
  <div class="toolbar">
    <label>Actif :</label>
    <select id="instrument"></select>
    <label>TF :</label>
    <select id="tf">
      <option value="M1">M1</option>
      <option value="M5" selected>M5</option>
      <option value="M15">M15</option>
      <option value="H1">H1</option>
    </select>
    <label>Jours :</label>
    <input id="days" type="number" value="3" min="1" max="30" style="width:60px">
    <label>Vitesse :</label>
    <select id="speed">
      <option value="50">×1 (50ms/bar)</option>
      <option value="20" selected>×5 (20ms)</option>
      <option value="10">×10 (10ms)</option>
      <option value="2">×50 (2ms)</option>
    </select>
    <button id="play" class="primary">▶ Replay live</button>
    <button id="pause" disabled>⏸ Pause</button>
    <button id="stop" class="danger" disabled>⏹ Stop</button>
    <button id="static">Charger tout (statique)</button>
  </div>
  <div id="chart"></div>
  <div id="stats"></div>
  <div id="log"></div>

<script>
let chart, candleSeries, eventSource = null, paused = false;
let allCandles = [], allMarkers = [], stats = { trades: 0, candidates: 0, candles: 0 };
let pendingCandles = [];
let lastCandleTime = 0;
let tradeZones = []; // {tpSeries, slSeries, entryLine, slLine, tpLine}

function drawTradeZone(s) {
  if (!s.entry || !s.sl || !s.tp) return;

  // Point d'entrée temporel : on prefere fill_time si dispo, sinon validation_time
  const startTime = s.fill_time || s.validation_time;
  const tfSec = chart._tfSeconds || 300;
  // Fin du rectangle : exit_time si on a un outcome, sinon +30 bougies (PENDING)
  const endTime = s.exit_time || (startTime + 30 * tfSec);

  // Couleurs selon outcome
  // - WIN  -> vert vif pour TP (rempli), rouge pale pour SL (annulé)
  // - LOSS -> rouge vif pour SL (rempli), vert pale pour TP (annulé)
  // - PENDING/autre -> normal
  let tpAlpha = 0.30, slAlpha = 0.30, tpStrong = false, slStrong = false;
  if (s.outcome === 'WIN') { tpStrong = true; slAlpha = 0.10; }
  else if (s.outcome === 'LOSS') { slStrong = true; tpAlpha = 0.10; }

  const tpFill = chart.addBaselineSeries({
    baseValue: { type: 'price', price: s.entry },
    topFillColor1: `rgba(63, 185, 80, ${tpStrong ? 0.55 : tpAlpha})`,
    topFillColor2: `rgba(63, 185, 80, ${tpStrong ? 0.25 : tpAlpha * 0.4})`,
    topLineColor: `rgba(63, 185, 80, 0.7)`,
    bottomFillColor1: `rgba(63, 185, 80, ${tpStrong ? 0.55 : tpAlpha})`,
    bottomFillColor2: `rgba(63, 185, 80, ${tpStrong ? 0.25 : tpAlpha * 0.4})`,
    bottomLineColor: `rgba(63, 185, 80, 0.7)`,
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  tpFill.setData([
    { time: startTime, value: s.tp },
    { time: endTime, value: s.tp },
  ]);

  const slFill = chart.addBaselineSeries({
    baseValue: { type: 'price', price: s.entry },
    topFillColor1: `rgba(248, 81, 73, ${slStrong ? 0.55 : slAlpha})`,
    topFillColor2: `rgba(248, 81, 73, ${slStrong ? 0.25 : slAlpha * 0.4})`,
    topLineColor: `rgba(248, 81, 73, 0.7)`,
    bottomFillColor1: `rgba(248, 81, 73, ${slStrong ? 0.55 : slAlpha})`,
    bottomFillColor2: `rgba(248, 81, 73, ${slStrong ? 0.25 : slAlpha * 0.4})`,
    bottomLineColor: `rgba(248, 81, 73, 0.7)`,
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  slFill.setData([
    { time: startTime, value: s.sl },
    { time: endTime, value: s.sl },
  ]);

  // Ligne HORIZONTALE au prix d'entry (limitee dans le temps, pas infinie)
  const entryColor = s.direction === 'bullish' ? '#3fb950' : '#f85149';
  const entryLineSeries = chart.addLineSeries({
    color: entryColor,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  entryLineSeries.setData([
    { time: startTime, value: s.entry },
    { time: endTime, value: s.entry },
  ]);

  // Ligne VERTICALE à l'entry : on utilise priceLine sur une series temporaire
  // Astuce : on cree une LineSeries qui va du minPrice au maxPrice à startTime,
  // mais c'est pas faisable en LWC v4 nativement.
  // Solution : on dessine un MARKER tres visible sur la candleSeries au startTime
  // (deja fait avec l'arrow, mais on peut renforcer en ajoutant des markers sur 2 prix).

  // Pour la "ligne verticale" -> on cree une LineSeries de 2 points au meme time
  // mais a 2 prix differents pour simuler. Mais LWC ne le supporte pas (1 valeur par time).
  // Workaround : on cree un MARKER additionnel au prix SL et au prix TP pour marquer la verticale.
  // En realite la fleche du marker fait deja le job indiquant le moment.

  // Status overlay (WIN/LOSS) avec un marker textuel SUR la barre d'exit
  let outcomeMarker = null;
  if (s.outcome === 'WIN' && s.exit_time) {
    outcomeMarker = {
      time: s.exit_time,
      position: s.direction === 'bullish' ? 'aboveBar' : 'belowBar',
      color: '#3fb950',
      shape: 'circle',
      text: `WIN +${s.pnl_usd ? s.pnl_usd.toFixed(0) : ''}$`,
    };
    allMarkers.push(outcomeMarker);
  } else if (s.outcome === 'LOSS' && s.exit_time) {
    outcomeMarker = {
      time: s.exit_time,
      position: s.direction === 'bullish' ? 'belowBar' : 'aboveBar',
      color: '#f85149',
      shape: 'circle',
      text: `LOSS ${s.pnl_usd ? s.pnl_usd.toFixed(0) : ''}$`,
    };
    allMarkers.push(outcomeMarker);
  }
  if (outcomeMarker) candleSeries.setMarkers(allMarkers);

  tradeZones.push({ tpFill, slFill, entryLineSeries });

  // Limite a 8 trades visibles simultanement
  if (tradeZones.length > 8) {
    const old = tradeZones.shift();
    try { chart.removeSeries(old.tpFill); } catch (e) {}
    try { chart.removeSeries(old.slFill); } catch (e) {}
    try { chart.removeSeries(old.entryLineSeries); } catch (e) {}
  }
}

function clearAllZones() {
  for (const z of tradeZones) {
    try { chart.removeSeries(z.tpFill); } catch (e) {}
    try { chart.removeSeries(z.slFill); } catch (e) {}
    try { chart.removeSeries(z.entryLineSeries); } catch (e) {}
  }
  tradeZones = [];
}

async function init() {
  const insts = await fetch('/api/instruments').then(r => r.json());
  const sel = document.getElementById('instrument');
  for (const i of insts) {
    const opt = document.createElement('option');
    opt.value = i.symbol;
    opt.textContent = `${i.symbol} — ${i.label}`;
    sel.appendChild(opt);
  }

  chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
    grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
    timeScale: { timeVisible: true, secondsVisible: false },
    rightPriceScale: { autoScale: true },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });

  document.getElementById('play').addEventListener('click', startReplay);
  document.getElementById('pause').addEventListener('click', togglePause);
  document.getElementById('stop').addEventListener('click', stopReplay);
  document.getElementById('static').addEventListener('click', loadStatic);
}

function startReplay() {
  console.log('[Replay] startReplay clicked');
  stopReplay();
  const inst = document.getElementById('instrument').value;
  const tf = document.getElementById('tf').value;
  const days = document.getElementById('days').value;
  const speed = document.getElementById('speed').value;
  console.log('[Replay] params:', { inst, tf, days, speed });

  // Reset
  allCandles = []; allMarkers = []; pendingCandles = [];
  stats = { trades: 0, candidates: 0, candles: 0 };
  clearAllZones();
  candleSeries.setData([]);
  candleSeries.setMarkers([]);
  document.getElementById('log').innerHTML = '';
  document.getElementById('info').textContent = '⏳ Pré-calcul (5-15s)... ne fermez pas la page';
  paused = false;

  // Memorise le TF en secondes pour dimensionner les rectangles
  const tfSec = { M1: 60, M5: 300, M15: 900, H1: 3600 }[tf] || 300;
  chart._tfSeconds = tfSec;

  document.getElementById('play').disabled = true;
  document.getElementById('pause').disabled = false;
  document.getElementById('stop').disabled = false;

  document.getElementById('info').textContent = '⏳ Pré-calcul des OB (peut prendre 5-15s)...';
  eventSource = new EventSource(`/api/replay?instrument=${inst}&tf=${tf}&days=${days}&speed=${speed}`);
  eventSource.addEventListener('ready', e => {
    const d = JSON.parse(e.data);
    document.getElementById('info').textContent =
      `▶ Replay en cours (${d.total} bougies, ${d.setups} setups detectes)`;
  });
  eventSource.addEventListener('candle', e => {
    if (paused) { pendingCandles.push(JSON.parse(e.data)); return; }
    const c = JSON.parse(e.data);
    candleSeries.update(c);
    stats.candles++;
    updateStats();
  });
  eventSource.addEventListener('setup', e => {
    const s = JSON.parse(e.data);
    stats.candidates++;
    if (s.verdict === 'TRADE') {
      stats.trades++;
      allMarkers.push({
        time: s.validation_time,
        position: s.direction === 'bullish' ? 'belowBar' : 'aboveBar',
        color: s.direction === 'bullish' ? '#3fb950' : '#f85149',
        shape: s.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
        text: `${s.direction === 'bullish' ? 'LONG' : 'SHORT'} ${s.score} | RR ${s.rr ? s.rr.toFixed(1) : '?'}`,
      });
      candleSeries.setMarkers(allMarkers);
      // Dessine zone TP (vert) + zone SL (rouge) en price lines
      drawTradeZone(s);
    }
    logSetup(s);
    updateStats();
  });
  eventSource.addEventListener('done', e => {
    document.getElementById('info').textContent = '✓ Replay terminé';
    stopReplay();
  });
  eventSource.addEventListener('error', e => {
    document.getElementById('info').textContent = '⚠ Connexion fermée';
    stopReplay();
  });
}

async function togglePause() {
  paused = !paused;
  document.getElementById('pause').textContent = paused ? '▶ Reprendre' : '⏸ Pause';
  // On informe le SERVEUR pour qu'il pause/reprenne lui-meme le stream
  try {
    await fetch(paused ? '/api/replay/pause' : '/api/replay/resume', { method: 'POST' });
    document.getElementById('info').textContent = paused
      ? '⏸ En pause (le serveur a stoppé le stream)'
      : '▶ Replay en cours';
  } catch (e) {
    console.error('Erreur pause/resume:', e);
  }
}

async function stopReplay() {
  // Informe le serveur d'arrêter le stream proprement
  if (eventSource) {
    try { await fetch('/api/replay/stop', { method: 'POST' }); } catch (e) {}
    eventSource.close();
    eventSource = null;
  }
  document.getElementById('play').disabled = false;
  document.getElementById('pause').disabled = true;
  document.getElementById('stop').disabled = true;
  document.getElementById('pause').textContent = '⏸ Pause';
  paused = false;
}

function updateStats() {
  document.getElementById('stats').innerHTML = `
    <span><span class="label">Bougies :</span> <span class="val">${stats.candles}</span></span>
    <span><span class="label">OB candidats :</span> <span class="val">${stats.candidates}</span></span>
    <span><span class="label">Trades :</span> <span class="val" style="color:#3fb950">${stats.trades}</span></span>
  `;
}

function logSetup(s) {
  const ts = new Date(s.validation_ts).toLocaleString('fr-FR', { hour12: false });
  const cls = s.verdict === 'TRADE' ? 'trade' : 'reject';
  const dirBadge = s.direction === 'bullish'
    ? '<span class="badge bull">▲ LONG</span>'
    : '<span class="badge bear">▼ SHORT</span>';
  const verdictBadge = s.verdict === 'TRADE'
    ? `<span class="badge score">SCORE ${s.score}</span>`
    : '<span class="badge reject">REJET</span>';
  const reason = s.rejection_reason ? ` — ${s.rejection_reason}` : '';
  const confl = (s.confluences || []).slice(0, 6).join(', ');
  const row = document.createElement('div');
  row.className = `row ${cls}`;
  row.innerHTML = `${ts} ${dirBadge} ${verdictBadge} ${confl}${reason}`;
  const log = document.getElementById('log');
  log.insertBefore(row, log.firstChild);
  while (log.children.length > 100) log.removeChild(log.lastChild);
}

async function loadStatic() {
  stopReplay();
  const inst = document.getElementById('instrument').value;
  const tf = document.getElementById('tf').value;
  const days = document.getElementById('days').value;
  document.getElementById('info').textContent = 'Chargement statique...';

  try {
    const candles = await fetch(`/api/candles?instrument=${inst}&tf=${tf}&days=${days}`).then(r => r.json());
    candleSeries.setData(candles);
    chart.timeScale().fitContent();
    stats.candles = candles.length;
    updateStats();
  } catch (e) {
    document.getElementById('info').textContent = 'Erreur : ' + e.message;
    return;
  }

  fetch(`/api/setups?instrument=${inst}&days=${days}`).then(r => r.json()).then(setups => {
    const markers = setups.filter(s => s.verdict === 'TRADE').map(s => ({
      time: Math.floor(new Date(s.validation_ts).getTime() / 1000),
      position: s.direction === 'bullish' ? 'belowBar' : 'aboveBar',
      color: s.direction === 'bullish' ? '#3fb950' : '#f85149',
      shape: s.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
      text: String(s.score),
    }));
    candleSeries.setMarkers(markers);
    setups.forEach(logSetup);
    stats.trades = setups.filter(s => s.verdict === 'TRADE').length;
    stats.candidates = setups.length;
    updateStats();
    document.getElementById('info').textContent =
      `${stats.trades} trades / ${stats.candidates} candidats`;
  });
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    from fastapi.responses import Response
    return Response(
        content=HTML,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/instruments")
def instruments():
    return [
        {"symbol": k, "label": v["label"], "type": v.get("type", "")}
        for k, v in INSTRUMENTS.items()
        if v["role"] == "primary"
    ]


@app.get("/api/candles")
def candles(instrument: str = Query(...), tf: str = Query("M5"), days: int = Query(7)):
    df = load(instrument, tf)
    mask = df.index >= (df.index.max() - pd.Timedelta(days=days))
    df = df[mask]
    return [
        {"time": int(t.timestamp()), "open": float(r["open"]), "high": float(r["high"]),
         "low": float(r["low"]), "close": float(r["close"])}
        for t, r in df.iterrows()
    ]


@app.get("/api/setups")
def setups(instrument: str = Query(...), days: int = Query(7)):
    results = run_pipeline(
        instrument, ltf_name="M5", htf_name="H1", htf2_name=None,
        days=days, min_score=0, min_quality=0,
    )
    out = []
    for r in results:
        s = {
            "validation_ts": r.ob.validation_ts.isoformat(),
            "direction": r.ob.direction,
            "score": r.score,
            "verdict": r.verdict,
            "confluences": r.confluences,
            "rejection_reason": r.rejection_reason,
        }
        out.append(s)
    return out


@app.get("/api/backtest")
def backtest(instrument: str = Query(...), days: int = Query(30)):
    rpt = backtest_instrument(instrument, days=days)
    return {
        "instrument": rpt.instrument,
        "n_trades": rpt.n_trades,
        "n_wins": rpt.n_wins,
        "n_losses": rpt.n_losses,
        "n_no_fill": rpt.n_no_fill,
        "n_pending": rpt.n_pending,
        "win_rate": rpt.win_rate,
        "profit_factor": rpt.profit_factor,
        "return_pct": rpt.return_pct,
        "max_drawdown_pct": rpt.max_drawdown_pct,
        "final_balance": rpt.final_balance,
    }


# ============ REPLAY LIVE (Server-Sent Events) ============

@app.get("/api/replay")
async def replay(
    instrument: str = Query(...),
    tf: str = Query("M5"),
    days: int = Query(3),
    speed: int = Query(20),       # ms entre bougies
):
    """Stream les bougies une par une + setups detectes au fur et a mesure.

    Logique :
    1. Charge le LTF (et HTF/D1 pour le pipeline).
    2. Pour chaque bougie i dans la fenetre :
       - Envoie la bougie (event "candle")
       - Detecte les OB sur l'historique [0..i]
       - Si nouvel OB valide (validation_ts == cette bougie) -> envoie event "setup"
    3. Envoie "done" a la fin.
    """
    df_ltf = load(instrument, tf)
    mask = df_ltf.index >= (df_ltf.index.max() - pd.Timedelta(days=days))
    df_ltf = df_ltf[mask]

    # HTF et D1 pour le pipeline
    htf_map = {"M1": "M15", "M5": "H1", "M15": "H1", "H1": "D1"}
    htf_name = htf_map.get(tf, "H1")
    try:
        df_htf = load(instrument, htf_name)
        mask_h = df_htf.index >= (df_htf.index.max() - pd.Timedelta(days=days + 5))
        df_htf = df_htf[mask_h]
    except Exception:
        df_htf = df_ltf

    try:
        df_d1 = load(instrument, "D1")
    except Exception:
        df_d1 = build_d1_from_h1(df_htf)

    delay = max(speed, 1) / 1000.0  # secondes

    # DETECTION HONNETE : on parcourt chaque bougie, et a chaque bougie on relance
    # detect_order_blocks sur df_ltf[:i+1] (= PASSE seulement, pas de future).
    # C'est plus lent mais c'est ce qu'un bot live ferait reellement.
    # On note pour chaque (direction, group_start, group_end, validation_index) la PREMIERE
    # bougie i ou il est detecte : c'est a CE moment qu'on l'emet dans le stream.
    #
    # Le resultat "outcome" (WIN/LOSS) est calcule a posteriori avec simulate_trade
    # (on a forcement besoin du futur pour savoir si SL ou TP touche), mais c'est emis
    # AVEC le setup au moment de la detection -> coloration des rectangles.

    setups_by_index: dict[int, list] = {}
    detected_keys: set = set()

    for i in range(5, len(df_ltf)):
        df_seen = df_ltf.iloc[: i + 1]
        current_ts = df_ltf.index[i]
        # On tronque HTF et D1 au temps present pour eviter le look-ahead
        df_htf_seen = df_htf[df_htf.index <= current_ts]
        df_d1_seen = df_d1[df_d1.index <= current_ts]

        try:
            obs_now = detect_order_blocks(df_seen)
        except Exception:
            continue

        for ob in obs_now:
            key = (ob.direction, ob.group_start_index, ob.group_end_index, ob.validation_index)
            if key in detected_keys:
                continue
            # Filtre : on ne traite que les OB dont la validation est sur la bougie courante i
            # (= la bougie qui vient d'arriver). Sinon, c'est qu'on les voit pour la premiere fois
            # mais ils se sont valides bien avant -> trop tard pour les jouer.
            if ob.validation_index != i:
                detected_keys.add(key)
                continue
            detected_keys.add(key)

            try:
                # IMPORTANT : on evalue avec df_seen + df_htf_seen + df_d1_seen (= passe uniquement)
                res = evaluate_ob(
                    ob, df_seen, df_htf_seen, df_d1_seen, instrument,
                    ltf_name=tf, htf_name=htf_name,
                    df_htf2=None, htf2_name=None,
                    min_score=0, min_quality=0,
                )
                setup_event = {
                    "validation_ts": ob.validation_ts.isoformat(),
                    "validation_time": int(ob.validation_ts.timestamp()),
                    "direction": ob.direction,
                    "score": res.score,
                    "verdict": res.verdict,
                    "confluences": res.confluences,
                    "rejection_reason": res.rejection_reason,
                    "ob_high": float(ob.ob_high),
                    "ob_low": float(ob.ob_low),
                }
                if res.trade_setup is not None:
                    setup_event["entry"] = float(res.trade_setup.entry_price)
                    setup_event["sl"] = float(res.trade_setup.stop_loss)
                    setup_event["tp"] = float(res.trade_setup.take_profit)
                    setup_event["rr"] = float(res.trade_setup.rr)
                    # Simulation a posteriori : on a besoin du futur pour savoir l'outcome,
                    # mais c'est OK car c'est l'equivalent de "si le bot avait pris ce trade,
                    # voici ce qui se serait passe". Pas de cheat sur la decision elle-meme.
                    try:
                        sim = simulate_trade(res.trade_setup, df_ltf, ob.validation_index + 1)
                        setup_event["outcome"] = sim.outcome
                        if sim.fill_ts is not None:
                            setup_event["fill_time"] = int(sim.fill_ts.timestamp())
                        if sim.exit_ts is not None:
                            setup_event["exit_time"] = int(sim.exit_ts.timestamp())
                        if sim.exit_price is not None:
                            setup_event["exit_price"] = float(sim.exit_price)
                        setup_event["pnl_usd"] = float(sim.pnl_usd)
                    except Exception:
                        setup_event["outcome"] = "PENDING"
                setups_by_index.setdefault(i, []).append(setup_event)
            except Exception:
                pass

    # Reset etat replay
    REPLAY.paused = False
    REPLAY.stop = False

    async def event_stream():
        # On stream un message "ready" en premier pour confirmer au client que ca demarre
        yield f"event: ready\ndata: {json.dumps({'total': len(df_ltf), 'setups': sum(len(v) for v in setups_by_index.values())})}\n\n"

        for i in range(len(df_ltf)):
            # Stop demande par le client
            if REPLAY.stop:
                break

            # Pause : on attend tant que paused=True, sans envoyer de bougies
            while REPLAY.paused and not REPLAY.stop:
                await asyncio.sleep(0.1)
            if REPLAY.stop:
                break

            row = df_ltf.iloc[i]
            ts = df_ltf.index[i]
            candle = {
                "time": int(ts.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            yield f"event: candle\ndata: {json.dumps(candle)}\n\n"

            # Si des OB ont leur validation EXACTEMENT sur cette bougie => on emet les events
            if i in setups_by_index:
                for se in setups_by_index[i]:
                    yield f"event: setup\ndata: {json.dumps(se)}\n\n"

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
    REPLAY.paused = False
    return {"stopped": True}


if __name__ == "__main__":
    import uvicorn
    print(f"Serveur Bot V2 : http://{HOST}:8001")
    uvicorn.run(app, host=HOST, port=8001)
