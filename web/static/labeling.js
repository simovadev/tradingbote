// TradingBot Labeling - utilise l'API NATIVE de Lightweight Charts
// Pas de canvas custom, on utilise priceLines + markers + createSeriesPrimitive
function labeler() {
  return {
    instruments: ["XAUUSD", "NAS100", "GER40", "USOIL"],
    currentInstrument: "XAUUSD",
    currentDay: new Date().toISOString().slice(0, 10),
    currentTf: "M1",
    availableDays: [],

    chart: null,
    series: null,
    candles: [],

    // Lignes de prix natives (gerees par lightweight-charts)
    priceLines: { entry: null, sl: null, tp: null },
    savedPriceLines: [],   // lignes des trades deja sauves
    algoPriceLines: [],    // lignes des trades algo

    currentTool: "none",
    saving: false,
    lastSavedMsg: "",

    trade: {
      direction: "bullish",
      entry_price: null,
      stop_loss: null,
      take_profit: null,
      ob_zone: { high: null, low: null, t1: null, t2: null },
      ob_candles: { timeframe: "M1", candles: [] },
      reasoning: "",
    },

    dayTrades: [],
    algoTrades: [],
    showAlgoOnChart: false,    // Par défaut : chart propre, sans pollution algo
    hoveredCandle: null,

    currentToolLabel() {
      return ({
        ob_candle: "🟧 Click sur une BOUGIE precise pour la marquer OB",
        entry: "📏 Click au prix EXACT desire pour Entry",
        sl: "🔴 Click au prix EXACT desire pour SL",
        tp: "🟢 Click au prix EXACT desire pour TP",
        none: "✋ Navigation",
      })[this.currentTool] || "";
    },

    async boot() {
      this.initChart();
      await this.loadAvailableDays();
      await this.randomDay();
      window.addEventListener("resize", () => this.fitChart());
    },

    async loadAvailableDays() {
      try {
        const res = await fetch(`/api/labeling/available-days?instrument=${this.currentInstrument}`);
        this.availableDays = await res.json();
      } catch (e) { this.availableDays = []; }
    },

    async loadAll() {
      await this.loadChart();
      await this.refreshDayTrades();
      await this.refreshAlgoTrades();
    },

    async randomDay() {
      const res = await fetch(`/api/labeling/random-day?instrument=${this.currentInstrument}`);
      if (!res.ok) { alert("Pas de données pour " + this.currentInstrument); return; }
      const data = await res.json();
      this.currentDay = data.day;
      this.availableDays = data.all_days;
      this.resetTradeDraft();
      await this.loadAll();
    },

    async onInstrumentChange() {
      this.resetTradeDraft();
      await this.loadAvailableDays();
      if (!this.availableDays.includes(this.currentDay)) {
        await this.randomDay();
      } else {
        await this.loadAll();
      }
    },

    async nextInstrument() {
      const i = this.instruments.indexOf(this.currentInstrument);
      this.currentInstrument = this.instruments[(i + 1) % this.instruments.length];
      await this.onInstrumentChange();
    },

    async switchTf(tf) {
      this.currentTf = tf;
      this.trade.ob_candles.timeframe = tf;
      await this.loadChart();
    },

    initChart() {
      const el = document.getElementById("chart");
      this.chart = LightweightCharts.createChart(el, {
        layout: { background: { color: "#0f172a" }, textColor: "#94a3b8" },
        grid: { vertLines: { color: "#1e293b" }, horzLines: { color: "#1e293b" } },
        timeScale: { timeVisible: true, secondsVisible: false, borderColor: "#334155" },
        rightPriceScale: { borderColor: "#334155" },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      });
      this.series = this.chart.addCandlestickSeries({
        upColor: "#10b981", downColor: "#ef4444",
        wickUpColor: "#10b981", wickDownColor: "#ef4444",
        borderVisible: false,
      });

      // CRUCIAL : subscribe aux clicks via l'API native
      this.chart.subscribeClick((param) => this.handleChartClick(param));
      this.chart.subscribeCrosshairMove((param) => this.handleCrosshairMove(param));
    },

    handleChartClick(param) {
      if (this.currentTool === "none") return;
      if (!param.point) return;

      // Prix exact où le user a cliqué
      const clickedPrice = this.series.coordinateToPrice(param.point.y);
      if (clickedPrice == null) return;

      // Bougie au time du click (gère format number ou BusinessDay)
      const candle = this.findCandleAtTime(param.time);

      if (this.currentTool === "ob_candle") {
        if (!candle) return;
        this.toggleOBCandle(candle);
        return;
      }

      // Entry / SL / TP : on prend le PRIX EXACT où tu as cliqué
      // (plus de snap auto confus - tu cliques, tu obtiens ce prix)
      const finalPrice = this.roundPrice(clickedPrice);

      if (this.currentTool === "entry") {
        this.trade.entry_price = finalPrice;
      } else if (this.currentTool === "sl") {
        this.trade.stop_loss = finalPrice;
      } else if (this.currentTool === "tp") {
        this.trade.take_profit = finalPrice;
      }
      this.updatePriceLines();
    },

    // Round propre selon l'instrument (XAU = 2 decimales, indices = 2, oil = 2)
    roundPrice(p) {
      return Math.round(p * 100) / 100;
    },

    // Trouve la bougie pour un time donne (gere number, BusinessDay, etc.)
    findCandleAtTime(t) {
      if (t == null) return null;
      // t peut etre un number (seconds) ou un object {year, month, day}
      if (typeof t === "number") {
        return this.candles.find(c => c.time === t);
      }
      if (typeof t === "object" && t.year != null) {
        // Convert BusinessDay to seconds Unix midnight UTC
        const dateMs = Date.UTC(t.year, (t.month || 1) - 1, t.day || 1);
        const sec = Math.floor(dateMs / 1000);
        return this.candles.find(c => c.time === sec);
      }
      return null;
    },

    handleCrosshairMove(param) {
      if (!param.time) {
        this.hoveredCandle = null;
        return;
      }
      const c = this.candles.find(c => c.time === param.time);
      if (c) {
        this.hoveredCandle = {
          ...c,
          iso: new Date(c.time * 1000).toISOString(),
        };
      }
    },

    toggleOBCandle(candle) {
      const exists = this.trade.ob_candles.candles.find(c => c.time === candle.time);
      if (exists) {
        this.trade.ob_candles.candles = this.trade.ob_candles.candles.filter(c => c.time !== candle.time);
      } else {
        this.trade.ob_candles.candles.push({
          time: candle.time, open: candle.open, high: candle.high,
          low: candle.low, close: candle.close,
        });
        this.trade.ob_candles.candles.sort((a, b) => a.time - b.time);
      }
      this.recomputeOBZone();
      this.updateMarkers();
      this.updatePriceLines();
    },

    snapToCandle(candle, clickedPrice) {
      const mid = (candle.high + candle.low) / 2;
      const levels = [candle.high, candle.low, candle.open, candle.close, mid];
      let best = levels[0], bestDist = Math.abs(levels[0] - clickedPrice);
      for (const l of levels) {
        const d = Math.abs(l - clickedPrice);
        if (d < bestDist) { bestDist = d; best = l; }
      }
      return best;
    },

    recomputeOBZone() {
      const candles = this.trade.ob_candles.candles;
      if (candles.length === 0) {
        this.trade.ob_zone = { high: null, low: null, t1: null, t2: null };
        return;
      }
      const high = Math.max(...candles.map(c => c.high));
      const low = Math.min(...candles.map(c => c.low));
      this.trade.ob_zone = {
        high, low,
        t1: new Date(candles[0].time * 1000).toISOString(),
        t2: new Date(candles[candles.length - 1].time * 1000).toISOString(),
      };
    },

    // ============ Markers : OB candles + entry markers (bot + user) ============
    updateMarkers() {
      const markers = [];

      // 1. Bougies OB du draft en cours (cercle jaune)
      for (const c of this.trade.ob_candles.candles) {
        markers.push({
          time: c.time, position: "inBar", color: "#fbbf24",
          shape: "circle", text: "OB", size: 2,
        });
      }

      // 2. Entry markers des TRADES USER SAUVEGARDES (etoile bleue avec heure)
      for (const t of this.dayTrades) {
        const refTime = t.ob_zone?.t2 || t.ob_zone?.t1;
        if (!refTime) continue;
        const entryUnix = Math.floor(new Date(refTime).getTime() / 1000);
        // Trouve la bougie la plus proche
        const closest = this._findClosestCandleTime(entryUnix);
        if (closest == null) continue;
        const hour = new Date(entryUnix * 1000).toUTCString().slice(17, 22);
        const isLong = t.direction === "bullish";
        markers.push({
          time: closest,
          position: isLong ? "belowBar" : "aboveBar",
          color: isLong ? "#3b82f6" : "#a855f7",
          shape: isLong ? "arrowUp" : "arrowDown",
          text: `TON ${isLong ? "LONG" : "SHORT"} #${t.id} ${hour}`,
          size: 2,
        });
      }

      // 3. Entry markers des TRADES BOT (fleche colore avec heure)
      if (this.showAlgoOnChart) {
        for (const t of this.algoTrades) {
          if (!t.entry_time) continue;
          const entryUnix = Math.floor(new Date(t.entry_time).getTime() / 1000);
          const closest = this._findClosestCandleTime(entryUnix);
          if (closest == null) continue;
          const hour = new Date(entryUnix * 1000).toUTCString().slice(17, 22);
          const isLong = t.direction === "bullish";
          markers.push({
            time: closest,
            position: isLong ? "belowBar" : "aboveBar",
            color: isLong ? "rgba(16, 185, 129, 0.7)" : "rgba(239, 68, 68, 0.7)",
            shape: isLong ? "arrowUp" : "arrowDown",
            text: `BOT ${hour}`,
            size: 1,
          });
        }
      }

      // Sort by time (lightweight charts exige tri croissant)
      markers.sort((a, b) => a.time - b.time);
      this.series.setMarkers(markers);
    },

    _findClosestCandleTime(targetUnix) {
      if (!this.candles.length) return null;
      let bestTime = null;
      let bestDiff = Infinity;
      for (const c of this.candles) {
        const d = Math.abs(c.time - targetUnix);
        if (d < bestDiff) { bestDiff = d; bestTime = c.time; }
      }
      return bestTime;
    },

    // ============ Price lines : entry / sl / tp + lignes OB zone ============
    updatePriceLines() {
      // Supprime anciennes lignes draft
      for (const key of ["entry", "sl", "tp", "obHigh", "obLow"]) {
        if (this.priceLines[key]) {
          try { this.series.removePriceLine(this.priceLines[key]); } catch (e) {}
          this.priceLines[key] = null;
        }
      }

      // Entry
      if (this.trade.entry_price != null) {
        this.priceLines.entry = this.series.createPriceLine({
          price: this.trade.entry_price,
          color: "#3b82f6", lineWidth: 2, lineStyle: 0,
          axisLabelVisible: true, title: "ENTRY",
        });
      }
      // SL
      if (this.trade.stop_loss != null) {
        this.priceLines.sl = this.series.createPriceLine({
          price: this.trade.stop_loss,
          color: "#ef4444", lineWidth: 2, lineStyle: 0,
          axisLabelVisible: true, title: "SL",
        });
      }
      // TP
      if (this.trade.take_profit != null) {
        this.priceLines.tp = this.series.createPriceLine({
          price: this.trade.take_profit,
          color: "#10b981", lineWidth: 2, lineStyle: 0,
          axisLabelVisible: true, title: "TP",
        });
      }
      // OB zone (haut + bas avec fill via 2 priceLines pointillées)
      if (this.trade.ob_zone.high != null && this.trade.ob_zone.low != null) {
        this.priceLines.obHigh = this.series.createPriceLine({
          price: this.trade.ob_zone.high,
          color: "#fbbf24", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: "OB top",
        });
        this.priceLines.obLow = this.series.createPriceLine({
          price: this.trade.ob_zone.low,
          color: "#fbbf24", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: "OB bot",
        });
      }
    },

    drawSavedAndAlgoLines() {
      // Supprime anciennes
      for (const pl of this.savedPriceLines) {
        try { this.series.removePriceLine(pl); } catch (e) {}
      }
      this.savedPriceLines = [];
      for (const pl of this.algoPriceLines) {
        try { this.series.removePriceLine(pl); } catch (e) {}
      }
      this.algoPriceLines = [];

      // Trades sauvegardés humains (en couleur attenuee)
      for (const t of this.dayTrades) {
        const idLbl = `#${t.id}`;
        if (t.entry_price) this.savedPriceLines.push(this.series.createPriceLine({
          price: t.entry_price, color: "rgba(59, 130, 246, 0.5)", lineWidth: 1, lineStyle: 1,
          axisLabelVisible: false, title: `${idLbl} E`,
        }));
        if (t.stop_loss) this.savedPriceLines.push(this.series.createPriceLine({
          price: t.stop_loss, color: "rgba(239, 68, 68, 0.5)", lineWidth: 1, lineStyle: 1,
          axisLabelVisible: false, title: `${idLbl} SL`,
        }));
        if (t.take_profit) this.savedPriceLines.push(this.series.createPriceLine({
          price: t.take_profit, color: "rgba(16, 185, 129, 0.5)", lineWidth: 1, lineStyle: 1,
          axisLabelVisible: false, title: `${idLbl} TP`,
        }));
        if (t.ob_zone?.high) {
          this.savedPriceLines.push(this.series.createPriceLine({
            price: t.ob_zone.high, color: "rgba(251, 191, 36, 0.4)", lineWidth: 1, lineStyle: 2,
            axisLabelVisible: false, title: "",
          }));
          this.savedPriceLines.push(this.series.createPriceLine({
            price: t.ob_zone.low, color: "rgba(251, 191, 36, 0.4)", lineWidth: 1, lineStyle: 2,
            axisLabelVisible: false, title: "",
          }));
        }
      }

      // Trades algo (en couleur typee) - SEULEMENT zones OB, pas les entry lines
      if (this.showAlgoOnChart) {
        for (const t of this.algoTrades) {
          const color = t.direction === "bullish" ? "rgba(16, 185, 129, 0.35)" : "rgba(239, 68, 68, 0.35)";
          if (t.ob_zone?.high && t.ob_zone?.low) {
            this.algoPriceLines.push(this.series.createPriceLine({
              price: t.ob_zone.high, color, lineWidth: 1, lineStyle: 3,
              axisLabelVisible: false, title: "",
            }));
            this.algoPriceLines.push(this.series.createPriceLine({
              price: t.ob_zone.low, color, lineWidth: 1, lineStyle: 3,
              axisLabelVisible: false, title: "",
            }));
          }
        }
      }
    },

    async loadChart() {
      if (!this.currentInstrument || !this.currentDay) return;
      const url = `/api/labeling/candles?instrument=${this.currentInstrument}&day=${this.currentDay}&tf=${this.currentTf}`;
      const res = await fetch(url);
      if (!res.ok) {
        this.candles = []; this.series.setData([]); return;
      }
      this.candles = await res.json();
      this.series.setData(this.candles);
      this.fitChart();
      // Redraw lignes
      this.updatePriceLines();
      this.drawSavedAndAlgoLines();
      this.updateMarkers();
    },

    fitChart() {
      if (!this.chart) return;
      const dayStart = Math.floor(new Date(this.currentDay).getTime() / 1000);
      const dayEnd = dayStart + 86400;
      try {
        this.chart.timeScale().setVisibleRange({ from: dayStart, to: dayEnd });
      } catch (e) {
        this.chart.timeScale().fitContent();
      }
    },

    async refreshDayTrades() {
      const res = await fetch(`/api/labeling/trades?instrument=${this.currentInstrument}&day=${this.currentDay}`);
      this.dayTrades = res.ok ? await res.json() : [];
      this.drawSavedAndAlgoLines();
      this.updateMarkers();
    },

    async refreshAlgoTrades() {
      const res = await fetch(`/api/labeling/algo-trades?instrument=${this.currentInstrument}&day=${this.currentDay}`);
      const all = res.ok ? await res.json() : [];
      all.sort((a, b) => (b.score || 0) - (a.score || 0));
      this.algoTrades = all.slice(0, 10);
      this.drawSavedAndAlgoLines();
      this.updateMarkers();
    },

    redrawAll() {
      this.updatePriceLines();
      this.drawSavedAndAlgoLines();
      this.updateMarkers();
    },

    copyAlgoToDraft(t) {
      this.trade.direction = t.direction;
      this.trade.entry_price = t.entry_price;
      this.trade.stop_loss = t.stop_loss;
      this.trade.take_profit = t.take_profit;
      this.trade.ob_zone = {
        high: t.ob_zone?.high, low: t.ob_zone?.low,
        t1: t.ob_zone?.from_time, t2: t.entry_time,
      };
      this.trade.ob_candles = { timeframe: this.currentTf, candles: [] };
      this.lastSavedMsg = "Trade algo copié. Ajuste puis enregistre.";
      this.updatePriceLines();
      this.updateMarkers();
      // Zoom sur la zone du trade
      this.zoomToTrade(t);
    },

    zoomToTrade(t) {
      if (!t || !t.entry_time) return;
      const entryUnix = Math.floor(new Date(t.entry_time).getTime() / 1000);
      this._scrollChartToTime(entryUnix);
    },

    zoomToUserTrade(t) {
      if (!t || !t.ob_zone) return;
      const ref = t.ob_zone.t2 || t.ob_zone.t1;
      if (!ref) return;
      const refUnix = Math.floor(new Date(ref).getTime() / 1000);
      this._scrollChartToTime(refUnix);
    },

    _scrollChartToTime(targetUnix) {
      if (!this.chart || !this.candles.length) {
        console.warn("[zoom] Chart pas pret ou pas de candles");
        return;
      }
      // Trouve l'index de la bougie la plus proche
      let bestIdx = 0;
      let bestDiff = Infinity;
      for (let i = 0; i < this.candles.length; i++) {
        const diff = Math.abs(this.candles[i].time - targetUnix);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIdx = i;
        }
      }
      console.log(`[zoom] target=${new Date(targetUnix*1000).toISOString()} bestIdx=${bestIdx}/${this.candles.length} bestTime=${new Date(this.candles[bestIdx].time*1000).toISOString()}`);

      const ts = this.chart.timeScale();
      const half = 60;
      const fromIdx = Math.max(0, bestIdx - half);
      const toIdx = Math.min(this.candles.length - 1, bestIdx + half);

      // Methode 1 : scrollToPosition (decalage par rapport a la fin)
      try {
        const offsetFromEnd = this.candles.length - 1 - bestIdx;
        ts.scrollToPosition(-offsetFromEnd, false);
        console.log(`[zoom] scrollToPosition(${-offsetFromEnd})`);
      } catch (e) {
        console.warn("[zoom] scrollToPosition failed:", e);
      }

      // Methode 2 : setVisibleLogicalRange
      try {
        ts.setVisibleLogicalRange({ from: fromIdx, to: toIdx });
        console.log(`[zoom] setVisibleLogicalRange(${fromIdx} -> ${toIdx})`);
      } catch (e) {
        console.warn("[zoom] setVisibleLogicalRange failed:", e);
      }
    },

    setTool(t) {
      this.currentTool = t;
    },

    computedRR() {
      const { entry_price, stop_loss, take_profit } = this.trade;
      if (entry_price == null || stop_loss == null || take_profit == null) return null;
      const risk = Math.abs(entry_price - stop_loss);
      const reward = Math.abs(take_profit - entry_price);
      return risk > 0 ? reward / risk : null;
    },

    canSave() {
      return this.trade.direction &&
             this.trade.entry_price != null &&
             this.trade.stop_loss != null &&
             this.trade.take_profit != null;
    },

    resetTradeDraft() {
      this.trade = {
        direction: "bullish",
        entry_price: null, stop_loss: null, take_profit: null,
        ob_zone: { high: null, low: null, t1: null, t2: null },
        ob_candles: { timeframe: this.currentTf, candles: [] },
        reasoning: "",
      };
      this.lastSavedMsg = "";
      this.updatePriceLines();
      this.updateMarkers();
    },

    async saveTrade() {
      if (!this.canSave()) return;
      this.saving = true;
      this.lastSavedMsg = "";
      try {
        // entry_time = derniere bougie OB selectionnee + 1 min (= moment de l'entree probable)
        // Si pas d'OB selectionne, fallback au debut du jour
        let entry_time = null;
        const obCandles = this.trade.ob_candles.candles || [];
        if (obCandles.length > 0) {
          const lastOB = obCandles[obCandles.length - 1];
          // +1 min apres la derniere bougie OB
          entry_time = new Date((lastOB.time + 60) * 1000).toISOString();
        }

        const res = await fetch("/api/labeling/save-trade", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            instrument: this.currentInstrument,
            day: this.currentDay,
            direction: this.trade.direction,
            entry_price: this.trade.entry_price,
            stop_loss: this.trade.stop_loss,
            take_profit: this.trade.take_profit,
            entry_time: entry_time,
            ob_zone: this.trade.ob_zone,
            ob_candles: this.trade.ob_candles,
            reasoning: this.trade.reasoning,
            extra_drawings: {},
          }),
        });
        if (!res.ok) { alert("Erreur enregistrement"); return; }
        const data = await res.json();
        this.lastSavedMsg = `✓ Trade #${data.id} enregistré - Outcome: ${data.outcome.toUpperCase()} (RR ${data.rr.toFixed(2)})`;
        await this.refreshDayTrades();
        this.resetTradeDraft();
      } finally {
        this.saving = false;
      }
    },

    async deleteTrade(id) {
      if (!confirm("Supprimer ce trade ?")) return;
      await fetch(`/api/labeling/trades/${id}`, { method: "DELETE" });
      await this.refreshDayTrades();
    },
  };
}
