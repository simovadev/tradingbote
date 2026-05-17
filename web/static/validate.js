// Validator : 1 trade a la fois, Y/N/skip + EDIT
function validator() {
  return {
    instruments: ["XAUUSD", "NAS100", "GER40", "USOIL"],
    filterInstrument: "",
    current: null,
    candles: [],
    loading: false,
    saving: false,
    currentTool: "none",
    isEdited: false,
    comment: "",
    revealed: false,                     // false = mode realiste (futur cache), true = on voit tout
    allCandles: [],                       // toutes les bougies (24h)
    genStatus: { current_generation: 1, validations_in_current_gen: 0 },

    // Valeurs editables (live)
    edited: {
      direction: "bullish",
      entry_price: 0,
      stop_loss: 0,
      take_profit: 0,
      ob_zone: { high: 0, low: 0 },
    },

    chart: null,
    series: null,
    priceLines: [],

    stats: {
      total: 0, yes: 0, no: 0, skip: 0, edited: 0,
      yes_winrate: 0, yes_expectancy_r: 0, no_winrate: 0,
    },

    isInputFocused(e) {
      return e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT";
    },

    async boot() {
      this.initChart();
      await this.refreshStats();
      await this.refreshGenStatus();
      await this.next();
      window.addEventListener("resize", () => this.fitChart());
    },

    async refreshGenStatus() {
      try {
        const r = await fetch("/api/validate/generation-status");
        if (r.ok) this.genStatus = await r.json();
      } catch (e) {}
    },

    async retrain() {
      if (!confirm(`Lancer le retrain ? Le bot va ajuster ses filtres et passer a la Gen ${(this.genStatus.current_generation || 1) + 1}.`)) return;
      const r = await fetch("/api/validate/retrain", { method: "POST" });
      const data = await r.json();
      if (!data.ok) { alert("Retrain echoue : " + data.reason); return; }
      let msg = `✓ Retrain effectue !\n\nGen ${data.old_generation} -> ${data.new_generation}\n\n`;
      msg += `Stats Gen ${data.old_generation}:\n`;
      msg += `  - YES: ${data.stats.yes} / NO: ${data.stats.no} / Skip: ${data.stats.skip}\n`;
      msg += `  - YES winrate: ${data.stats.yes_winrate}%\n`;
      msg += `  - Expectancy: ${data.stats.yes_expectancy_r}R\n\n`;
      msg += `Changements de parametres:\n`;
      msg += data.changes.length ? data.changes.map(c => "  - " + c).join("\n") : "  (aucun ajustement significatif)";
      alert(msg);
      await this.refreshGenStatus();
      await this.next();
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
      // Click sur chart pour ajuster un niveau
      this.chart.subscribeClick((param) => this.onChartClick(param));
    },

    onChartClick(param) {
      if (this.currentTool === "none" || !param.point) return;
      const price = this.series.coordinateToPrice(param.point.y);
      if (price == null) return;
      const rounded = Math.round(price * 100) / 100;

      if (this.currentTool === "entry") this.edited.entry_price = rounded;
      else if (this.currentTool === "sl") this.edited.stop_loss = rounded;
      else if (this.currentTool === "tp") this.edited.take_profit = rounded;
      else if (this.currentTool === "ob_high") this.edited.ob_zone.high = rounded;
      else if (this.currentTool === "ob_low") this.edited.ob_zone.low = rounded;

      this.onEdit();
    },

    onEdit() {
      // Verifie si quelque chose a change par rapport a current
      if (!this.current) return;
      const same = (
        this.edited.direction === this.current.direction &&
        Math.abs(this.edited.entry_price - this.current.entry_price) < 0.001 &&
        Math.abs(this.edited.stop_loss - this.current.stop_loss) < 0.001 &&
        Math.abs(this.edited.take_profit - this.current.take_profit) < 0.001 &&
        Math.abs((this.edited.ob_zone.high || 0) - (this.current.ob_zone?.high || 0)) < 0.001 &&
        Math.abs((this.edited.ob_zone.low || 0) - (this.current.ob_zone?.low || 0)) < 0.001
      );
      this.isEdited = !same;
      this.redraw();
    },

    resetToOriginal() {
      if (!this.current) return;
      this.edited = {
        direction: this.current.direction,
        entry_price: this.current.entry_price,
        stop_loss: this.current.stop_loss,
        take_profit: this.current.take_profit,
        ob_zone: {
          high: this.current.ob_zone?.high || 0,
          low: this.current.ob_zone?.low || 0,
        },
      };
      this.isEdited = false;
      this.redraw();
    },

    computedRR() {
      const e = this.edited.entry_price, s = this.edited.stop_loss, t = this.edited.take_profit;
      if (!e || !s || !t) return null;
      const r = Math.abs(e - s);
      const rew = Math.abs(t - e);
      return r > 0 ? rew / r : null;
    },

    async refreshStats() {
      try {
        const r = await fetch("/api/validate/stats");
        if (r.ok) this.stats = await r.json();
      } catch (e) {}
    },

    async next() {
      this.loading = true;
      this.isEdited = false;
      this.comment = "";
      try {
        const url = this.filterInstrument
          ? `/api/validate/next?instrument=${this.filterInstrument}`
          : "/api/validate/next";
        const r = await fetch(url);
        if (!r.ok) { alert("Pas de setup à valider"); return; }
        this.current = await r.json();
        this.resetToOriginal();
        await this.loadCandles();
      } finally {
        this.loading = false;
      }
    },

    async loadCandles() {
      if (!this.current) return;
      const r = await fetch(`/api/labeling/candles?instrument=${this.current.instrument}&day=${this.current.day}&tf=M1`);
      this.allCandles = r.ok ? await r.json() : [];
      this.applyRealistic();
      this.fitChart();
      this.redraw();
    },

    applyRealistic() {
      if (this.revealed || !this.current?.visible_until_unix) {
        this.candles = this.allCandles;
      } else {
        const cutoff = this.current.visible_until_unix;
        this.candles = this.allCandles.filter(c => c.time <= cutoff);
      }
      this.series.setData(this.candles);
      // Force le rescale de l'axe Y sur les nouvelles donnees
      try {
        this.chart.priceScale("right").applyOptions({ autoScale: true });
      } catch (e) {}
    },

    fitChart() {
      if (!this.chart || !this.current || !this.candles.length) return;
      const entryUnix = Math.floor(new Date(this.current.entry_time).getTime() / 1000);
      let bestIdx = 0;
      let bestDiff = Infinity;
      for (let i = 0; i < this.candles.length; i++) {
        const d = Math.abs(this.candles[i].time - entryUnix);
        if (d < bestDiff) { bestDiff = d; bestIdx = i; }
      }
      const fromIdx = Math.max(0, bestIdx - 90);
      const toIdx = Math.min(this.candles.length - 1, bestIdx + 60);
      try {
        this.chart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: toIdx });
        // Force le rescale Y sur la fenetre visible
        this.chart.priceScale("right").applyOptions({ autoScale: true });
      } catch (e) {
        console.warn("fitChart failed:", e);
      }
    },

    onToggleRevealed() {
      this.applyRealistic();
      this.redraw();
    },

    redraw() {
      // Re-applique le mode realiste a chaque redraw au cas ou
      this.applyRealistic();
      // Supprime anciennes priceLines
      for (const pl of this.priceLines) {
        try { this.series.removePriceLine(pl); } catch (e) {}
      }
      this.priceLines = [];
      if (!this.current) return;

      // Si edite : on affiche les originales en pointille leger + les edited en plein
      if (this.isEdited) {
        // Originales (light)
        this.priceLines.push(this.series.createPriceLine({
          price: this.current.entry_price, color: "rgba(59, 130, 246, 0.3)", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: false, title: "bot E",
        }));
        this.priceLines.push(this.series.createPriceLine({
          price: this.current.stop_loss, color: "rgba(239, 68, 68, 0.3)", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: false, title: "bot SL",
        }));
        this.priceLines.push(this.series.createPriceLine({
          price: this.current.take_profit, color: "rgba(16, 185, 129, 0.3)", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: false, title: "bot TP",
        }));
      }

      // Edited (plein)
      this.priceLines.push(this.series.createPriceLine({
        price: this.edited.entry_price, color: "#3b82f6", lineWidth: 2, lineStyle: 0,
        axisLabelVisible: true, title: "ENTRY",
      }));
      this.priceLines.push(this.series.createPriceLine({
        price: this.edited.stop_loss, color: "#ef4444", lineWidth: 2, lineStyle: 0,
        axisLabelVisible: true, title: "SL",
      }));
      this.priceLines.push(this.series.createPriceLine({
        price: this.edited.take_profit, color: "#10b981", lineWidth: 2, lineStyle: 0,
        axisLabelVisible: true, title: "TP",
      }));

      // OB zone
      if (this.edited.ob_zone.high && this.edited.ob_zone.low) {
        this.priceLines.push(this.series.createPriceLine({
          price: this.edited.ob_zone.high, color: "#fbbf24", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: "OB top",
        }));
        this.priceLines.push(this.series.createPriceLine({
          price: this.edited.ob_zone.low, color: "#fbbf24", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: "OB bot",
        }));
      }

      // Marker direction
      this.drawMarker();
    },

    drawMarker() {
      if (!this.current || !this.candles.length) return;
      const markers = [];

      // 1. Markers sur les bougies du push (les OB du bot) — cercles jaunes
      const obCandles = this.current.ob_candles || [];
      for (const oc of obCandles) {
        // Trouve la bougie la plus proche
        let bestTime = oc.time;
        let bestDiff = Infinity;
        for (const c of this.candles) {
          const d = Math.abs(c.time - oc.time);
          if (d < bestDiff) { bestDiff = d; bestTime = c.time; }
        }
        markers.push({
          time: bestTime,
          position: "inBar",
          color: "#fbbf24",
          shape: "circle",
          text: "OB",
          size: 2,
        });
      }

      // 2. Marker de l'entry (fleche directionnelle)
      const entryUnix = Math.floor(new Date(this.current.entry_time).getTime() / 1000);
      let bestTime = entryUnix;
      let bestDiff = Infinity;
      for (const c of this.candles) {
        const d = Math.abs(c.time - entryUnix);
        if (d < bestDiff) { bestDiff = d; bestTime = c.time; }
      }
      const isLong = this.edited.direction === "bullish";
      markers.push({
        time: bestTime,
        position: isLong ? "belowBar" : "aboveBar",
        color: isLong ? "#3b82f6" : "#a855f7",
        shape: isLong ? "arrowUp" : "arrowDown",
        text: `ENTRY ${new Date(entryUnix * 1000).toUTCString().slice(17, 22)}`,
        size: 2,
      });

      // Sort par time (LightweightCharts exige ordre chronologique)
      markers.sort((a, b) => a.time - b.time);
      this.series.setMarkers(markers);
    },

    async validate(verdict) {
      if (!this.current || this.saving) return;
      // Si edite et l'user clique YES, on force "edited"
      if (this.isEdited && verdict === "yes") verdict = "edited";
      await this._save(verdict);
    },

    async saveEdited() {
      if (!this.current || this.saving) return;
      if (!this.isEdited) {
        // Rien d'edite : on traite comme YES
        await this._save("yes");
        return;
      }
      await this._save("edited");
    },

    async _save(verdict) {
      this.saving = true;
      try {
        const payload = {
          instrument: this.current.instrument,
          day: this.current.day,
          direction: this.edited.direction,
          entry_time: this.current.entry_time,
          entry_price: this.edited.entry_price,
          stop_loss: this.edited.stop_loss,
          take_profit: this.edited.take_profit,
          risk_reward: this.computedRR() || this.current.risk_reward,
          ob_zone: {
            high: this.edited.ob_zone.high,
            low: this.edited.ob_zone.low,
            from_time: this.current.ob_zone?.from_time,
            to_time: this.current.ob_zone?.to_time,
          },
          features: this.current.features,
          outcome: this.isEdited ? "recompute" : this.current.outcome,
          verdict: verdict,
          original: this.isEdited ? {
            direction: this.current.direction,
            entry_price: this.current.entry_price,
            stop_loss: this.current.stop_loss,
            take_profit: this.current.take_profit,
            ob_zone: this.current.ob_zone,
          } : {},
          comment: this.comment || "",
        };
        const r = await fetch("/api/validate/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!r.ok) { alert("Erreur sauvegarde"); return; }
        await this.refreshStats();
        await this.refreshGenStatus();
        await this.next();
      } finally {
        this.saving = false;
      }
    },
  };
}
