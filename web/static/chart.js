// TradingBot Trainer - mode entrainement humain
function trainer() {
  return {
    sessions: [],
    current: null,
    currentTf: "M1",

    chart: null,
    series: null,

    loading: false,
    saving: false,
    lastSaved: null,

    feedback: { comment: "", user_drawings: { lines: [], rects: [], arrows: [] } },

    currentTool: "none",
    drawingState: null,

    tools: [
      { id: "none",  label: "✋ Pan" },
      { id: "rect",  label: "▭ Zone OB" },
      { id: "line",  label: "─ Liquidite" },
      { id: "arrow", label: "↗ Fleche" },
    ],

    categories: ["temporal", "htf", "liquidity", "structure", "ob_quality", "fvg", "zone", "entry", "volume"],

    categoryLabel(c) {
      return ({
        temporal:   "🕐 Contexte temporel",
        htf:        "📈 Multi-Timeframe",
        liquidity:  "💧 Liquidite",
        structure:  "🔨 Structure",
        ob_quality: "🟧 Qualite OB",
        fvg:        "📊 FVG",
        zone:       "📍 Premium/Discount",
        entry:      "🎯 Entree",
        volume:     "📊 Volume",
      })[c] || c;
    },

    factorsOfCategory(cat) {
      return (this.current?.setup_analysis?.factors || []).filter(f => f.category === cat);
    },

    get hasPrev() {
      if (!this.current) return false;
      return this.current.session_index > 0;
    },

    async boot() {
      this.initChart();
      await this.refreshSessions();
      // Charge automatiquement le dernier trade reviewé si y'en a un
      const last = this.sessions[this.sessions.length - 1];
      if (last) await this.loadTrade(last.id);
      window.addEventListener("resize", () => this.resizeOverlays());
    },

    async refreshSessions() {
      const res = await fetch("/api/training/list");
      this.sessions = await res.json();
    },

    async nextTrade() {
      this.loading = true;
      try {
        const res = await fetch("/api/training/next", { method: "POST" });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Erreur" }));
          alert("Aucun setup trouvé : " + err.detail);
          return;
        }
        this.current = await res.json();
        await this.refreshSessions();
        this.resetFeedback();
        this.currentTf = "M1";
        await this.loadChart();
      } finally {
        this.loading = false;
      }
    },

    async loadTrade(id) {
      const res = await fetch(`/api/training/${id}`);
      if (!res.ok) return;
      this.current = await res.json();
      this.feedback = {
        comment: this.current.user_comment || "",
        user_drawings: this.current.user_drawings || { lines: [], rects: [], arrows: [] },
      };
      this.currentTf = "M1";
      this.lastSaved = null;
      await this.loadChart();
    },

    async prevTrade() {
      if (!this.hasPrev) return;
      const prev = this.sessions.find(s => s.session_index === this.current.session_index - 1);
      if (prev) await this.loadTrade(prev.id);
    },

    resetFeedback() {
      this.feedback = { comment: "", user_drawings: { rects: [], lines: [], arrows: [] } };
      this.lastSaved = null;
    },

    async switchTf(tf) {
      this.currentTf = tf;
      await this.loadChart();
    },

    initChart() {
      const el = document.getElementById("chart");
      if (!el) return;
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
      this.chart.timeScale().subscribeVisibleTimeRangeChange(() => this.redrawAll());
    },

    async loadChart() {
      if (!this.current || !this.chart) return;
      const url = `/api/candles/${this.currentTf}?around=${encodeURIComponent(this.current.entry_time)}`;
      const res = await fetch(url);
      if (!res.ok) {
        this.series.setData([]);
        return;
      }
      const candles = await res.json();
      this.series.setData(candles);
      this.chart.timeScale().fitContent();

      // Force le zoom vertical pour serrer autour des niveaux clés
      this._fitPriceRange(candles);

      this.resizeOverlays();
      setTimeout(() => this.redrawAll(), 50);  // laisse le chart se settle
    },

    _fitPriceRange(candles) {
      if (!candles?.length || !this.current) return;
      const ov = this.current.chart_overlays || {};
      const trade = this.current;

      // Collecte tous les prix d'intérêt pour le zoom
      const keyPrices = [];
      candles.forEach(c => { keyPrices.push(c.high, c.low); });
      [trade.entry_price, trade.stop_loss, trade.take_profit].forEach(p => {
        if (typeof p === "number") keyPrices.push(p);
      });
      if (ov.ob_zone) keyPrices.push(ov.ob_zone.high, ov.ob_zone.low);
      if (ov.liquidity?.price) keyPrices.push(ov.liquidity.price);

      const min = Math.min(...keyPrices);
      const max = Math.max(...keyPrices);
      const padding = (max - min) * 0.10;

      this.series.applyOptions({
        autoscaleInfoProvider: () => ({
          priceRange: { minValue: min - padding, maxValue: max + padding },
        }),
      });
    },

    resizeOverlays() {
      const chartEl = document.getElementById("chart");
      if (!chartEl) return;
      const w = chartEl.clientWidth, h = chartEl.clientHeight;
      for (const id of ["overlay", "drawing"]) {
        const c = document.getElementById(id);
        if (!c) continue;
        c.width = w; c.height = h;
        c.style.width = w + "px"; c.style.height = h + "px";
      }
      const drawing = document.getElementById("drawing");
      if (drawing) drawing.classList.toggle("active", this.currentTool !== "none");
    },

    priceToY(p)  { return this.series.priceToCoordinate(p); },
    timeToX(iso) { return this.chart.timeScale().timeToCoordinate(Math.floor(new Date(iso).getTime() / 1000)); },
    yToPrice(y)  { return this.series.coordinateToPrice(y); },
    xToTime(x)   { return this.chart.timeScale().coordinateToTime(x); },

    redrawAll() {
      this.drawAlgoOverlay();
      this.drawUserDrawings();
    },

    drawAlgoOverlay() {
      const c = document.getElementById("overlay");
      if (!c) return;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      if (!this.current || !this.current.chart_overlays) return;
      const ov = this.current.chart_overlays;

      if (ov.ob_zone) {
        const yHigh = this.priceToY(ov.ob_zone.high);
        const yLow = this.priceToY(ov.ob_zone.low);
        const xFrom = this.timeToX(ov.ob_zone.from_time);
        if (yHigh !== null && yLow !== null && xFrom !== null) {
          // Zone OB beaucoup plus visible
          ctx.fillStyle = "rgba(251, 191, 36, 0.25)";
          ctx.strokeStyle = "rgba(251, 191, 36, 1.0)";
          ctx.lineWidth = 2;
          ctx.fillRect(xFrom, Math.min(yHigh, yLow), c.width - xFrom, Math.abs(yHigh - yLow));
          ctx.strokeRect(xFrom, Math.min(yHigh, yLow), c.width - xFrom, Math.abs(yHigh - yLow));

          const yMid = this.priceToY(ov.ob_zone.mid);
          if (yMid !== null) {
            ctx.strokeStyle = "rgba(251, 191, 36, 0.7)";
            ctx.setLineDash([6, 4]);
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(xFrom, yMid); ctx.lineTo(c.width, yMid); ctx.stroke();
            ctx.setLineDash([]);
          }

          // Label "OB BOT" plus visible
          ctx.fillStyle = "rgba(251, 191, 36, 1.0)";
          ctx.font = "bold 12px monospace";
          ctx.fillText(`◆ OB BOT (${ov.ob_zone.type})`, xFrom + 6, Math.min(yHigh, yLow) - 6);
        }
      }

      if (ov.liquidity) {
        const y = this.priceToY(ov.liquidity.price);
        if (y !== null) {
          ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
          ctx.fillStyle = "#3b82f6"; ctx.font = "11px monospace";
          ctx.fillText("liquidite " + ov.liquidity.price.toFixed(2), 8, y - 4);
        }
      }

      this._drawHLine(ctx, c, ov.sl, "#ef4444", "SL");
      this._drawHLine(ctx, c, ov.tp, "#10b981", "TP");
      if (ov.entry) this._drawHLine(ctx, c, ov.entry.price, "#3b82f6", "ENTRY");
    },

    _drawHLine(ctx, c, price, color, label) {
      if (price == null) return;
      const y = this.priceToY(price);
      if (y === null) return;
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
      ctx.fillStyle = color; ctx.font = "11px monospace";
      ctx.fillText(`${label} ${price.toFixed(2)}`, c.width - 100, y - 4);
    },

    drawUserDrawings() {
      const c = document.getElementById("drawing");
      if (!c) return;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);

      const d = this.feedback.user_drawings || {};
      ctx.strokeStyle = "#a855f7"; ctx.lineWidth = 2;

      for (const r of (d.rects || [])) {
        const x1 = this.timeToX(r.t1), x2 = this.timeToX(r.t2);
        const y1 = this.priceToY(r.p1), y2 = this.priceToY(r.p2);
        if ([x1, x2, y1, y2].some(v => v === null)) continue;
        ctx.fillStyle = "rgba(168, 85, 247, 0.12)";
        ctx.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        ctx.fillStyle = "rgba(168, 85, 247, 0.9)";
        ctx.font = "11px monospace";
        ctx.fillText("USER", Math.min(x1, x2) + 4, Math.min(y1, y2) - 4);
      }
      for (const l of (d.lines || [])) {
        const x1 = this.timeToX(l.t1), x2 = this.timeToX(l.t2);
        const y1 = this.priceToY(l.p1), y2 = this.priceToY(l.p2);
        if ([x1, x2, y1, y2].some(v => v === null)) continue;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      for (const a of (d.arrows || [])) {
        const x1 = this.timeToX(a.t1), x2 = this.timeToX(a.t2);
        const y1 = this.priceToY(a.p1), y2 = this.priceToY(a.p2);
        if ([x1, x2, y1, y2].some(v => v === null)) continue;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const sz = 8;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - sz * Math.cos(angle - Math.PI / 6), y2 - sz * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - sz * Math.cos(angle + Math.PI / 6), y2 - sz * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
      }

      if (this.drawingState) {
        const { type, startX, startY, currentX, currentY } = this.drawingState;
        ctx.strokeStyle = "#c084fc"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
        if (type === "rect") {
          ctx.strokeRect(Math.min(startX, currentX), Math.min(startY, currentY), Math.abs(currentX - startX), Math.abs(currentY - startY));
        } else {
          ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(currentX, currentY); ctx.stroke();
        }
        ctx.setLineDash([]);
      }
    },

    setTool(t) { this.currentTool = t; this.resizeOverlays(); },
    clearDrawings() {
      this.feedback.user_drawings = { lines: [], rects: [], arrows: [] };
      this.redrawAll();
    },

    onMouseDown(e) {
      if (this.currentTool === "none") return;
      const r = e.currentTarget.getBoundingClientRect();
      this.drawingState = {
        type: this.currentTool,
        startX: e.clientX - r.left, startY: e.clientY - r.top,
        currentX: e.clientX - r.left, currentY: e.clientY - r.top,
      };
    },
    onMouseMove(e) {
      if (!this.drawingState) return;
      const r = e.currentTarget.getBoundingClientRect();
      this.drawingState.currentX = e.clientX - r.left;
      this.drawingState.currentY = e.clientY - r.top;
      this.redrawAll();
    },
    onMouseUp(e) {
      if (!this.drawingState) return;
      const { type, startX, startY, currentX, currentY } = this.drawingState;
      const t1 = this.xToTime(startX), t2 = this.xToTime(currentX);
      const p1 = this.yToPrice(startY), p2 = this.yToPrice(currentY);
      if (t1 == null || t2 == null || p1 == null || p2 == null) {
        this.drawingState = null; return;
      }
      const t1Iso = new Date(t1 * 1000).toISOString();
      const t2Iso = new Date(t2 * 1000).toISOString();

      if (type === "rect")  this.feedback.user_drawings.rects.push({ t1: t1Iso, t2: t2Iso, p1, p2 });
      else if (type === "line")  this.feedback.user_drawings.lines.push({ t1: t1Iso, t2: t2Iso, p1, p2 });
      else if (type === "arrow") this.feedback.user_drawings.arrows.push({ t1: t1Iso, t2: t2Iso, p1, p2 });

      this.drawingState = null;
      this.redrawAll();
    },

    async saveFeedback() {
      if (!this.current) return;
      this.saving = true;
      try {
        await fetch(`/api/training/${this.current.id}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.feedback),
        });
        this.lastSaved = new Date().toLocaleTimeString("fr-FR");
        await this.refreshSessions();
      } finally {
        this.saving = false;
      }
    },

    async saveAndNext() {
      await this.saveFeedback();
      await this.nextTrade();
    },

    fmtTime(iso) {
      if (!iso) return "";
      return new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
    },
  };
}
