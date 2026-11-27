// A plot panel: any metrics from any runs on one uPlot chart, with
// server-side downsampling, zoom re-query, grouping, smoothing, live updates.
import { html, useState, useRef, useEffect, useMemo } from "./vendor/htm-preact.module.js";
import uPlot from "./vendor/uPlot.esm.js";
import { api } from "./api.js";
import { MultiSelect, Select, Toggle } from "./widgets.js";
import { ema, fmtDuration, fmtNum, fmtStep, fmtValue, slotColor, slotDash, throttle, withAlpha, DASHES, isDark } from "./util.js";

const AGG_LABELS = { mean_std: "mean ± std", min_max: "mean, min–max", median_iqr: "median, IQR" };
const MAX_RUNS_UNGROUPED = 100;

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// --------------------------------------------------------------- data shaping
// Build per-line arrays [{label, color, dash, xs, ys, lo?, hi?, runId, key}]
function buildLines(data, panel, runsById, slots, colorSlots) {
  const lines = [];
  if (!data) return lines;
  const keyIdx = new Map(panel.keys.map((k, i) => [k, i]));
  const multiKey = panel.keys.length > 1;
  if (data.groups) {
    data.groups.forEach((g, gi) => {
      const gl = Object.entries(g.group).map(([k, v]) => `${k}=${fmtValue(v)}`).join(", ");
      const gslot = colorSlots.groups.get(JSON.stringify(g.group)) ?? gi;
      lines.push({
        label: (multiKey ? g.key + " · " : "") + gl + ` (n=${g.runs.length})`,
        color: slotColor(gslot), dash: DASHES[(keyIdx.get(g.key) || 0) % DASHES.length],
        xs: g.x || g.step, steps: g.step, ys: g.center, lo: g.lo, hi: g.hi, n: g.n, key: g.key, group: true,
      });
    });
    return lines;
  }
  for (const s of data.series) {
    const run = runsById.get(s.run_id);
    if (!run) continue;
    let slot = slots.get(s.run_id) ?? 0;
    if (panel.colorBy) slot = colorSlots.values.get(JSON.stringify(run.config[panel.colorBy] ?? null)) ?? 0;
    const down = s.width > 1;
    const dash = multiKey ? DASHES[(keyIdx.get(s.key) || 0) % DASHES.length] : slotDash(slot);
    lines.push({
      label: (multiKey ? s.key + " · " : "") + run.name, runId: s.run_id, key: s.key,
      color: slotColor(slot), dash,
      xs: s.x || s.step, steps: s.step, ys: down ? s.mean : s.last,
      lo: down && panel.band ? s.min : null, hi: down && panel.band ? s.max : null,
    });
  }
  return lines;
}

// Align lines onto one sorted x array (uPlot's data model).
function align(lines, logx, logy) {
  const prepared = lines.map((l) => {
    const n = l.xs.length;
    const idx = [];
    for (let i = 0; i < n; i++) {
      const x = l.xs[i];
      if (x == null || !isFinite(x) || (logx && x <= 0)) continue;
      idx.push(i);
    }
    let sorted = true;
    for (let j = 1; j < idx.length; j++) if (l.xs[idx[j]] < l.xs[idx[j - 1]]) { sorted = false; break; }
    if (!sorted) idx.sort((a, b) => l.xs[a] - l.xs[b]);
    return idx;
  });
  // union of x values
  let total = 0;
  for (const p of prepared) total += p.length;
  const all = new Float64Array(total);
  let o = 0;
  prepared.forEach((p, li) => { const xs = lines[li].xs; for (const i of p) all[o++] = xs[i]; });
  all.sort();
  const ux = [];
  for (let i = 0; i < all.length; i++) if (i === 0 || all[i] !== all[i - 1]) ux.push(all[i]);
  const pos = new Map();
  ux.forEach((x, i) => pos.set(x, i));
  const fix = (v) => (v == null || !isFinite(v) || (logy && v <= 0) ? null : v);
  const cols = [];
  const stepCol = [];
  lines.forEach((l, li) => {
    const y = new Array(ux.length).fill(null);
    const lo = l.lo ? new Array(ux.length).fill(null) : null;
    const hi = l.hi ? new Array(ux.length).fill(null) : null;
    const st = new Array(ux.length).fill(null);
    for (const i of prepared[li]) {
      const p = pos.get(l.xs[i]);
      y[p] = fix(l.ys[i]);
      if (lo) lo[p] = fix(l.lo[i]);
      if (hi) hi[p] = fix(l.hi[i]);
      st[p] = l.steps ? l.steps[i] : null;
    }
    cols.push({ y, lo, hi });
    stepCol.push(st);
  });
  return { x: ux, cols, stepCol };
}

// --------------------------------------------------------------- merge for live updates
const ARR_FIELDS = ["step", "min", "max", "mean", "last", "count", "wall_time", "x"];
function mergeSeries(old, inc, start) {
  let cut = old.step.length;
  while (cut > 0 && old.step[cut - 1] >= start) cut--;
  const out = { ...old };
  for (const f of ARR_FIELDS) {
    if (!old[f] && !inc[f]) continue;
    const a = old[f] ? old[f].slice(0, cut) : new Array(cut).fill(null);
    const b = inc[f] || new Array(inc.step.length).fill(null);
    out[f] = a.concat(b);
  }
  return out;
}

// --------------------------------------------------------------- component
export function Panel({ panel, runs, runsById, slots, keys, onChange, onRemove, live, expanded, onExpand }) {
  const box = useRef(null);
  const chartEl = useRef(null);
  const plot = useRef(null);
  const tip = useRef(null);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [width, setWidth] = useState(600);
  const [hidden, setHidden] = useState(new Set());
  const req = useRef(0);
  const meta = useRef({ span: null, mp: 0 });

  const grouped = panel.group.length > 0;
  const plotRuns = useMemo(() => grouped ? runs : runs.slice(0, MAX_RUNS_UNGROUPED), [runs, grouped]);
  const runIds = useMemo(() => plotRuns.map((r) => r.id), [plotRuns]);
  const runKey = runIds.join(",");
  const stepX = panel.x === "step";

  // ---- size ----
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      const w = chartEl.current ? chartEl.current.clientWidth : 600;
      setWidth(w);
      if (plot.current) plot.current.setSize({ width: w, height: plotHeight() });
    });
    ro.observe(chartEl.current);
    return () => ro.disconnect();
  }, []);
  const plotHeight = () => (expanded ? 420 : 260);
  useEffect(() => { if (plot.current) plot.current.setSize({ width, height: plotHeight() }); }, [expanded]);

  // ---- full fetch ----
  const mp = Math.max(200, Math.min(4000, Math.round(width * 2)));
  async function fetchFull() {
    if (!panel.keys.length || !runIds.length) { setData(null); return; }
    const my = ++req.current;
    setLoading(true);
    try {
      const body = { runs: runIds, keys: panel.keys, max_points: mp, x: panel.x };
      if (grouped) { body.group_by = panel.group; body.agg = panel.agg; }
      if (panel.range && stepX) { body.start = Math.floor(panel.range[0]); body.end = Math.ceil(panel.range[1]) + 1; }
      const r = await api.series(body);
      if (my !== req.current) return;
      meta.current = { span: r.end != null && r.start != null ? r.end - r.start : null, mp, start: r.start, end: r.end };
      setData(r);
      setErr(null);
    } catch (e) {
      if (my === req.current) setErr(e.message);
    } finally {
      if (my === req.current) setLoading(false);
    }
  }
  const widthBucket = Math.round(mp / 400);  // refetch only on substantial resizes
  useEffect(() => { fetchFull(); }, [runKey, panel.keys.join("\n"), panel.x, panel.group.join("\n"), panel.agg,
                                    JSON.stringify(panel.range), widthBucket]);

  // ---- live incremental update ----
  const incremental = useRef(null);
  incremental.current = async (changed) => {
    if (!data || panel.range) return;  // zoomed views stay put
    const ids = runIds.filter((id) => changed.has(id));
    if (!ids.length) return;
    if (grouped || !stepX && panel.x !== "wall" && panel.x !== "relative") { fetchFull(); return; }
    const { mp: m, start: s0 } = meta.current;
    const mine = data.series.filter((s) => changed.has(s.run_id));
    if (mine.length < ids.length * panel.keys.length) { fetchFull(); return; }  // new run/key
    const w = Math.max(...data.series.map((s) => s.width || 1));
    let start = Infinity;
    for (const s of mine) {
      const ls = s.step.length ? s.step[s.step.length - 1] : (s0 ?? 0);
      start = Math.min(start, ls - (((ls % w) + w) % w));
    }
    if (!isFinite(start)) { fetchFull(); return; }
    const end = start + w * (m - 1);
    const my = ++req.current;
    try {
      const r = await api.series({ runs: ids, keys: panel.keys, max_points: m, x: panel.x, start, end });
      if (my !== req.current) return;
      const overflow = r.series.some((s) => s.step.length && s.step[s.step.length - 1] >= end - w);
      const grown = meta.current.span && (start + w - (meta.current.start ?? 0)) > 2 * meta.current.span;
      if (overflow || grown) { fetchFull(); return; }
      const inc = new Map(r.series.map((s) => [s.run_id + "\n" + s.key, s]));
      const merged = data.series.map((s) => {
        const n = inc.get(s.run_id + "\n" + s.key);
        return n ? mergeSeries(s, n, start) : s;
      });
      setData({ ...data, series: merged });
    } catch (e) { /* next event retries */ }
  };
  const onLive = useRef(throttle((changed) => incremental.current(changed), 1500)).current;
  const pendingChanged = useRef(new Set());
  useEffect(() => {
    if (!live || !live.runs) return;
    if (live.full && runIds.some((id) => live.full.has(id))) { fetchFull(); return; }  // rewind: history changed
    let any = false;
    for (const id of live.runs) if (runIds.includes(id)) { pendingChanged.current.add(id); any = true; }
    if (any) {
      const ch = pendingChanged.current;
      pendingChanged.current = new Set();
      onLive(ch);
    }
  }, [live && live.seq]);

  // ---- color slots for color-by / groups ----
  const colorSlots = useMemo(() => {
    const values = new Map();
    if (panel.colorBy) {
      const vals = [...new Set(plotRuns.map((r) => JSON.stringify(r.config[panel.colorBy] ?? null)))].sort();
      vals.forEach((v, i) => values.set(v, i));
    }
    const groups = new Map();
    if (data && data.groups) {
      const gs = [...new Set(data.groups.map((g) => JSON.stringify(g.group)))].sort();
      gs.forEach((g, i) => groups.set(g, i));
    }
    return { values, groups };
  }, [panel.colorBy, plotRuns, data]);

  // ---- build / update chart ----
  const lines = useMemo(() => buildLines(data, panel, runsById, slots, colorSlots),
                        [data, panel.keys, panel.band, panel.colorBy, runsById, slots, colorSlots]);
  useEffect(() => {
    if (!chartEl.current) return;
    if (!lines.length) {
      if (plot.current) { plot.current.destroy(); plot.current = null; }
      return;
    }
    const smoothed = lines.map((l) => ({ ...l, ys: ema(l.ys, panel.smooth), lo: l.lo, hi: l.hi }));
    const al = align(smoothed, panel.logx, panel.logy);
    const udata = [al.x];
    const series = [{}];
    const bands = [];
    const dark = isDark();
    smoothed.forEach((l, i) => {
      const c = al.cols[i];
      const show = !hidden.has(l.label);
      if (c.hi && c.lo) {
        udata.push(c.hi, c.lo);
        const hiIdx = series.length;
        series.push({ label: l.label + " max", stroke: "transparent", points: { show: false }, show, spanGaps: true, _aux: true });
        series.push({ label: l.label + " min", stroke: "transparent", points: { show: false }, show, spanGaps: true, _aux: true });
        bands.push({ series: [hiIdx, hiIdx + 1], fill: withAlpha(l.color, dark ? 0.22 : 0.16) });
      }
      udata.push(c.y);
      series.push({
        label: l.label, stroke: l.color, width: l.group ? 2 : 1.5, dash: l.dash, spanGaps: true, show,
        points: { show: false }, _line: i,
      });
    });

    const axisStroke = css("--text-secondary");
    const gridStroke = css("--grid");
    const xIsTime = panel.x === "wall";
    const xFmt = panel.x === "step" ? (u, vals) => vals.map(fmtStep)
      : panel.x === "relative" ? (u, vals) => vals.map(fmtDuration)
      : xIsTime ? undefined : (u, vals) => vals.map(fmtNum);
    const opts = {
      width: chartEl.current.clientWidth, height: plotHeight(),
      series, bands,
      legend: { show: false },
      scales: {
        x: { time: xIsTime, distr: panel.logx ? 3 : 1 },
        y: { distr: panel.logy ? 3 : 1 },
      },
      axes: [
        { stroke: axisStroke, grid: { stroke: gridStroke, width: 1 }, ticks: { stroke: gridStroke }, values: xFmt, size: 36 },
        { stroke: axisStroke, grid: { stroke: gridStroke, width: 1 }, ticks: { stroke: gridStroke },
          values: (u, vals) => vals.map(fmtNum), size: 56 },
      ],
      cursor: {
        drag: { x: true, y: false, setScale: false },
        points: { size: 7, fill: (u, i) => series[i].stroke },
        focus: { prox: 24 },
        bind: {
          dblclick: () => () => { onChange({ range: null }); return null; },
        },
      },
      focus: { alpha: 0.35 },
      hooks: {
        setSelect: [(u) => {
          if (u.select.width < 4) return;
          const lo = u.posToVal(u.select.left, "x");
          const hi = u.posToVal(u.select.left + u.select.width, "x");
          u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
          onChange({ range: [lo, hi] });
        }],
        setCursor: [(u) => updateTip(u, al, smoothed)],
      },
    };
    const structure = JSON.stringify([series.map((s) => [s.label, s.stroke, s.show]), panel.logx, panel.logy, panel.x, expanded, dark]);
    if (plot.current && plot.current._structure === structure) {
      plot.current.setData(udata);
    } else {
      if (plot.current) plot.current.destroy();
      plot.current = new uPlot(opts, udata, chartEl.current);
      plot.current._structure = structure;
      plot.current.over.addEventListener("mouseleave", () => { if (tip.current) tip.current.style.display = "none"; });
    }
    if (panel.range && !stepX) plot.current.setScale("x", { min: panel.range[0], max: panel.range[1] });
  }, [lines, panel.smooth, panel.logx, panel.logy, hidden, expanded]);

  useEffect(() => () => { if (plot.current) plot.current.destroy(); }, []);

  function updateTip(u, al, lines) {
    const el = tip.current;
    if (!el) return;
    const idx = u.cursor.idx;
    if (idx == null || u.cursor.left < 0) { el.style.display = "none"; return; }
    const rows = [];
    lines.forEach((l, i) => {
      if (hidden.has(l.label)) return;
      const ys = al.cols[i].y;
      let j = idx, found = null;
      for (let d = 0; d < 64; d++) {
        if (ys[idx + d] != null) { j = idx + d; found = ys[j]; break; }
        if (ys[idx - d] != null) { j = idx - d; found = ys[j]; break; }
      }
      if (found == null) return;
      rows.push({ l, v: found, step: al.stepCol[i][j], n: l.n ? null : null });
    });
    rows.sort((a, b) => b.v - a.v);
    const xv = al.x[idx];
    const xLabel = panel.x === "step" ? `step ${fmtNum(Math.round(xv))}`
      : panel.x === "wall" ? new Date(xv * 1000).toLocaleString()
      : panel.x === "relative" ? fmtDuration(xv) : `${panel.x} ${fmtNum(xv)}`;
    const maxRows = 12;
    el.innerHTML = `<div class="tip-x">${xLabel}</div>` + rows.slice(0, maxRows).map((r) =>
      `<div class="tip-row"><span class="tip-sw" style="background:${r.l.color}"></span><span class="tip-name">${escapeHtml(r.l.label)}</span><span class="tip-v">${fmtNum(r.v)}</span></div>`).join("")
      + (rows.length > maxRows ? `<div class="tip-more">+${rows.length - maxRows} more</div>` : "");
    el.style.display = "block";
    const bw = box.current.clientWidth;
    const left = u.cursor.left + u.over.offsetLeft;
    const tw = el.offsetWidth;
    el.style.left = (left + 16 + tw > bw ? left - tw - 16 : left + 16) + "px";
    el.style.top = Math.max(0, u.cursor.top + u.over.offsetTop - 20) + "px";
  }

  // ---- controls ----
  const metricOpts = keys.metrics.map((m) => ({ value: m }));
  const configOpts = keys.config.map((k) => ({ value: k.key, kind: `${k.distinct}` }));
  const xOpts = [{ value: "step", label: "Step" }, { value: "wall", label: "Wall time" },
                 { value: "relative", label: "Relative time" },
                 ...keys.metrics.map((m) => ({ value: m, label: m }))];
  const set = (patch) => onChange(patch);
  const legend = lines;

  return html`
    <div class=${"panel" + (expanded ? " expanded" : "")} ref=${box}>
      <div class="panel-head">
        <${MultiSelect} label="Metrics" options=${metricOpts} value=${panel.keys} onChange=${(v) => set({ keys: v })} placeholder="choose…" />
        <${Select} label="x" value=${panel.x} options=${xOpts} onChange=${(v) => set({ x: v, range: null })} title="X axis" />
        <${MultiSelect} label="Group" options=${configOpts} value=${panel.group} onChange=${(v) => set({ group: v })} title="Group runs by config keys and aggregate" />
        ${grouped && html`<${Select} label="Agg" value=${panel.agg}
            options=${Object.entries(AGG_LABELS).map(([value, label]) => ({ value, label }))} onChange=${(v) => set({ agg: v })} />`}
        ${!grouped && html`<${MultiSelect} label="Color" single=${true} options=${configOpts}
            value=${panel.colorBy ? [panel.colorBy] : []} onChange=${(v) => set({ colorBy: v[0] || "" })} placeholder="run" title="Color by config key" />`}
        <label class="smooth" title="EMA smoothing">
          <span class="ms-label">Smooth</span>
          <input type="range" min="0" max="0.99" step="0.01" value=${panel.smooth}
            onInput=${(e) => set({ smooth: parseFloat(e.target.value) })} />
          <span class="smooth-v">${panel.smooth.toFixed(2)}</span>
        </label>
        <${Toggle} label="log x" value=${panel.logx} onChange=${(v) => set({ logx: v })} />
        <${Toggle} label="log y" value=${panel.logy} onChange=${(v) => set({ logy: v })} />
        ${!grouped && html`<${Toggle} label="band" value=${panel.band} onChange=${(v) => set({ band: v })} title="Min/max band of each downsampled bucket" />`}
        <span class="spacer"></span>
        ${loading && html`<span class="spinner" title="Loading"></span>`}
        ${panel.range && html`<button class="btn small ghost" onClick=${() => set({ range: null })} title="Reset zoom (or double-click the plot)">Reset zoom</button>`}
        <button class="icon-btn" title=${expanded ? "Collapse" : "Expand"} onClick=${onExpand}>${expanded ? "⤡" : "⤢"}</button>
        <button class="icon-btn" title="Remove panel" onClick=${onRemove}>×</button>
      </div>
      <div class="chart" ref=${chartEl}>
        ${!panel.keys.length && html`<div class="chart-empty">Choose metrics to plot</div>`}
        ${panel.keys.length > 0 && !runIds.length && html`<div class="chart-empty">No runs to plot</div>`}
        ${err && html`<div class="chart-empty error">${err}</div>`}
      </div>
      <div class="tip" ref=${tip}></div>
      ${legend.length > 0 && html`
        <div class="legend">
          ${legend.slice(0, 40).map((l) => html`
            <button class=${"leg" + (hidden.has(l.label) ? " off" : "")} title="Click to hide/show"
              onClick=${() => { const h = new Set(hidden); h.has(l.label) ? h.delete(l.label) : h.add(l.label); setHidden(h); }}>
              <svg viewBox="0 0 18 8" class="swatch"><line x1="1" y1="4" x2="17" y2="4" stroke=${l.color} stroke-width="3"
                stroke-linecap="round" stroke-dasharray=${l.dash && l.dash.length ? l.dash.map((d) => d / 2).join(" ") : ""} /></svg>
              <span>${l.label}</span>
            </button>`)}
          ${legend.length > 40 && html`<span class="muted">+${legend.length - 40} more</span>`}
          ${!grouped && runs.length > MAX_RUNS_UNGROUPED && html`<span class="muted">showing first ${MAX_RUNS_UNGROUPED} runs, group to see all</span>`}
        </div>`}
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
