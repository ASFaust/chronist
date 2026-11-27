// Histogram panel: heatmap (step x value), quantile bands, stats lines, single step.
import { html, useState, useRef, useEffect, useMemo } from "./vendor/htm-preact.module.js";
import uPlot from "./vendor/uPlot.esm.js";
import { api } from "./api.js";
import { Select, Toggle } from "./widgets.js";
import { fmtNum, fmtStep, slotColor, withAlpha, isDark, throttle } from "./util.js";

const VIEWS = [
  { value: "heatmap", label: "Heatmap" }, { value: "quantiles", label: "Quantiles" },
  { value: "stats", label: "Stats" }, { value: "step", label: "Single step" },
];
const MAX_HEATMAPS = 6;

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// Sequential single-hue ramp (validated blue steps), light -> dark; reversed in dark mode
// so that more mass is always more prominent against the surface.
const RAMP_LIGHT = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"];
const RAMP_DARK = ["#0e3d7e", "#1757ac", "#2370cd", "#338ced", "#6aabf1", "#9dc8f7", "#cce3fd"];
function rampRGB() {
  return (isDark() ? RAMP_DARK : RAMP_LIGHT).map((h) => {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  });
}
function colorAt(ramp, t) {
  const x = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.min(Math.floor(x), ramp.length - 2), f = x - i;
  return ramp[i].map((c, k) => Math.round(c + (ramp[i + 1][k] - c) * f));
}

function axes() {
  const s = css("--text-secondary"), g = css("--grid");
  return [
    { stroke: s, grid: { stroke: g, width: 1 }, ticks: { stroke: g }, values: (u, v) => v.map(fmtStep), size: 34 },
    { stroke: s, grid: { stroke: g, width: 1 }, ticks: { stroke: g }, values: (u, v) => v.map(fmtNum), size: 56 },
  ];
}

// ---- one heatmap (one run) drawn into a uPlot plot area ----
function Heatmap({ entry, yRange, rows, vmax, logColor, title, color, onZoom, height }) {
  const el = useRef(null);
  const plot = useRef(null);
  const tip = useRef(null);
  useEffect(() => {
    if (!el.current || !entry || !entry.steps.length) return;
    const steps = entry.steps, n = steps.length;
    const m = entry.matrix;
    // offscreen image: one pixel per (step, row)
    const off = document.createElement("canvas");
    off.width = n; off.height = rows;
    const ctx2 = off.getContext("2d");
    const img = ctx2.createImageData(n, rows);
    const ramp = rampRGB();
    const norm = logColor ? (v) => Math.log1p(v) / Math.log1p(vmax || 1) : (v) => v / (vmax || 1);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < rows; j++) {
        const v = m[i * rows + j];
        const p = ((rows - 1 - j) * n + i) * 4;
        if (!v) { img.data[p + 3] = 0; continue; }
        const [r, g, b] = colorAt(ramp, 0.08 + 0.92 * norm(v));
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
      }
    }
    ctx2.putImageData(img, 0, 0);
    const last = n > 1 ? steps[n - 1] + (steps[n - 1] - steps[n - 2]) : steps[0] + 1;
    const opts = {
      width: el.current.clientWidth, height, legend: { show: false },
      scales: { x: { time: false, range: () => [steps[0], last] }, y: { range: () => yRange } },
      axes: axes(),
      series: [{}, { stroke: "transparent", points: { show: false } }],
      cursor: { drag: { x: true, y: false, setScale: false }, points: { show: false },
                bind: { dblclick: () => () => { onZoom(null); return null; } } },
      hooks: {
        draw: [(u) => {
          const c = u.ctx;
          c.save();
          c.imageSmoothingEnabled = false;
          const top = u.bbox.top, h = u.bbox.height;
          for (let i = 0; i < n; i++) {
            const x0 = u.valToPos(steps[i], "x", true);
            const x1 = u.valToPos(i + 1 < n ? steps[i + 1] : last, "x", true);
            c.drawImage(off, i, 0, 1, rows, x0, top, Math.max(1, x1 - x0), h);
          }
          c.restore();
        }],
        setSelect: [(u) => {
          if (u.select.width < 4) return;
          const lo = u.posToVal(u.select.left, "x"), hi = u.posToVal(u.select.left + u.select.width, "x");
          u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
          onZoom([lo, hi]);
        }],
        setCursor: [(u) => {
          const t = tip.current;
          if (!t) return;
          const { left, top } = u.cursor;
          if (left == null || left < 0) { t.style.display = "none"; return; }
          const xv = u.posToVal(left, "x"), yv = u.posToVal(top, "y");
          let i = 0;
          while (i + 1 < n && steps[i + 1] <= xv) i++;
          const j = Math.floor((yv - yRange[0]) / (yRange[1] - yRange[0]) * rows);
          if (j < 0 || j >= rows) { t.style.display = "none"; return; }
          const w = (yRange[1] - yRange[0]) / rows;
          const lo = yRange[0] + j * w;
          const nan = entry.nan_count[i], inf = entry.posinf_count[i] + entry.neginf_count[i];
          const outside = entry.outside ? Math.round(entry.outside[i]) : 0;
          t.innerHTML = `<div class="tip-x">step ${fmtNum(steps[i])}</div>`
            + `<div class="tip-row"><span class="tip-name">[${fmtNum(lo)}, ${fmtNum(lo + w)})</span><span class="tip-v">${fmtNum(Math.round(m[i * rows + j] * 10) / 10)}</span></div>`
            + `<div class="tip-row"><span class="tip-name">mean ± std</span><span class="tip-v">${fmtNum(entry.mean[i])} ± ${fmtNum(entry.std[i])}</span></div>`
            + (outside > 0 ? `<div class="tip-row"><span class="tip-name">outside view</span><span class="tip-v">${fmtNum(outside)}</span></div>` : "")
            + (nan || inf ? `<div class="tip-row warnline"><span class="tip-name">NaN / Inf</span><span class="tip-v">${nan} / ${inf}</span></div>` : "")
            + ((entry.underflow[i] || entry.overflow[i]) ? `<div class="tip-row"><span class="tip-name">under / over</span><span class="tip-v">${entry.underflow[i]} / ${entry.overflow[i]}</span></div>` : "");
          t.style.display = "block";
          const bw = el.current.parentElement.clientWidth;
          const x = left + u.over.offsetLeft;
          t.style.left = (x + 16 + t.offsetWidth > bw ? x - t.offsetWidth - 16 : x + 16) + "px";
          t.style.top = (top + u.over.offsetTop + el.current.offsetTop - 10) + "px";
        }],
      },
    };
    if (plot.current) plot.current.destroy();
    plot.current = new uPlot(opts, [steps, steps.map(() => null)], el.current);
    plot.current.over.addEventListener("mouseleave", () => { if (tip.current) tip.current.style.display = "none"; });
    const ro = new ResizeObserver(() => plot.current && plot.current.setSize({ width: el.current.clientWidth, height }));
    ro.observe(el.current);
    return () => { ro.disconnect(); };
  }, [entry, yRange && yRange.join(), rows, vmax, logColor, height]);
  useEffect(() => () => { if (plot.current) plot.current.destroy(); }, []);
  // NaN/Inf marker strip: steps with non-finite values
  const bad = entry ? entry.steps.map((s, i) => entry.nan_count[i] + entry.posinf_count[i] + entry.neginf_count[i]) : [];
  const anyBad = bad.some((b) => b > 0);
  return html`
    <div class="hm">
      <div class="hm-title"><span class="tip-sw" style=${{ background: color }}></span>${title}
        ${anyBad && html`<span class="flag" title="Some steps contain NaN or Inf values (excluded from bins)">⚠ NaN/Inf at ${bad.filter((b) => b > 0).length} steps</span>`}
      </div>
      <div ref=${el}></div>
      <div class="tip" ref=${tip}></div>
    </div>`;
}

// ---- line-style views (quantiles / stats / step) on one uPlot ----
function unionX(lists) {
  const s = new Set();
  for (const l of lists) for (const x of l) s.add(x);
  return [...s].sort((a, b) => a - b);
}
function alignTo(xs, srcX, ys) {
  const m = new Map(srcX.map((x, i) => [x, ys[i]]));
  return xs.map((x) => (m.has(x) ? m.get(x) : null));
}

function LinePlot({ build, height, onZoom, xIsValue, deps }) {
  const el = useRef(null);
  const plot = useRef(null);
  useEffect(() => {
    if (!el.current) return;
    const spec = build();
    if (!spec) { if (plot.current) { plot.current.destroy(); plot.current = null; } return; }
    const ax = axes();
    if (xIsValue) ax[0].values = (u, v) => v.map(fmtNum);
    const opts = {
      width: el.current.clientWidth, height, legend: { show: false }, axes: ax,
      scales: { x: { time: false } }, series: spec.series, bands: spec.bands || [],
      cursor: { drag: { x: !xIsValue, y: false, setScale: false }, points: { size: 6 },
                bind: { dblclick: () => () => { onZoom && onZoom(null); return null; } } },
      hooks: {
        setSelect: [(u) => {
          if (xIsValue || u.select.width < 4) return;
          const lo = u.posToVal(u.select.left, "x"), hi = u.posToVal(u.select.left + u.select.width, "x");
          u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
          onZoom && onZoom([lo, hi]);
        }],
      },
    };
    if (plot.current) plot.current.destroy();
    plot.current = new uPlot(opts, spec.data, el.current);
    const ro = new ResizeObserver(() => plot.current && plot.current.setSize({ width: el.current.clientWidth, height }));
    ro.observe(el.current);
    return () => ro.disconnect();
  }, deps);
  useEffect(() => () => { if (plot.current) plot.current.destroy(); }, []);
  return html`<div ref=${el}></div>`;
}

export function HistPanel({ panel, runs, slots, keys, onChange, onRemove, live, expanded, onExpand }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stepData, setStepData] = useState(null);
  const req = useRef(0);
  const plotRuns = useMemo(() => runs.slice(0, panel.view === "heatmap" ? MAX_HEATMAPS : 30), [runs, panel.view]);
  const runIds = plotRuns.map((r) => r.id);
  const runKey = runIds.join(",");
  const byId = useMemo(() => new Map(plotRuns.map((r) => [r.id, r])), [plotRuns]);
  const height = expanded ? 360 : 220;
  const rows = Math.max(32, Math.min(256, Math.round(height / 1.5)));

  async function fetchData() {
    if (!panel.key || !runIds.length) { setData(null); return; }
    const my = ++req.current;
    setLoading(true);
    try {
      const body = { runs: runIds, key: panel.key, max_steps: 400, rows, robust: !panel.fullRange,
                     mode: panel.view === "heatmap" ? "heatmap" : "quantiles" };
      if (panel.range) { body.start = Math.floor(panel.range[0]); body.end = Math.ceil(panel.range[1]) + 1; }
      const r = await api.histRender(body);
      if (my !== req.current) return;
      setData(r); setErr(null);
    } catch (e) { if (my === req.current) setErr(e.message); }
    finally { if (my === req.current) setLoading(false); }
  }
  useEffect(() => { fetchData(); }, [runKey, panel.key, panel.view, JSON.stringify(panel.range), rows, panel.fullRange]);

  // single step view
  const allSteps = useMemo(() => data && data.runs ? unionX(data.runs.map((r) => r.steps)) : [], [data]);
  const stepVal = panel.step != null ? panel.step : allSteps.length ? allSteps[allSteps.length - 1] : null;
  useEffect(() => {
    if (panel.view !== "step" || stepVal == null || !panel.key) return;
    let alive = true;
    api.histRender({ runs: runIds, key: panel.key, mode: "step", step: stepVal, rows: 64 })
      .then((r) => alive && setStepData(r)).catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [panel.view, stepVal, runKey, panel.key, data]);

  // live: refetch when one of our runs logged histograms
  const refetch = useRef(throttle(() => fetchData(), 2500)).current;
  useEffect(() => {
    if (live && live.hist && !panel.range && runIds.some((id) => live.hist.has(id))) refetch();
  }, [live && live.seq]);

  const color = (rid) => slotColor(slots.get(rid) ?? 0);
  const set = (p) => onChange(p);
  const onZoom = (r) => set({ range: r });

  let body = null;
  if (!panel.key) body = html`<div class="chart-empty">Choose a histogram key</div>`;
  else if (err) body = html`<div class="chart-empty error">${err}</div>`;
  else if (data && data.runs && data.runs.length === 0) body = html`<div class="chart-empty">No histogram data for these runs</div>`;
  else if (data && panel.view === "heatmap" && data.mode === "heatmap") {
    body = html`<div class="hm-grid">${data.runs.map((e) => html`
      <${Heatmap} key=${e.run_id} entry=${e} yRange=${data.y_range} rows=${data.rows} vmax=${data.vmax}
        logColor=${panel.logColor} title=${(byId.get(e.run_id) || {}).name || e.run_id} color=${color(e.run_id)}
        onZoom=${onZoom} height=${data.runs.length > 1 && !expanded ? 160 : height} />`)}
      ${runs.length > MAX_HEATMAPS && html`<div class="muted small">showing ${MAX_HEATMAPS} of ${runs.length} runs</div>`}
    </div>`;
  } else if (data && data.mode === "quantiles" && (panel.view === "quantiles" || panel.view === "stats")) {
    const dark = isDark();
    const build = () => {
      const xs = unionX(data.runs.map((r) => r.steps));
      const series = [{}], cols = [xs], bands = [];
      for (const r of data.runs) {
        const c = color(r.run_id);
        const name = (byId.get(r.run_id) || {}).name || r.run_id;
        const add = (ys, s) => { cols.push(alignTo(xs, r.steps, ys)); series.push({ spanGaps: true, points: { show: false }, ...s }); return series.length - 1; };
        if (panel.view === "quantiles") {
          const a = add(r.q95, { label: name + " p95", stroke: "transparent" });
          const b = add(r.q05, { label: name + " p5", stroke: "transparent" });
          const c1 = add(r.q75, { label: name + " p75", stroke: "transparent" });
          const d = add(r.q25, { label: name + " p25", stroke: "transparent" });
          bands.push({ series: [a, b], fill: withAlpha(c, dark ? 0.14 : 0.1) });
          bands.push({ series: [c1, d], fill: withAlpha(c, dark ? 0.3 : 0.22) });
          add(r.q50, { label: name + " median", stroke: c, width: 1.75 });
        } else {
          const mean = r.mean, std = r.std;
          const hi = add(mean.map((m, i) => (m == null || std[i] == null ? null : m + std[i])), { label: name + " +std", stroke: "transparent" });
          const lo = add(mean.map((m, i) => (m == null || std[i] == null ? null : m - std[i])), { label: name + " -std", stroke: "transparent" });
          bands.push({ series: [hi, lo], fill: withAlpha(c, dark ? 0.25 : 0.18) });
          add(mean, { label: name + " mean", stroke: c, width: 1.75 });
          add(r.min, { label: name + " min", stroke: c, width: 1, dash: [3, 3] });
          add(r.max, { label: name + " max", stroke: c, width: 1, dash: [3, 3] });
        }
      }
      return { data: cols, series, bands };
    };
    body = html`<${LinePlot} build=${build} height=${height} onZoom=${onZoom} deps=${[data, panel.view, height, dark]} />
      <div class="legend">${data.runs.map((r) => html`<span class="leg"><span class="tip-sw" style=${{ background: color(r.run_id) }}></span>${(byId.get(r.run_id) || {}).name}</span>`)}
        <span class="muted small">${panel.view === "quantiles" ? "median · 25–75% · 5–95%" : "mean ± std · min/max dashed"}</span></div>`;
  } else if (panel.view === "step" && stepData && stepData.edges) {
    const e = stepData.edges;
    const centers = e.slice(0, -1).map((x, i) => (x + e[i + 1]) / 2);
    const build = () => {
      const series = [{}], cols = [centers];
      for (const r of stepData.runs) {
        cols.push(r.density);
        series.push({ label: (byId.get(r.run_id) || {}).name, stroke: color(r.run_id), width: 1.5,
                      fill: withAlpha(color(r.run_id), 0.12), paths: uPlot.paths.stepped({ align: 0 }), points: { show: false } });
      }
      return { data: cols, series };
    };
    body = html`
      <${LinePlot} build=${build} height=${height} xIsValue=${true} deps=${[stepData, height]} />
      <div class="step-ctl">
        <input type="range" min="0" max=${Math.max(0, allSteps.length - 1)}
          value=${Math.max(0, allSteps.indexOf(stepVal) >= 0 ? allSteps.indexOf(stepVal) : allSteps.length - 1)}
          onInput=${(ev) => set({ step: allSteps[parseInt(ev.target.value)] })} />
        <span class="mono small">step ${fmtNum(stepVal)}</span>
        ${stepData.runs.map((r) => html`<span class="leg"><span class="tip-sw" style=${{ background: color(r.run_id) }}></span>${(byId.get(r.run_id) || {}).name} @ ${fmtNum(r.step)}</span>`)}
      </div>`;
  }

  return html`
    <div class=${"panel" + (expanded ? " expanded" : "")}>
      <div class="panel-head">
        <span class="chip">histogram</span>
        <${Select} label="Key" value=${panel.key || ""} options=${[{ value: "", label: "choose…" }, ...keys.hists.map((k) => ({ value: k }))]}
          onChange=${(v) => set({ key: v, range: null, step: null })} />
        <${Select} label="View" value=${panel.view} options=${VIEWS} onChange=${(v) => set({ view: v })} />
        ${panel.view === "heatmap" && html`<${Toggle} label="log color" value=${!!panel.logColor} onChange=${(v) => set({ logColor: v })} />
          <${Toggle} label="full range" value=${!!panel.fullRange} onChange=${(v) => set({ fullRange: v })}
            title="Off: value axis covers the 0.5–99.5% quantiles (tails outside are counted in the tooltip). On: min to max." />`}
        <span class="spacer"></span>
        ${loading && html`<span class="spinner"></span>`}
        ${panel.range && html`<button class="btn small ghost" onClick=${() => set({ range: null })}>Reset zoom</button>`}
        <button class="icon-btn" title=${expanded ? "Collapse" : "Expand"} onClick=${onExpand}>${expanded ? "⤡" : "⤢"}</button>
        <button class="icon-btn" title="Remove panel" onClick=${onRemove}>×</button>
      </div>
      <div class="chart hist-chart">${body}</div>
    </div>`;
}
