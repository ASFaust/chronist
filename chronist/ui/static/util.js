// Shared helpers: formatting, colors, smoothing, URL state.

// Categorical palette (validated reference order; see docs). Light / dark steps.
export const PALETTE = {
  light: ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"],
  dark: ["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#0aac23", "#9887ef", "#eb6263"],
};
// Secondary encoding once the 8 hues are used up (and to tell metrics apart).
export const DASHES = [[], [6, 4], [2, 3], [10, 3, 2, 3], [1, 5]];

export function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t === "dark") return true;
  if (t === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function slotColor(slot) {
  const p = isDark() ? PALETTE.dark : PALETTE.light;
  return p[((slot % p.length) + p.length) % p.length];
}
export function slotDash(slot) { return DASHES[Math.floor(slot / 8) % DASHES.length]; }

export function withAlpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// Stable run -> color slot assignment for the session: a run keeps its slot
// while it stays plotted; new runs take the least-used slot.
export class SlotAllocator {
  constructor() { this.slots = new Map(); }
  assign(ids) {
    const live = new Set(ids);
    for (const id of [...this.slots.keys()]) if (!live.has(id)) this.slots.delete(id);
    const used = new Map();
    for (const s of this.slots.values()) used.set(s, (used.get(s) || 0) + 1);
    for (const id of ids) {
      if (this.slots.has(id)) continue;
      let s = 0;
      while (used.has(s)) s++;
      this.slots.set(id, s);
      used.set(s, 1);
    }
    return this.slots;
  }
}

export function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e7) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return Number(v.toPrecision(5)).toString();
}

export function fmtStep(v) {
  if (v == null) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "G";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
  return String(Math.round(v * 100) / 100);
}

export function fmtDuration(s) {
  if (s == null || !isFinite(s)) return "–";
  if (s < 60) return s.toFixed(0) + "s";
  if (s < 3600) return (s / 60).toFixed(1) + "m";
  if (s < 86400) return (s / 3600).toFixed(1) + "h";
  return (s / 86400).toFixed(1) + "d";
}

export function fmtAgo(t) {
  if (!t) return "–";
  const d = Date.now() / 1000 - t;
  if (d < 45) return "just now";
  if (d < 3600) return Math.round(d / 60) + "m ago";
  if (d < 86400) return Math.round(d / 3600) + "h ago";
  if (d < 86400 * 30) return Math.round(d / 86400) + "d ago";
  return new Date(t * 1000).toISOString().slice(0, 10);
}

export function fmtValue(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return JSON.stringify(v);
  return String(v);
}

// TensorBoard-style debiased EMA; weight in [0, 1).
export function ema(ys, weight) {
  if (!weight) return ys;
  const out = new Array(ys.length);
  let last = 0, n = 0;
  for (let i = 0; i < ys.length; i++) {
    const y = ys[i];
    if (y == null || !isFinite(y)) { out[i] = y; continue; }
    last = last * weight + (1 - weight) * y;
    n++;
    out[i] = last / (1 - Math.pow(weight, n));
  }
  return out;
}

export function debounce(fn, ms) {
  let t;
  const d = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}

// Throttle with trailing call.
export function throttle(fn, ms) {
  let last = 0, timer = null, pending = null;
  return (...a) => {
    pending = a;
    const now = Date.now();
    if (now - last >= ms && !timer) {
      last = now; fn(...pending); pending = null;
    } else if (!timer) {
      timer = setTimeout(() => { timer = null; last = Date.now(); if (pending) fn(...pending); pending = null; }, ms - (now - last));
    }
  };
}

// ---- URL-encoded view state ----
export function readHashState() {
  const h = location.hash.slice(1);
  if (!h) return null;
  try { return JSON.parse(decodeURIComponent(h)); } catch { return null; }
}
export function writeHashState(st) {
  const h = "#" + encodeURIComponent(JSON.stringify(st));
  if (h !== location.hash) history.replaceState(null, "", h);
}

let _pid = 0;
export function newPanel(keys = [], extra = {}) {
  return { id: Date.now().toString(36) + (_pid++), keys, x: "step", group: [], agg: "mean_std",
           smooth: 0, logx: false, logy: false, band: true, colorBy: "", range: null, ...extra };
}

export function runFieldValue(run, key) {
  if (key.startsWith("last(")) {
    const s = run.summary && run.summary[key.slice(5, -1)];
    return s ? s.value : undefined;
  }
  if (key in run && key !== "config" && key !== "summary") return run[key];
  return run.config ? run.config[key] : undefined;
}

export function cmpValues(a, b) {
  const an = a === undefined || a === null, bn = b === undefined || b === null;
  if (an || bn) return an === bn ? 0 : an ? 1 : -1;  // missing last
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}
