// Run table: virtualized rows, sortable, selectable columns.
import { html, useState, useRef, useEffect, useMemo } from "./vendor/htm-preact.module.js";
import { configCol, fmtAgo, fmtNum, fmtStep, fmtValue, runFieldValue, slotColor, slotDash } from "./util.js";

const ROW_H = 30;

const BASE_COLS = [
  { key: "name", label: "Name", width: 200 },
  { key: "status", label: "Status", width: 96 },
  { key: "steps", label: "Steps", width: 76, num: true },
  { key: "created", label: "Created", width: 92 },
];

export function StatusBadge({ status }) {
  const icon = { running: "●", finished: "✓", crashed: "!" }[status] || "?";
  return html`<span class=${"badge st-" + status}><span class="badge-icon" aria-hidden="true">${icon}</span>${status}</span>`;
}

function Swatch({ slot }) {
  if (slot === undefined) return html`<span class="swatch empty"></span>`;
  const dash = slotDash(slot);
  return html`<svg class="swatch" viewBox="0 0 18 8" aria-hidden="true">
    <line x1="1" y1="4" x2="17" y2="4" stroke=${slotColor(slot)} stroke-width="3" stroke-linecap="round"
      stroke-dasharray=${dash.length ? dash.map((d) => d / 2).join(" ") : ""} /></svg>`;
}

function cell(run, col) {
  const v = runFieldValue(run, col.key);
  if (col.key === "status") return html`<${StatusBadge} status=${run.status} />`;
  if (col.key === "created") return html`<span title=${new Date(run.created * 1000).toLocaleString()}>${fmtAgo(run.created)}</span>`;
  if (col.key === "steps") return fmtStep(v);
  if (col.key === "name") return run.name;
  return typeof v === "number" ? fmtNum(v) : v === undefined ? html`<span class="muted">–</span>` : fmtValue(v);
}

export function RunTable({ runs, selected, slots, cols, sort, onSort, onToggle, onSelectAll, onOpen, keys, onCols }) {
  const scroller = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(400);
  const [picker, setPicker] = useState(false);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const columns = useMemo(() => [
    ...BASE_COLS,
    ...cols.map((k) => ({ key: k, label: k, width: 110, extra: true,
                          num: k.startsWith("last(") || runs.slice(0, 50).some((r) => typeof runFieldValue(r, k) === "number") })),
  ], [cols, runs]);
  const template = "34px " + columns.map((c) => c.width + "px").join(" ") + " 1fr";

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - 10);
  const last = Math.min(runs.length, Math.ceil((scrollTop + height) / ROW_H) + 10);
  const allSel = runs.length > 0 && runs.every((r) => selected.has(r.id));

  return html`
    <div class="table">
      <div class="thead" style=${{ gridTemplateColumns: template }}>
        <div class="th check">
          <input type="checkbox" checked=${allSel} title="Select all matching runs"
            onChange=${() => onSelectAll(!allSel)} />
        </div>
        ${columns.map((c) => html`
          <div class=${"th" + (c.num ? " num" : "") + (sort.key === c.key ? " sorted" : "")}
               onClick=${() => onSort(c.key)} title=${c.key}>
            <span class="th-label">${c.label}</span>
            ${sort.key === c.key && html`<span class="sort-ind">${sort.desc ? "▼" : "▲"}</span>`}
            ${c.extra && html`<button class="th-x" title="Remove column"
              onClick=${(e) => { e.stopPropagation(); onCols(cols.filter((k) => k !== c.key)); }}>×</button>`}
          </div>`)}
        <div class="th add-col">
          <button class="btn small ghost" onClick=${() => setPicker(!picker)}>+ Column</button>
          ${picker && html`<${ColumnPicker} keys=${keys} cols=${cols} onCols=${onCols} onClose=${() => setPicker(false)} />`}
        </div>
      </div>
      <div class="tbody" ref=${scroller} onScroll=${(e) => setScrollTop(e.target.scrollTop)}>
        <div style=${{ height: runs.length * ROW_H + "px", position: "relative" }}>
          ${runs.slice(first, last).map((r, i) => html`
            <div key=${r.id} class=${"tr" + (selected.has(r.id) ? " selected" : "")}
                 style=${{ gridTemplateColumns: template, top: (first + i) * ROW_H + "px" }}
                 onClick=${(e) => onToggle(r.id, e)}>
              <div class="td check" onClick=${(e) => e.stopPropagation()}>
                <input type="checkbox" checked=${selected.has(r.id)} onChange=${(e) => onToggle(r.id, e)} />
              </div>
              ${columns.map((c) => c.key === "name" ? html`
                <div class="td name">
                  <${Swatch} slot=${slots.get(r.id)} />
                  <a href="#" onClick=${(e) => { e.preventDefault(); e.stopPropagation(); onOpen(r.id); }} title="Run details">${r.name}</a>
                  ${r.meta && r.meta.code && r.meta.code.outside_imports && r.meta.code.outside_imports.length > 0 &&
                    html`<span class="flag" title="Code snapshot may be incomplete: modules imported from outside the code root">⚠</span>`}
                  ${r.tags && r.tags.length > 0 && html`<span class="tags">${r.tags.slice(0, 3).map((t) => html`<span class="tag">${t}</span>`)}</span>`}
                </div>` : html`<div class=${"td" + (c.num ? " num" : "")}>${cell(r, c)}</div>`)}
              <div class="td"></div>
            </div>`)}
        </div>
        ${runs.length === 0 && html`<div class="empty-state">No runs match.</div>`}
      </div>
    </div>`;
}

function ColumnPicker({ keys, cols, onCols, onClose }) {
  const [f, setF] = useState("");
  const ref = useRef(null);
  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    setTimeout(() => document.addEventListener("mousedown", h));
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const all = [
    ...keys.config.map((k) => ({ key: configCol(k.key), kind: "config" })),
    ...keys.metrics.map((m) => ({ key: `last(${m})`, kind: "metric" })),
  ].filter((k) => k.key.toLowerCase().includes(f.toLowerCase()));
  const toggle = (k) => onCols(cols.includes(k) ? cols.filter((c) => c !== k) : [...cols, k]);
  return html`
    <div class="popover picker" ref=${ref} onClick=${(e) => e.stopPropagation()}>
      <input class="input" placeholder="Search keys" value=${f} onInput=${(e) => setF(e.target.value)} autofocus />
      <div class="picker-list">
        ${all.map((k) => html`
          <label class="picker-item">
            <input type="checkbox" checked=${cols.includes(k.key)} onChange=${() => toggle(k.key)} />
            <span>${k.key}</span><span class="s-kind">${k.kind}</span>
          </label>`)}
        ${all.length === 0 && html`<div class="muted pad">No keys</div>`}
      </div>
    </div>`;
}
