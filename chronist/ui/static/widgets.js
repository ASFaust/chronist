// Small reusable widgets.
import { html, useState, useRef, useEffect } from "./vendor/htm-preact.module.js";

export function useOutside(ref, onOutside, active = true) {
  useEffect(() => {
    if (!active) return;
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) onOutside(); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [active]);
}

// Dropdown with checkboxes + search. options: [{value, label, kind?}]
export function MultiSelect({ label, options, value, onChange, single = false, placeholder = "none", title }) {
  const [open, setOpen] = useState(false);
  const [f, setF] = useState("");
  const ref = useRef(null);
  useOutside(ref, () => setOpen(false), open);
  const sel = new Set(value);
  const shown = options.filter((o) => (o.label || o.value).toLowerCase().includes(f.toLowerCase()));
  const toggle = (v) => {
    if (single) { onChange(sel.has(v) ? [] : [v]); setOpen(false); return; }
    onChange(sel.has(v) ? value.filter((x) => x !== v) : [...value, v]);
  };
  const summary = value.length === 0 ? placeholder
    : value.length <= 2 ? value.join(", ") : `${value[0]} +${value.length - 1}`;
  return html`
    <div class="ms" ref=${ref}>
      <button class=${"ms-btn" + (value.length ? " on" : "")} title=${title || label} onClick=${() => setOpen(!open)}>
        <span class="ms-label">${label}</span><span class="ms-val">${summary}</span><span class="caret">▾</span>
      </button>
      ${open && html`
        <div class="popover ms-pop">
          ${options.length > 8 && html`<input class="input" placeholder="Search" value=${f}
              onInput=${(e) => setF(e.target.value)} autofocus />`}
          <div class="picker-list">
            ${shown.map((o) => html`
              <label class="picker-item">
                <input type=${single ? "radio" : "checkbox"} checked=${sel.has(o.value)} onChange=${() => toggle(o.value)} />
                <span>${o.label || o.value}</span>${o.kind && html`<span class="s-kind">${o.kind}</span>`}
              </label>`)}
            ${shown.length === 0 && html`<div class="muted pad">Nothing here</div>`}
          </div>
          ${!single && value.length > 0 && html`<button class="btn small ghost" onClick=${() => onChange([])}>Clear</button>`}
        </div>`}
    </div>`;
}

export function Select({ label, value, options, onChange, title }) {
  return html`
    <label class="sel" title=${title || label}>
      <span class="ms-label">${label}</span>
      <select value=${value} onChange=${(e) => onChange(e.target.value)}>
        ${options.map((o) => html`<option value=${o.value}>${o.label || o.value}</option>`)}
      </select>
    </label>`;
}

export function Toggle({ label, value, onChange, title }) {
  return html`<button class=${"tog" + (value ? " on" : "")} aria-pressed=${value} title=${title || label}
    onClick=${() => onChange(!value)}>${label}</button>`;
}

export function Modal({ title, children, onClose }) {
  useEffect(() => {
    const h = (e) => { if (e.key === "Escape") onClose && onClose(); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, []);
  return html`
    <div class="modal-back" onMouseDown=${(e) => { if (e.target === e.currentTarget && onClose) onClose(); }}>
      <div class="modal" role="dialog" aria-label=${title}>
        <h3>${title}</h3>
        ${children}
      </div>
    </div>`;
}
