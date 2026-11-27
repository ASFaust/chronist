// Search bar: query language input with server-side autocomplete.
import { html, useState, useEffect, useRef } from "./vendor/htm-preact.module.js";
import { api } from "./api.js";
import { debounce } from "./util.js";

export function QueryInput({ project, value, error, onChange }) {
  const [text, setText] = useState(value || "");
  const [items, setItems] = useState([]);
  const [ctx, setCtx] = useState(null);
  const [active, setActive] = useState(0);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const abort = useRef(null);

  useEffect(() => { setText(value || ""); }, [value]);

  const fetchSuggest = useRef(debounce(async (proj, t, caret) => {
    if (abort.current) abort.current.abort();
    abort.current = new AbortController();
    try {
      const r = await api.suggest(proj, t.slice(0, caret), abort.current.signal);
      setItems(r.items || []);
      setCtx({ start: r.start, caret, context: r.context });
      setActive(0);
      setOpen((r.items || []).length > 0);
    } catch (e) { if (e.name !== "AbortError") setOpen(false); }
  }, 70)).current;

  const commit = useRef(debounce((t) => onChange(t), 350)).current;

  function suggestNow(t) {
    const el = ref.current;
    fetchSuggest(project, t, el ? el.selectionStart : t.length);
  }

  function onInput(e) {
    const t = e.target.value;
    setText(t);
    suggestNow(t);
    commit(t);
  }

  function accept(item) {
    const el = ref.current;
    if (!ctx || !el) return;
    const before = text.slice(0, ctx.start);
    let after = text.slice(ctx.caret);
    // drop the rest of the token being replaced
    after = after.replace(/^[^\s()\[\],]*/, "");
    let ins = item.insert;
    const needsSpace = !after.startsWith(" ");
    const t = before + ins + (needsSpace ? " " : "") + after;
    const caret = (before + ins).length + 1;
    setText(t);
    setOpen(false);
    commit.cancel();
    onChange(t);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(caret, caret);
      fetchSuggest(project, t, caret);
    });
  }

  function onKeyDown(e) {
    if (open && items.length) {
      if (e.key === "ArrowDown") { e.preventDefault(); setActive((active + 1) % items.length); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); setActive((active - 1 + items.length) % items.length); return; }
      if (e.key === "Tab" || (e.key === "Enter" && ctx && ctx.context !== "operator")) {
        e.preventDefault(); accept(items[active]); return;
      }
      if (e.key === "Escape") { setOpen(false); return; }
    }
    if (e.key === "Enter") { commit.cancel(); onChange(text); setOpen(false); }
    if (e.key === "Escape" && !open && text) { setText(""); commit.cancel(); onChange(""); }
  }

  return html`
    <div class="query">
      <div class=${"query-box" + (error ? " has-error" : "")}>
        <svg class="query-icon" viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M11 11l3.5 3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        <input ref=${ref} value=${text} spellcheck="false" autocomplete="off"
          placeholder='Filter runs, e.g.  config.model.arch = "lstm" and metric.loss < 0.1'
          onInput=${onInput} onKeyDown=${onKeyDown}
          onFocus=${() => suggestNow(text)} onClick=${() => suggestNow(text)}
          onBlur=${() => setTimeout(() => setOpen(false), 150)} />
        ${text && html`<button class="icon-btn" title="Clear" onClick=${() => { setText(""); onChange(""); }}>×</button>`}
      </div>
      ${error && html`<div class="query-error">${error}</div>`}
      ${open && html`
        <ul class="suggest">
          ${items.map((it, i) => html`
            <li class=${i === active ? "active" : ""} onMouseDown=${(e) => { e.preventDefault(); accept(it); }}
                onMouseEnter=${() => setActive(i)}>
              <span class="s-label">${it.label}</span>
              <span class="s-kind">${it.detail || it.kind}</span>
            </li>`)}
        </ul>`}
    </div>`;
}
