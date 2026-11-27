// Chronist UI root.
import { html, render, useState, useEffect, useMemo, useRef, useCallback } from "./vendor/htm-preact.module.js";
import { api, openStream, setAuthHandler, setToken, getToken, AuthError } from "./api.js";
import { QueryInput } from "./query.js";
import { RunTable } from "./table.js";
import { Panel } from "./plot.js";
import { RunDrawer } from "./drawer.js";
import { Modal } from "./widgets.js";
import { SlotAllocator, cmpValues, newPanel, readHashState, runFieldValue, throttle, writeHashState, debounce } from "./util.js";

const PLOT_DEFAULT_N = 20;  // with no selection, plot the first N matching runs
const THEME_KEY = "chronist.theme";

function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}
try { applyTheme(localStorage.getItem(THEME_KEY)); } catch {}

function App() {
  const init = useMemo(() => readHashState() || {}, []);
  const [projects, setProjects] = useState(null);
  const [project, setProject] = useState(init.project || null);
  const [q, setQ] = useState(init.q || "");
  const [qErr, setQErr] = useState(null);
  const [runs, setRuns] = useState([]);
  const [total, setTotal] = useState(0);
  const [keys, setKeys] = useState({ config: [], metrics: [] });
  const [selected, setSelected] = useState(new Set(init.sel || []));
  const [cols, setCols] = useState(init.cols || []);
  const [sort, setSort] = useState(init.sort || { key: "created", desc: true });
  const [panels, setPanels] = useState(init.panels || null);
  const [expanded, setExpanded] = useState(null);
  const [drawer, setDrawer] = useState(null);
  const [auth, setAuth] = useState(null);
  const [live, setLive] = useState({ seq: 0, runs: null });
  const [split, setSplit] = useState(init.split || 38);
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem(THEME_KEY) || "auto"; } catch { return "auto"; } });
  const slotsRef = useRef(new SlotAllocator());
  const lastClicked = useRef(null);

  useEffect(() => { setAuthHandler((status) => setAuth(status)); }, []);

  // ---- projects ----
  const loadProjects = useCallback(async () => {
    try {
      const r = await api.projects();
      setProjects(r.projects);
      if (!project && r.projects.length) setProject(r.projects[0].name);
    } catch (e) { if (!(e instanceof AuthError)) console.error(e); }
  }, [project]);
  useEffect(() => { loadProjects(); }, []);

  // ---- runs + keys ----
  const runsAbort = useRef(null);
  const loadRuns = useCallback(async () => {
    if (!project) return;
    if (runsAbort.current) runsAbort.current.abort();
    runsAbort.current = new AbortController();
    try {
      const r = await api.runs(project, q, runsAbort.current.signal);
      setRuns(r.runs); setTotal(r.total); setQErr(null);
    } catch (e) {
      if (e.name === "AbortError" || e instanceof AuthError) return;
      setQErr(e.message);
    }
  }, [project, q]);
  const loadKeys = useCallback(async () => {
    if (!project) return;
    try { setKeys(await api.keys(project)); } catch (e) {}
  }, [project]);
  useEffect(() => { loadRuns(); }, [loadRuns]);
  useEffect(() => { loadKeys(); }, [loadKeys]);

  // default panels: one per metric (up to 4) the first time a project is opened
  useEffect(() => {
    if (panels === null && keys.metrics.length) {
      setPanels(keys.metrics.slice(0, 4).map((m) => newPanel([m])));
    }
  }, [keys, panels]);

  // ---- live updates ----
  const reloadRunsSoon = useMemo(() => throttle(() => { loadRuns(); }, 2000), [loadRuns]);
  const reloadKeysSoon = useMemo(() => throttle(() => { loadKeys(); }, 5000), [loadKeys]);
  useEffect(() => {
    if (!project) return;
    return openStream(project, (evs) => {
      const changed = new Set();
      let runEv = false;
      for (const ev of evs) {
        if (ev.type === "points") changed.add(ev.run_id);
        else runEv = true;
      }
      if (changed.size) { setLive((l) => ({ seq: l.seq + 1, runs: changed })); reloadRunsSoon(); reloadKeysSoon(); }
      if (runEv) { reloadRunsSoon(); loadProjects(); }
    });
  }, [project, reloadRunsSoon]);

  // ---- URL state ----
  useEffect(() => {
    writeHashState({ project, q, sel: [...selected], cols, sort, panels, split });
  }, [project, q, selected, cols, sort, panels, split]);

  // ---- derived ----
  const sortedRuns = useMemo(() => {
    const out = runs.slice();
    const k = sort.key;
    out.sort((a, b) => {
      const c = cmpValues(runFieldValue(a, k), runFieldValue(b, k));
      return sort.desc ? -c : c;
    });
    return out;
  }, [runs, sort]);
  const runsById = useMemo(() => new Map(runs.map((r) => [r.id, r])), [runs]);
  const plotRuns = useMemo(() => {
    if (selected.size) return sortedRuns.filter((r) => selected.has(r.id));
    return sortedRuns.slice(0, PLOT_DEFAULT_N);
  }, [sortedRuns, selected]);
  // Grouped panels aggregate over everything that matches unless runs are selected.
  const groupRuns = useMemo(() => selected.size ? plotRuns : sortedRuns.slice(0, 1000), [plotRuns, sortedRuns, selected]);
  const slots = useMemo(() => new Map(slotsRef.current.assign(plotRuns.map((r) => r.id))), [plotRuns]);

  // ---- handlers ----
  function onToggle(id, e) {
    const s = new Set(selected);
    if (e && e.shiftKey && lastClicked.current != null) {
      const ids = sortedRuns.map((r) => r.id);
      const a = ids.indexOf(lastClicked.current), b = ids.indexOf(id);
      if (a >= 0 && b >= 0) {
        const [lo, hi] = a < b ? [a, b] : [b, a];
        for (let i = lo; i <= hi; i++) s.add(ids[i]);
        setSelected(s); lastClicked.current = id;
        return;
      }
    }
    s.has(id) ? s.delete(id) : s.add(id);
    lastClicked.current = id;
    setSelected(s);
  }
  function onSelectAll(on) {
    setSelected(on ? new Set(sortedRuns.map((r) => r.id)) : new Set());
  }
  function onSort(key) {
    setSort(sort.key === key ? { key, desc: !sort.desc } : { key, desc: key !== "name" });
  }
  const updatePanel = (id, patch) => setPanels((ps) => ps.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  const removePanel = (id) => setPanels((ps) => ps.filter((p) => p.id !== id));

  async function bulkTag() {
    const t = prompt(`Add tag to ${selected.size} runs:`);
    if (!t) return;
    await Promise.all([...selected].map((id) => api.patch(id, { add_tags: [t.trim()] })));
    loadRuns();
  }
  async function bulkDelete() {
    if (!confirm(`Delete ${selected.size} runs and all their data? This cannot be undone.`)) return;
    for (const id of selected) await api.del(id);
    setSelected(new Set());
    loadRuns(); loadProjects();
  }
  function setThemeChoice(t) {
    setTheme(t); applyTheme(t);
    try { localStorage.setItem(THEME_KEY, t); } catch {}
  }

  // drag the splitter between table and panels
  function startSplit(e) {
    e.preventDefault();
    const move = (ev) => {
      const pct = (ev.clientY / window.innerHeight) * 100;
      setSplit(Math.max(12, Math.min(80, pct)));
    };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  if (projects && projects.length === 0) {
    return html`<div class="welcome">
      <h1>chronist</h1>
      <p>No runs yet. Log one:</p>
      <pre class="code">import chronist
run = chronist.init(project="my-project", config={"lr": 3e-4})
for step in range(1000):
    run.log({"loss": 1 / (step + 1)})
run.finish()</pre>
      <p class="muted">With <code>chronist ui</code>, offline runs in <code>~/.chronist</code> show up automatically.
      Otherwise set <code>CHRONIST_URL</code> or run <code>chronist sync</code>.</p>
      ${auth && html`<${TokenModal} status=${auth} onDone=${() => { setAuth(null); loadProjects(); }} />`}
    </div>`;
  }

  return html`
    <div class="app" style=${{ "--split": split + "vh" }}>
      <header class="top">
        <div class="brand"><span class="logo" aria-hidden="true">◷</span>chronist</div>
        <select class="proj" value=${project || ""} onChange=${(e) => { setProject(e.target.value); setSelected(new Set()); setPanels(null); setQ(""); }}
          title="Project">
          ${(projects || []).map((p) => html`<option value=${p.name}>${p.name} (${p.runs})</option>`)}
        </select>
        <${QueryInput} project=${project} value=${q} error=${qErr} onChange=${setQ} />
        <div class="top-right">
          <span class="count">${total} runs${selected.size ? html` · <b>${selected.size}</b> selected` : ""}</span>
          <a class="btn small ghost" href=${api.exportUrl(project, q, [], "parquet")} title="Download matching runs as Parquet">Export</a>
          <select class="theme" value=${theme} onChange=${(e) => setThemeChoice(e.target.value)} title="Theme">
            <option value="auto">Auto</option><option value="light">Light</option><option value="dark">Dark</option>
          </select>
        </div>
      </header>
      <section class="runs-pane">
        ${selected.size > 0 && html`
          <div class="bulk">
            <span>${selected.size} selected</span>
            <button class="btn small ghost" onClick=${() => setSelected(new Set())}>Clear</button>
            <button class="btn small ghost" onClick=${bulkTag}>Tag…</button>
            <button class="btn small ghost danger" onClick=${bulkDelete}>Delete…</button>
          </div>`}
        <${RunTable} runs=${sortedRuns} selected=${selected} slots=${slots} cols=${cols} sort=${sort}
          onSort=${onSort} onToggle=${onToggle} onSelectAll=${onSelectAll} onOpen=${setDrawer}
          keys=${keys} onCols=${setCols} />
      </section>
      <div class="splitter" onMouseDown=${startSplit} title="Drag to resize"></div>
      <section class="panels-pane">
        <div class="panels-bar">
          <span class="muted">${selected.size ? `Plotting ${plotRuns.length} selected runs`
            : `Plotting the first ${plotRuns.length} matching runs (select runs to choose); grouped panels use all matches`}</span>
          <span class="spacer"></span>
          <button class="btn small" onClick=${() => setPanels([...(panels || []), newPanel(keys.metrics.slice(0, 1))])}>+ Panel</button>
        </div>
        <div class="panels">
          ${(panels || []).map((p) => html`
            <${Panel} key=${p.id} panel=${p} runs=${p.group.length ? groupRuns : plotRuns} runsById=${runsById}
              slots=${slots} keys=${keys} live=${live}
              expanded=${expanded === p.id} onExpand=${() => setExpanded(expanded === p.id ? null : p.id)}
              onChange=${(patch) => updatePanel(p.id, patch)} onRemove=${() => removePanel(p.id)} />`)}
        </div>
      </section>
      ${drawer != null && html`<${RunDrawer} runId=${drawer} onClose=${() => setDrawer(null)}
          onChanged=${loadRuns} onDeleted=${(id) => { setDrawer(null); const s = new Set(selected); s.delete(id); setSelected(s); loadRuns(); }} />`}
      ${auth && html`<${TokenModal} status=${auth} onDone=${() => { setAuth(null); loadProjects(); loadRuns(); loadKeys(); }} />`}
    </div>`;
}

function TokenModal({ status, onDone }) {
  const [t, setT] = useState(getToken() || "");
  return html`<${Modal} title=${status === 403 ? "This token is read-only" : "Access token"}>
    <p class="muted">${status === 403 ? "Changes need a write token." : "This server requires a token."}</p>
    <input class="input" type="password" value=${t} onInput=${(e) => setT(e.target.value)} autofocus
      onKeyDown=${(e) => { if (e.key === "Enter") { setToken(t); onDone(); } }} />
    <div class="modal-actions">
      ${status === 403 && html`<button class="btn ghost" onClick=${onDone}>Keep read-only</button>`}
      <button class="btn" onClick=${() => { setToken(t); onDone(); }}>Save</button>
    </div>
  <//>`;
}

render(html`<${App} />`, document.getElementById("app"));
