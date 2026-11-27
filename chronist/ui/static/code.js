// "Code" tab: browse a run's code snapshot.
import { html, useState, useEffect, useMemo, useRef } from "./vendor/htm-preact.module.js";
import { api } from "./api.js";
import { fmtNum } from "./util.js";

// ---- syntax highlighting (lazy: only loaded when a file is opened) ----
const LANG_BY_EXT = {
  py: "python", pyi: "python", sh: "bash", bash: "bash", zsh: "bash", yml: "yaml", yaml: "yaml", json: "json",
  toml: "ini", ini: "ini", cfg: "ini", md: "markdown", c: "c", h: "c", cc: "cpp", cpp: "cpp", hpp: "cpp",
  cu: "cpp", cuh: "cpp", js: "javascript", mjs: "javascript", ts: "javascript", html: "xml", xml: "xml",
  rs: "rust", diff: "diff", patch: "diff", txt: "plaintext",
};
const LANG_BY_NAME = { Makefile: "makefile", Dockerfile: "dockerfile", "CMakeLists.txt": "cmake" };
let hljsP = null;
const loaded = new Set();
async function highlighter(lang) {
  if (!hljsP) hljsP = import("./vendor/hljs/core.min.js").then((m) => m.default);
  const hljs = await hljsP;
  if (lang && !loaded.has(lang)) {
    const mod = await import(`./vendor/hljs/${lang}.min.js`);
    hljs.registerLanguage(lang, mod.default);
    loaded.add(lang);
  }
  return hljs;
}
function langFor(path) {
  const name = path.split("/").pop();
  if (LANG_BY_NAME[name]) return LANG_BY_NAME[name];
  const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  return LANG_BY_EXT[ext] || null;
}

function fmtBytes(n) {
  if (n == null) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

// ---- file tree ----
function buildTree(files) {
  const root = { name: "", dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    for (const p of parts.slice(0, -1)) {
      if (!node.dirs.has(p)) node.dirs.set(p, { name: p, dirs: new Map(), files: [] });
      node = node.dirs.get(p);
    }
    node.files.push({ ...f, name: parts[parts.length - 1] });
  }
  return root;
}

function Tree({ node, depth, open, toggle, selected, onSelect, prefix }) {
  const dirs = [...node.dirs.values()].sort((a, b) => a.name.localeCompare(b.name));
  const files = node.files.slice().sort((a, b) => a.name.localeCompare(b.name));
  return html`${dirs.map((d) => {
      const p = prefix + d.name + "/";
      const isOpen = open.has(p);
      return html`
        <div class="ft-row ft-dir" style=${{ paddingLeft: 8 + depth * 14 + "px" }} onClick=${() => toggle(p)}>
          <span class="ft-caret">${isOpen ? "▾" : "▸"}</span>${d.name}/
        </div>
        ${isOpen && html`<${Tree} node=${d} depth=${depth + 1} open=${open} toggle=${toggle}
            selected=${selected} onSelect=${onSelect} prefix=${p} />`}`;
    })}
    ${files.map((f) => html`
      <div class=${"ft-row ft-file" + (selected === f.path ? " on" : "")} style=${{ paddingLeft: 22 + depth * 14 + "px" }}
           onClick=${() => onSelect(f.path)} title=${f.path}>
        <span class="ft-name">${f.name}</span>
        ${f.symlink ? html`<span class="ft-meta">→ ${f.symlink}</span>` : html`<span class="ft-meta">${fmtBytes(f.size)}</span>`}
      </div>`)}`;
}

function Viewer({ runId, path, file }) {
  const [text, setText] = useState(null);
  const [err, setErr] = useState(null);
  const codeRef = useRef(null);
  useEffect(() => {
    setText(null); setErr(null);
    if (!path) return;
    if (file && file.symlink) { setText(`symlink → ${file.symlink}`); return; }
    let alive = true;
    api.codeFile(runId, path).then((t) => alive && setText(t)).catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [runId, path]);
  useEffect(() => {
    if (text == null || !codeRef.current) return;
    const lang = path === "__env__" ? "json" : path === "__packages__" ? null : langFor(path);
    if (!lang || text.length > 400_000) { codeRef.current.textContent = text; return; }
    let alive = true;
    highlighter(lang).then((hljs) => {
      if (!alive || !codeRef.current) return;
      codeRef.current.innerHTML = hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
    }).catch(() => { if (codeRef.current) codeRef.current.textContent = text; });
    return () => { alive = false; };
  }, [text]);
  if (!path) return html`<div class="cv-empty muted">Select a file</div>`;
  if (err) return html`<div class="cv-empty query-error">${err}</div>`;
  if (text == null) return html`<div class="cv-empty muted">Loading…</div>`;
  const n = text.split("\n").length - (text.endsWith("\n") ? 1 : 0);
  return html`
    <div class="cv">
      <div class="cv-head mono">${path === "__env__" ? "environment" : path === "__packages__" ? "packages" : path}</div>
      <div class="cv-body">
        <pre class="cv-lines" aria-hidden="true">${Array.from({ length: Math.max(n, 1) }, (_, i) => i + 1).join("\n")}</pre>
        <pre class="cv-code hljs"><code ref=${codeRef}></code></pre>
      </div>
    </div>`;
}

export function CodeTab({ run }) {
  const [m, setM] = useState(null);
  const [err, setErr] = useState(null);
  const [sel, setSel] = useState(null);
  const [filter, setFilter] = useState("");
  const [open, setOpen] = useState(new Set());
  const [showSkipped, setShowSkipped] = useState(false);
  useEffect(() => {
    setM(null); setErr(null); setSel(null);
    api.code(run.id).then((r) => {
      setM(r);
      // open the directories on the path to the entry script, select it
      const ref = (run.meta.code || {}).reference || "";
      const root = (r.root || {}).root || "";
      const rel = ref.startsWith(root + "/") ? ref.slice(root.length + 1) : null;
      const first = rel && r.files.some((f) => f.path === rel) ? rel : (r.files.find((f) => f.path.endsWith(".py")) || r.files[0] || {}).path;
      if (first) {
        const parts = first.split("/");
        setOpen(new Set(parts.slice(0, -1).map((_, i) => parts.slice(0, i + 1).join("/") + "/")));
        setSel(first);
      }
    }).catch((e) => setErr(e.status === 404 ? "This run has no code snapshot." : e.message));
  }, [run.id]);

  const shown = useMemo(() => {
    if (!m) return [];
    const f = filter.toLowerCase();
    return f ? m.files.filter((x) => x.path.toLowerCase().includes(f)) : m.files;
  }, [m, filter]);
  const tree = useMemo(() => buildTree(shown), [shown]);
  const openAll = filter ? new Set(shown.flatMap((f) => {
    const parts = f.path.split("/");
    return parts.slice(0, -1).map((_, i) => parts.slice(0, i + 1).join("/") + "/");
  })) : open;
  const toggle = (p) => { const s = new Set(open); s.has(p) ? s.delete(p) : s.add(p); setOpen(s); };

  if (err) return html`<div class="muted pad">${err}</div>`;
  if (!m) return html`<div class="muted pad">Loading…</div>`;
  const snap = run.meta.code || {};
  const outside = snap.outside_imports || [];
  const byReason = {};
  for (const s of m.skipped) byReason[s.reason] = (byReason[s.reason] || 0) + 1;
  const total = m.files.reduce((a, f) => a + (f.size || 0), 0);
  const file = m.files.find((f) => f.path === sel);
  return html`
    <div class="code-tab">
      <div class="code-info">
        <div><span class="muted">root</span> <span class="mono">${m.root.root}</span>
          <span class="chip">${m.root.mode}${m.root.chosen_by ? " · " + m.root.chosen_by : ""}${m.root.spec != null ? " · " + JSON.stringify(m.root.spec) : ""}</span></div>
        <div class="muted">${m.files.length} files · ${fmtBytes(total)}
          ${m.skipped.length > 0 && html` · <a href="#" onClick=${(e) => { e.preventDefault(); setShowSkipped(!showSkipped); }}>${m.skipped.length} skipped
            (${Object.entries(byReason).map(([r, n]) => `${n} ${r}`).join(", ")})</a>`}
          · manifest <span class="mono">${m.hash.slice(0, 12)}</span></div>
        ${outside.length > 0 && html`
          <div class="warn-box" role="note"><b>⚠ Snapshot may be incomplete.</b> These modules were imported from outside the code root:
            <ul>${outside.slice(0, 8).map((o) => html`<li><span class="mono">${o.module}</span> <span class="muted mono">${o.file}</span></li>`)}</ul>
            ${outside.length > 8 && html`<span class="muted">+${outside.length - 8} more</span>`}
          </div>`}
        ${showSkipped && html`<table class="kv skip">${m.skipped.map((s) => html`<tr><th>${s.path}</th><td>${s.reason}${s.size ? ` (${fmtBytes(s.size)})` : ""}</td></tr>`)}</table>`}
      </div>
      <div class="code-split">
        <div class="ft">
          <input class="input ft-filter" placeholder="Filter files" value=${filter} onInput=${(e) => setFilter(e.target.value)} />
          <div class="ft-list">
            <${Tree} node=${tree} depth=${0} open=${openAll} toggle=${toggle} selected=${sel} onSelect=${setSel} prefix="" />
            <div class="ft-sep"></div>
            <div class=${"ft-row ft-file" + (sel === "__env__" ? " on" : "")} style=${{ paddingLeft: "8px" }} onClick=${() => setSel("__env__")}>⚙ environment</div>
            <div class=${"ft-row ft-file" + (sel === "__packages__" ? " on" : "")} style=${{ paddingLeft: "8px" }} onClick=${() => setSel("__packages__")}>⚙ packages</div>
          </div>
        </div>
        <${Viewer} runId=${run.id} path=${sel} file=${file} />
      </div>
    </div>`;
}
