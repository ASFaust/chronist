// Run detail drawer: config, metadata, summary, tags, notes, rename, delete.
import { html, useState, useEffect } from "./vendor/htm-preact.module.js";
import { api } from "./api.js";
import { StatusBadge } from "./table.js";
import { CodeTab } from "./code.js";
import { fmtAgo, fmtDuration, fmtNum, fmtValue } from "./util.js";

export function RunDrawer({ runId, onClose, onChanged, onDeleted }) {
  const [run, setRun] = useState(null);
  const [err, setErr] = useState(null);
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [tag, setTag] = useState("");
  const [tab, setTab] = useState("config");

  async function load() {
    try {
      const r = await api.run(runId);
      setRun(r); setName(r.name); setNotes(r.notes || ""); setErr(null);
    } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); }, [runId]);
  useEffect(() => {
    const h = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, []);

  async function patch(body) {
    try { await api.patch(runId, body); await load(); onChanged(); } catch (e) { setErr(e.message); }
  }

  async function del() {
    if (!confirm(`Delete run "${run.name}" and all its data? This cannot be undone.`)) return;
    try { await api.del(runId); onDeleted(runId); } catch (e) { setErr(e.message); }
  }

  if (!run) return html`<aside class="drawer"><div class="drawer-head"><span class="muted">${err || "Loading…"}</span>
    <button class="icon-btn" onClick=${onClose}>×</button></div></aside>`;

  const m = run.meta || {};
  const dur = (run.finished || run.heartbeat || run.updated) - run.created;
  const metaRows = [
    ["id / uid", `${run.id} / ${run.uid}`],
    ["project", run.project],
    ["created", new Date(run.created * 1000).toLocaleString() + ` (${fmtAgo(run.created)})`],
    ["duration", fmtDuration(dur)],
    ["last heartbeat", fmtAgo(run.heartbeat)],
    ["host", m.hostname], ["python", m.python],
    ["git commit", m.git_commit ? m.git_commit.slice(0, 12) + (m.git_dirty ? " (dirty, diff " + (m.git_diff_hash || "?") + ")" : "") : null],
    ["git branch", m.git_branch], ["git remote", m.git_remote],
    ["cwd", m.cwd],
    ["code root", m.code && m.code.root ? `${m.code.root} (${m.code.mode}, ${m.code.chosen_by})` : null],
    ["code snapshot", m.code ? (m.code.aborted ? "aborted: " + m.code.aborted : m.code.manifest ? `${m.code.files} files, manifest ${m.code.manifest.slice(0, 12)}` : null) : "off"],
    ["command", m.command ? m.command.join(" ") : null],
  ].filter((r) => r[1]);

  return html`
    <aside class=${"drawer" + (tab === "code" ? " wide" : "")} aria-label="Run details">
      <div class="drawer-head">
        <input class="input title-input" value=${name} onInput=${(e) => setName(e.target.value)}
          onBlur=${() => name !== run.name && name.trim() && patch({ name: name.trim() })}
          onKeyDown=${(e) => e.key === "Enter" && e.target.blur()} />
        <${StatusBadge} status=${run.status} />
        <button class="icon-btn" title="Close" onClick=${onClose}>×</button>
      </div>
      ${err && html`<div class="query-error">${err}</div>`}
      <div class="drawer-sec">
        <div class="tagrow">
          ${run.tags.map((t) => html`<span class="tag">${t}<button title="Remove tag" onClick=${() => patch({ remove_tags: [t] })}>×</button></span>`)}
          <input class="input tag-input" placeholder="+ tag" value=${tag} onInput=${(e) => setTag(e.target.value)}
            onKeyDown=${(e) => { if (e.key === "Enter" && tag.trim()) { patch({ add_tags: [tag.trim()] }); setTag(""); } }} />
        </div>
        <textarea class="input notes" placeholder="Notes…" value=${notes} onInput=${(e) => setNotes(e.target.value)}
          onBlur=${() => notes !== (run.notes || "") && patch({ notes })}></textarea>
      </div>
      <div class="tabs">
        ${["config", "summary", "code", "meta"].map((t) => html`<button class=${"tab" + (tab === t ? " on" : "")} onClick=${() => setTab(t)}>${t}</button>`)}
      </div>
      <div class=${"drawer-body" + (tab === "code" ? " flush" : "")}>
        ${tab === "config" && html`<table class="kv">${Object.entries(run.config).map(([k, v]) => html`<tr><th>${k}</th><td>${fmtValue(v)}</td></tr>`)}
          ${Object.keys(run.config).length === 0 && html`<tr><td class="muted">No config</td></tr>`}</table>`}
        ${tab === "summary" && html`<table class="kv">${Object.entries(run.summary).map(([k, s]) => html`<tr><th>${k}</th><td>${fmtNum(s.value)} <span class="muted">@ ${fmtNum(s.step)}</span></td></tr>`)}</table>`}
        ${tab === "code" && html`<${CodeTab} run=${run} />`}
        ${tab === "meta" && html`<table class="kv">${metaRows.map(([k, v]) => html`<tr><th>${k}</th><td class="mono">${v}</td></tr>`)}</table>`}
      </div>
      <div class="drawer-foot">
        <button class="btn danger" onClick=${del}>Delete run</button>
      </div>
    </aside>`;
}
