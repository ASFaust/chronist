// "Files" tab: versions grouped by name, aliases, metadata, download.
import { html, useState, useEffect } from "./vendor/htm-preact.module.js";
import { api } from "./api.js";
import { fmtAgo, fmtNum, fmtValue } from "./util.js";

function fmtBytes(n) {
  if (n == null) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 ** 2) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 ** 3) return (n / 1024 ** 2).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

export function FilesTab({ run, canEdit }) {
  const [rows, setRows] = useState(null);
  const [err, setErr] = useState(null);
  const load = () => api.files(run.id).then((r) => { setRows(r.files); setErr(null); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, [run.id]);

  async function act(fn) { try { await fn(); await load(); } catch (e) { setErr(e.message); } }
  const setAlias = (name, vid) => {
    const a = prompt("Alias for this version (e.g. best). It moves here if another version has it:");
    if (a && a.trim()) act(() => api.setAliases(run.id, name, { set: { [a.trim()]: vid } }));
  };
  const removeAlias = (name, alias) => act(() => api.setAliases(run.id, name, { remove: [alias] }));
  const del = (name, v) => {
    if (confirm(`Delete version ${v.version_id} of "${name}"${v.step != null ? ` (step ${v.step})` : ""}? Its storage is freed by the next GC.`))
      act(() => api.deleteFile(run.id, name, v.version_id));
  };

  if (err) return html`<div class="query-error pad">${err}</div>`;
  if (!rows) return html`<div class="muted pad">Loading…</div>`;
  if (!rows.length) return html`<div class="muted pad">No files. Attach some with <code>run.save_file(path, step=…)</code> or <code>run.save_object(obj, name=…)</code>.</div>`;
  const groups = new Map();
  for (const r of rows) { if (!groups.has(r.name)) groups.set(r.name, []); groups.get(r.name).push(r); }
  return html`
    <div class="files-tab">
      ${[...groups.entries()].map(([name, vs]) => html`
        <div class="fgroup">
          <div class="fgroup-head"><span class="mono">${name}</span>
            <span class="muted small">${vs.length} version${vs.length > 1 ? "s" : ""} · ${fmtBytes(vs.reduce((a, v) => a + v.size, 0))}</span></div>
          <table class="ftable">
            <thead><tr><th>step</th><th>size</th><th>kind</th><th>aliases</th><th>metadata</th><th>saved</th><th></th></tr></thead>
            <tbody>
              ${vs.slice().reverse().map((v) => html`
                <tr>
                  <td class="num">${v.step == null ? "–" : fmtNum(v.step)}</td>
                  <td class="num">${fmtBytes(v.size)}</td>
                  <td>${v.kind}</td>
                  <td>${v.aliases.map((a) => html`<span class=${"tag" + (a === "latest" ? " auto" : "")}>${a}${canEdit && a !== "latest" &&
                        html`<button title="Remove alias" onClick=${() => removeAlias(name, a)}>×</button>`}</span>`)}</td>
                  <td class="fmeta" title=${JSON.stringify(v.metadata)}>${Object.entries(v.metadata || {}).slice(0, 3).map(([k, x]) => html`<span><span class="muted">${k}</span> ${fmtValue(x)}</span>`)}</td>
                  <td class="muted">${fmtAgo(v.created)}</td>
                  <td class="factions">
                    <a class="btn small ghost" href=${api.fileUrl(v.version_id)} download title=${v.sha256}>Download</a>
                    ${canEdit && html`<button class="btn small ghost" onClick=${() => setAlias(name, v.version_id)}>Alias…</button>
                      <button class="icon-btn" title="Delete version" onClick=${() => del(name, v)}>×</button>`}
                  </td>
                </tr>`)}
            </tbody>
          </table>
        </div>`)}
    </div>`;
}
