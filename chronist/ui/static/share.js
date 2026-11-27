// Share dialog: create read-only links (optionally including code), list and revoke them.
import { html, useState, useEffect } from "./vendor/htm-preact.module.js";
import { api } from "./api.js";
import { Modal } from "./widgets.js";
import { fmtAgo } from "./util.js";

function shareUrl(token) {
  const u = new URL(location.href);
  u.searchParams.set("token", token);
  return u.toString();  // keeps the hash: the link opens this exact view
}

export function ShareModal({ me, onClose }) {
  const [label, setLabel] = useState("");
  const [code, setCode] = useState(false);
  const [link, setLink] = useState(null);
  const [list, setList] = useState([]);
  const [err, setErr] = useState(null);
  const [copied, setCopied] = useState(false);

  const load = () => api.tokens().then((r) => setList(r.tokens)).catch((e) => setErr(e.message));
  useEffect(() => { if (me.auth) load(); }, []);

  async function create() {
    setErr(null);
    try {
      const r = await api.createToken({ label: label.trim(), code });
      setLink({ url: shareUrl(r.token), code: r.code });
      setCopied(false);
      load();
    } catch (e) { setErr(e.message); }
  }
  async function copy() {
    try { await navigator.clipboard.writeText(link.url); setCopied(true); }
    catch { const el = document.getElementById("share-link"); el.select(); }
  }
  async function revoke(id) {
    try { await api.revokeToken(id); load(); } catch (e) { setErr(e.message); }
  }

  if (!me.auth) {
    return html`<${Modal} title="Share" onClose=${onClose}>
      <p>This server runs without authentication (local mode), so anyone who can reach it sees everything, including code.</p>
      <p class="muted">To share read-only views, run it with tokens, e.g. <code>chronist serve --token &lt;write-token&gt;</code>
        behind your reverse proxy, then create links here.</p>
      <div class="modal-actions"><button class="btn" onClick=${onClose}>OK</button></div>
    <//>`;
  }

  return html`<${Modal} title="Share a read-only link" onClose=${onClose}>
    <p class="muted small">The link opens this view. Recipients can browse runs and plots but can't change anything.</p>
    <label class="field"><span>Label</span>
      <input class="input" placeholder="e.g. for the reading group" value=${label} onInput=${(e) => setLabel(e.target.value)} />
    </label>
    <label class="check">
      <input type="checkbox" checked=${code} onChange=${(e) => setCode(e.target.checked)} />
      <span><b>Share code as well</b><br /><span class="muted small">Every file in the runs' code snapshots,
        plus environment and package lists. Off by default.</span></span>
    </label>
    ${err && html`<div class="query-error">${err}</div>`}
    ${link ? html`
      <div class="share-link">
        <input id="share-link" class="input mono" readonly value=${link.url} onFocus=${(e) => e.target.select()} />
        <button class="btn" onClick=${copy}>${copied ? "Copied" : "Copy"}</button>
      </div>
      <p class="muted small">${link.code ? "Includes code." : "Code is not included."} Anyone with this link has access until you revoke it.</p>`
    : html`<div class="modal-actions"><button class="btn ghost" onClick=${onClose}>Cancel</button>
        <button class="btn" onClick=${create}>Create link</button></div>`}
    ${list.length > 0 && html`
      <h4 class="share-h">Active links</h4>
      <div class="share-list">
        ${list.map((t) => html`
          <div class="share-row">
            <span class="share-label">${t.label || html`<span class="muted">unlabeled</span>`}</span>
            ${t.code ? html`<span class="chip">code</span>` : html`<span class="chip muted">no code</span>`}
            <span class="muted small">${t.created ? fmtAgo(t.created) : "from CLI"}</span>
            ${t.persistent ? html`<button class="btn small ghost danger" onClick=${() => revoke(t.id)}>Revoke</button>`
                           : html`<span class="muted small" title="Given on the command line">cli</span>`}
          </div>`)}
      </div>`}
    ${link && html`<div class="modal-actions"><button class="btn" onClick=${onClose}>Done</button></div>`}
  <//>`;
}
