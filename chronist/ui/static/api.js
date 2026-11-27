// API access: token handling, JSON helpers, SSE.
const TOKEN_KEY = "chronist.token";

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} }

// ?token=... in the URL (sharing a read-only link) -> remember and strip it.
(function grabTokenFromUrl() {
  const u = new URL(location.href);
  const t = u.searchParams.get("token");
  if (t) {
    safeSet(TOKEN_KEY, t);
    u.searchParams.delete("token");
    history.replaceState(null, "", u.pathname + u.search + u.hash);
  }
})();

export function getToken() { return safeGet(TOKEN_KEY); }
export function setToken(t) { safeSet(TOKEN_KEY, t || null); }

export class AuthError extends Error {}
export const CODE_HIDDEN = "code not shared";
export class ApiError extends Error {
  constructor(msg, status, pos) { super(msg); this.status = status; this.pos = pos; }
}

let onAuthNeeded = () => {};
export function setAuthHandler(fn) { onAuthNeeded = fn; }

async function request(method, path, body, signal) {
  const headers = {};
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  let data;
  try { data = await r.json(); } catch { data = {}; }
  if (r.status === 403 && data.error === CODE_HIDDEN) throw new ApiError(CODE_HIDDEN, 403);
  if (r.status === 401 || r.status === 403) {
    onAuthNeeded(r.status);
    throw new AuthError(r.status === 401 ? "token required" : "read-only token");
  }
  if (!r.ok) throw new ApiError(data.error || r.statusText, r.status, data.pos);
  return data;
}

async function requestText(path) {
  const headers = {};
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  const r = await fetch(path, { headers });
  if (r.status === 403) {
    let e = {}; try { e = await r.clone().json(); } catch {}
    if (e.error === CODE_HIDDEN) throw new ApiError(CODE_HIDDEN, 403);
  }
  if (r.status === 401 || r.status === 403) { onAuthNeeded(r.status); throw new AuthError("token required"); }
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch {} throw new ApiError(m, r.status); }
  return r.text();
}

function qs(params) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") p.set(k, v);
  const s = p.toString();
  return s ? "?" + s : "";
}

export const api = {
  projects: () => request("GET", "/api/projects"),
  runs: (project, q, signal) => request("GET", "/api/runs" + qs({ project, q }), undefined, signal),
  run: (id) => request("GET", `/api/runs/${id}`),
  keys: (project) => request("GET", "/api/keys" + qs({ project })),
  suggest: (project, text, signal) => request("GET", "/api/suggest" + qs({ project, text }), undefined, signal),
  series: (body, signal) => request("POST", "/api/series", body, signal),
  histRender: (body) => request("POST", "/api/hist/render", body),
  files: (id) => request("GET", `/api/runs/${id}/files`),
  setAliases: (id, name, body) => request("PATCH", `/api/runs/${id}/files/${encodeURIComponent(name).replace(/%2F/g, "/")}/aliases`, body),
  deleteFile: (id, name, vid) => request("DELETE", `/api/runs/${id}/files/${encodeURIComponent(name).replace(/%2F/g, "/")}/${vid}`),
  fileUrl: (vid) => `/api/files/${vid}/download` + qs({ token: getToken() }),
  patch: (id, body) => request("PATCH", `/api/runs/${id}`, body),
  del: (id) => request("DELETE", `/api/runs/${id}`),
  code: (id) => request("GET", `/api/runs/${id}/code`),
  whoami: () => request("GET", "/api/whoami"),
  login: async (password) => {
    const r = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ password, label: "browser: " + navigator.userAgent.slice(0, 40) }) });
    let d = {};
    try { d = await r.json(); } catch {}
    if (!r.ok) throw new ApiError(d.error || r.statusText, r.status);
    return d.token;
  },
  logout: () => request("POST", "/api/logout", {}),
  tokens: () => request("GET", "/api/tokens"),
  createToken: (body) => request("POST", "/api/tokens", body),
  revokeToken: (id) => request("DELETE", `/api/tokens/${id}`),
  codeFile: (id, path) => requestText(`/api/runs/${id}/code/file` + qs({ path })),
  exportUrl: (project, q, keys, format) => "/api/export" + qs({ project, q, keys: (keys || []).join(","), format, token: getToken() }),
};

// One EventSource for the app; listeners get arrays of events.
export function openStream(project, onEvents) {
  const t = getToken();
  const es = new EventSource("/api/stream" + qs({ project, token: t }));
  es.onmessage = (m) => {
    try { onEvents(JSON.parse(m.data)); } catch (e) { console.warn("bad SSE message", e); }
  };
  return () => es.close();
}
