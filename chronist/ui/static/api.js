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
  if (r.status === 401 || r.status === 403) {
    onAuthNeeded(r.status);
    throw new AuthError(r.status === 401 ? "token required" : "read-only token");
  }
  let data;
  try { data = await r.json(); } catch { data = {}; }
  if (!r.ok) throw new ApiError(data.error || r.statusText, r.status, data.pos);
  return data;
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
  patch: (id, body) => request("PATCH", `/api/runs/${id}`, body),
  del: (id) => request("DELETE", `/api/runs/${id}`),
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
