/** Thin REST client for the CrowdSentry API. */

// In Vite development, use the same-origin proxy configured in vite.config.js. This
// avoids localhost/127.0.0.1 mismatches and keeps REST and WebSocket traffic on one
// origin. Production builds default to same-origin too: the deployed topology is the
// frontend served behind a reverse proxy that forwards /api and /ws to the backend,
// so an absolute API origin is only needed when the two are genuinely split across
// hosts -- set VITE_API_BASE_URL for that case.
const configuredBase = import.meta.env.VITE_API_BASE_URL
const BASE = (configuredBase ?? '').replace(/\/$/, '')
const DISPLAY_BASE = BASE || (import.meta.env.DEV ? 'the Vite dev proxy' : 'this origin')

// Shared API key, baked in at build time. Sent as a header on REST and as a query
// parameter on the WebSocket handshake (the browser WS API cannot set headers).
// Empty means the backend has auth disabled -- the local-dev configuration.
const API_KEY = import.meta.env.VITE_API_KEY ?? ''


/** ws:// or wss:// URL for the live feed, derived from the API base. */
export function liveSocketUrl() {
  const url = new URL('/ws/live', BASE || window.location.origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  if (API_KEY) url.searchParams.set('api_key', API_KEY)
  return url.toString()
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
        ...options.headers,
      },
    })
  } catch (cause) {
    // Deliberate aborts (effect cleanup, StrictMode's double mount) are not outages;
    // let callers recognise them by name instead of showing "cannot reach the API".
    if (cause?.name === 'AbortError') throw cause
    // fetch only rejects on network-level failures, which for a dashboard almost
    // always means the API process is not running. Say that, rather than "Failed to
    // fetch".
    throw new ApiError(`Cannot reach the API at ${DISPLAY_BASE}. Is the backend running?`, 0)
  }

  if (!response.ok) {
    if (response.status === 401) {
      throw new ApiError(
        "The API rejected this dashboard's credentials. Check VITE_API_KEY matches the backend's API_KEY.",
        401,
      )
    }
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail
    } catch {
      /* response had no JSON body; the status line is all we have */
    }
    throw new ApiError(detail, response.status)
  }

  return response.json()
}

export const api = {
  summary: (signal) => request('/api/dashboard/summary', { signal }),

  cameras: (signal) => request('/api/cameras', { signal }),

  camera: (id, signal) => request(`/api/cameras/${id}`, { signal }),

  cameraHistory: (id, { from, to, limit = 500 } = {}, signal) => {
    const params = new URLSearchParams()
    if (from) params.set('from', from)
    if (to) params.set('to', to)
    params.set('limit', String(limit))
    return request(`/api/cameras/${id}/history?${params}`, { signal })
  },

  openAlerts: (signal) => api.alerts({ acknowledged: false }, signal),

  acknowledgeAlert: (id) => request(`/api/alerts/${id}/acknowledge`, { method: 'POST' }),

  alerts: ({ acknowledged, cameraId, limit = 50, offset = 0 } = {}, signal) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (acknowledged != null) params.set('acknowledged', String(acknowledged))
    if (cameraId != null) params.set('camera_id', String(cameraId))
    return request(`/api/alerts?${params}`, { signal })
  },

  health: (signal) => request('/health', { signal }),
  readiness: (signal) => request('/health/ready', { signal }),
  processorStatus: (signal) => request('/api/processor/status', { signal }),
}
