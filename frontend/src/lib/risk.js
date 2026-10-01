/** Risk-level vocabulary and formatting shared across the dashboard. */

export const RISK_LEVELS = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']

/** Levels that warrant an operator alert — mirrors ALERTING_LEVELS on the backend. */
export const ALERTING_LEVELS = ['HIGH', 'CRITICAL']

const SEVERITY = { LOW: 0, MODERATE: 1, HIGH: 2, CRITICAL: 3 }

/** CSS custom-property names per level, so colours stay defined in one place. */
export const RISK_VARS = {
  LOW: { fg: 'var(--low)', bg: 'var(--low-bg)' },
  MODERATE: { fg: 'var(--moderate)', bg: 'var(--moderate-bg)' },
  HIGH: { fg: 'var(--high)', bg: 'var(--high-bg)' },
  CRITICAL: { fg: 'var(--critical)', bg: 'var(--critical-bg)' },
}

const UNKNOWN = { fg: 'var(--unknown)', bg: 'var(--unknown-bg)' }

export function riskColors(level) {
  return RISK_VARS[level] ?? UNKNOWN
}

export function severity(level) {
  return SEVERITY[level] ?? -1
}

export function isAlerting(level) {
  return ALERTING_LEVELS.includes(level)
}

/** Highest level present, or null. */
export function highestLevel(levels) {
  let highest = null
  for (const level of levels) {
    if (level && severity(level) > severity(highest)) highest = level
  }
  return highest
}

/* Formatting ------------------------------------------------------------- */

export function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function formatDensity(value) {
  if (value === null || value === undefined) return '—'
  return `${Number(value).toFixed(2)}/m²`
}

export function formatTime(value) {
  if (!value) return '—'
  return new Date(value).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** "just now" / "12s ago" / "4m ago" — for judging whether a feed is live. */
export function formatAgo(value, now = Date.now()) {
  if (!value) return 'never'
  const seconds = Math.max(0, Math.round((now - new Date(value).getTime()) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

/**
 * A camera counts as live only if it reported recently.
 *
 * Matches the backend's freshness rule: a camera whose feed died an hour ago still has
 * a "latest" reading, and showing it as current would quietly misreport the venue.
 */
export function isFresh(timestamp, freshnessSeconds = 120, now = Date.now()) {
  if (!timestamp) return false
  return now - new Date(timestamp).getTime() <= freshnessSeconds * 1000
}
