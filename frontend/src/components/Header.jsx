import { useEffect, useRef, useState } from 'react'

import { RISK_LEVELS, formatAgo, formatNumber, riskColors } from '../lib/risk'
import './Header.css'

const FEED_LABELS = {
  live: 'Live',
  connecting: 'Connecting',
  reconnecting: 'Reconnecting',
  offline: 'Offline',
  unauthorized: 'Unauthorized',
}

const REDUCED_MOTION =
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/**
 * Ease a live number toward its target, so a headline figure ticks smoothly when a
 * websocket event changes it. The first finite value lands instantly — animating the
 * initial load would just look like the number was wrong a moment ago.
 */
function useAnimatedNumber(target, duration = 650) {
  const [display, setDisplay] = useState(target)
  const displayRef = useRef(target)

  useEffect(() => {
    displayRef.current = display
  }, [display])

  useEffect(() => {
    if (!Number.isFinite(target) || REDUCED_MOTION) return
    if (!Number.isFinite(displayRef.current)) {
      // After a non-numeric gap the rendered value already is the target; syncing
      // state asynchronously just re-arms the tween for the next change.
      const raf = requestAnimationFrame(() => setDisplay(target))
      return () => cancelAnimationFrame(raf)
    }
    const from = displayRef.current
    if (from === target) return

    let raf = 0
    const started = performance.now()
    const step = (now) => {
      const t = Math.min(1, (now - started) / duration)
      const value = from + (target - from) * (1 - (1 - t) ** 3)
      displayRef.current = value
      setDisplay(value)
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])

  if (REDUCED_MOTION || !Number.isFinite(target)) return target
  return Number.isFinite(display) ? display : target
}

/** Top bar: headline counts, per-level camera breakdown, and feed status. */
export default function Header({ derived, summary, feedStatus, lastEventAt }) {
  const { counts, totalPeople, reporting, stale, totalCameras, highest } = derived
  const highestColors = riskColors(highest)
  const animatedPeople = useAnimatedNumber(totalPeople)
  const animatedPeak = useAnimatedNumber(summary?.peak_density)

  return (
    <header className="header">
      <div className="header__brand">
        <div className="header__mark" style={{ '--mark': highestColors.fg }}>
          <span className="header__mark-ring" />
          <span className="header__mark-core" />
        </div>
        <div>
          <h1 className="header__title">CrowdSentry</h1>
          <p className="header__subtitle">Spatio-temporal crowd risk monitoring</p>
        </div>
      </div>

      <div className="header__stats">
        <Stat
          label="People tracked"
          value={formatNumber(animatedPeople)}
          hint={`across ${reporting} reporting camera${reporting === 1 ? '' : 's'}`}
        />
        <Stat
          label="Cameras"
          value={`${reporting}/${totalCameras}`}
          hint={stale > 0 ? `${stale} not reporting` : 'all reporting'}
          warn={stale > 0}
        />
        <Stat
          label="Peak density"
          value={animatedPeak != null ? animatedPeak.toFixed(2) : '—'}
          hint="people/m²"
        />
      </div>

      <div className="header__levels" role="group" aria-label="Cameras by risk level">
        {RISK_LEVELS.map((level) => {
          const { fg, bg } = riskColors(level)
          const count = counts[level] ?? 0
          return (
            <div
              key={level}
              className={[
                'level-chip',
                count > 0 && 'level-chip--active',
                count > 0 && level === 'CRITICAL' && 'level-chip--critical',
              ]
                .filter(Boolean)
                .join(' ')}
              style={{ '--chip-fg': fg, '--chip-bg': bg }}
            >
              <span className="level-chip__count mono">{count}</span>
              <span className="level-chip__label">{level}</span>
            </div>
          )
        })}
      </div>

      <div className={`feed feed--${feedStatus}`}>
        <span className="feed__dot" />
        <div className="feed__text">
          <span className="feed__status">{FEED_LABELS[feedStatus] ?? feedStatus}</span>
          <span className="feed__hint">
            {feedStatus === 'live'
              ? lastEventAt
                ? `last update ${formatAgo(lastEventAt)}`
                : 'awaiting first window'
              : 'no live feed'}
          </span>
        </div>
      </div>
    </header>
  )
}

function Stat({ label, value, hint, warn = false }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value mono">{value}</span>
      <span className={`stat__hint ${warn ? 'stat__hint--warn' : ''}`}>{hint}</span>
    </div>
  )
}
