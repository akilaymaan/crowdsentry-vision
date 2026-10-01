import { riskColors } from '../lib/risk'
import './RiskBadge.css'

/** Coloured pill showing a risk level, and optionally its 0-100 score. */
export default function RiskBadge({ level, score, size = 'md' }) {
  const label = level ?? 'NO DATA'
  const { fg, bg } = riskColors(level)

  return (
    <span
      className={`risk-badge risk-badge--${size} ${level === 'CRITICAL' ? 'risk-badge--critical' : ''}`}
      style={{ '--badge-fg': fg, '--badge-bg': bg }}
      title={score != null ? `Risk score ${score.toFixed(1)} / 100` : 'No risk score yet'}
    >
      <span className="risk-badge__dot" aria-hidden="true" />
      {label}
      {score != null && <span className="risk-badge__score mono">{Math.round(score)}</span>}
    </span>
  )
}
