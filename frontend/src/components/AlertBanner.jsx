import { formatAgo, riskColors } from '../lib/risk'
import './AlertBanner.css'

/**
 * Unacknowledged alerts, newest first.
 *
 * Severity comes from the alert's own risk_level column. Rows written before that
 * column existed fall back to inferring it from the message text, where the backend
 * writes "CRITICAL crowd risk..." for critical and "High crowd risk..." otherwise.
 */
function levelOf(alert) {
  if (alert.risk_level === 'CRITICAL' || alert.risk_level === 'HIGH') {
    return alert.risk_level
  }
  return alert.message?.startsWith('CRITICAL') ? 'CRITICAL' : 'HIGH'
}

export default function AlertBanner({ alerts, onAcknowledge, acknowledging, onSelectCamera }) {
  if (!alerts.length) {
    return (
      <section className="alerts alerts--clear panel">
        <span className="alerts__clear-dot" />
        <span>No active alerts — no unacknowledged crowd-risk events. Non-reporting sources remain unobserved.</span>
      </section>
    )
  }

  const critical = alerts.filter((alert) => levelOf(alert) === 'CRITICAL').length

  return (
    <section className="alerts panel" aria-live="polite">
      <div className="alerts__head">
        <div className="alerts__title">
          <span className="alerts__pulse" />
          {alerts.length} active alert{alerts.length === 1 ? '' : 's'}
          {critical > 0 && <span className="alerts__critical">{critical} critical</span>}
        </div>
      </div>

      <ul className="alerts__list">
        {alerts.map((alert, index) => {
          const level = levelOf(alert)
          const { fg, bg } = riskColors(level)
          const busy = acknowledging.has(alert.id)

          return (
            <li
              key={alert.id}
              className={`alert ${level === 'CRITICAL' ? 'alert--critical' : ''}`}
              style={{ '--alert-fg': fg, '--alert-bg': bg, '--ai': index }}
            >
              <span className="alert__bar" />
              <div className="alert__body">
                <div className="alert__meta">
                  <button
                    type="button"
                    className="alert__camera"
                    onClick={() => onSelectCamera?.(alert.camera_id)}
                  >
                    {alert.camera_name}
                  </button>
                  <span className="alert__level">{level}</span>
                  <span className="alert__time">{formatAgo(alert.timestamp)}</span>
                </div>
                <p className="alert__message">{alert.message}</p>
              </div>
              <button
                type="button"
                className="alert__ack"
                onClick={() => onAcknowledge(alert.id)}
                disabled={busy}
              >
                {busy ? 'Acknowledging…' : 'Acknowledge'}
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
