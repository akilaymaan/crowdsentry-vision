import { memo } from 'react'
import { CameraOff, Crosshair, Radio, ShieldCheck } from 'lucide-react'
import RiskBadge from './RiskBadge'
import { formatAgo, isFresh, riskColors } from '../lib/risk'

export function Status({ label, state = 'unknown' }) {
  return (
    <span className={`status status--${state}`} role="status">
      <span aria-hidden="true" />
      {label}
    </span>
  )
}

export function SectionHeader({ title, subtitle, action }) {
  return (
    <div className="section-head">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}

export function MetricCard({ label, value, unit, hint, icon: Icon = Crosshair }) {
  return (
    <div className="metric-card panel">
      <div className="metric-card__label">
        {label}
        <Icon size={16} aria-hidden="true" />
      </div>
      <div className="metric-card__number mono">
        {value ?? '—'}
        {unit && <span>{unit}</span>}
      </div>
      <p>{hint}</p>
    </div>
  )
}

export function EmptyState({ title, children, icon: Icon = Radio }) {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">
        <Icon size={25} aria-hidden="true" />
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  )
}

export function Skeleton({ label = 'Synchronizing telemetry' }) {
  return (
    <div className="skeleton" role="status" aria-label={label}>
      <span />
      <span />
      <span />
      <p>{label}</p>
    </div>
  )
}

function cameraState(camera, worker, freshnessSeconds = 120) {
  if (!camera?.is_active) return 'Inactive'
  if (!camera.stream_configured) return 'Not configured'
  if (worker && worker.state !== 'running')
    return (
      { reconnecting: 'Reconnecting', failed: 'Offline', stopped: 'Stopped', starting: 'Starting' }[
        worker.state
      ] ?? 'Unknown'
    )
  return isFresh(camera.latest_observation?.timestamp, freshnessSeconds)
    ? 'Reporting'
    : 'Waiting for source'
}

export const CameraFeed = memo(function CameraFeed({
  camera,
  worker,
  freshnessSeconds = 120,
  onInspect,
}) {
  const state = cameraState(camera, worker, freshnessSeconds)
  const fresh = isFresh(camera?.latest_observation?.timestamp, freshnessSeconds)
  return (
    <section className="panel camera-feed" aria-label="Camera monitoring panel">
      <div className="panel-title">
        <span>
          <Crosshair size={15} aria-hidden="true" /> Camera feed
        </span>
        <span className="mono">
          {camera ? `SOURCE ${String(camera.id).padStart(2, '0')}` : 'NO SOURCE'}
        </span>
      </div>
      <div className="camera-feed__viewport">
        <div className="camera-feed__corners" aria-hidden="true" />
        <div className="camera-feed__top">
          <span>{camera?.name ?? 'No camera selected'}</span>
          <Status label={state} state={fresh && state === 'Reporting' ? 'connected' : 'unknown'} />
        </div>
        <div className="camera-feed__message">
          <CameraOff size={38} strokeWidth={1} aria-hidden="true" />
          <h3>{fresh ? 'Telemetry is live' : 'Waiting for video source'}</h3>
          <p>
            {fresh
              ? 'Crowd measurements are arriving from the processor.'
              : 'No recent measurements received from this camera.'}
          </p>
          <span>Browser video is not provided by this API.</span>
        </div>
        <div className="camera-feed__bottom">
          <span>
            <ShieldCheck size={13} aria-hidden="true" /> Person detection only · No identity
            recognition
          </span>
          <span className="mono">
            {camera?.latest_observation
              ? formatAgo(camera.latest_observation.timestamp)
              : 'AWAITING SIGNAL'}
          </span>
        </div>
      </div>
      {fresh && (
        <div className="camera-feed__telemetry">
          <span>
            People <strong className="mono">{camera.latest_observation.person_count}</strong>
          </span>
          <span>
            Density{' '}
            <strong className="mono">{camera.latest_observation.density.toFixed(2)}/m²</strong>
          </span>
          <RiskBadge
            level={
              isFresh(camera.latest_risk?.timestamp, freshnessSeconds)
                ? camera.latest_risk.risk_level
                : null
            }
          />
        </div>
      )}
      <div className="camera-feed__foot">
        <div>
          <strong>{camera?.location_name || 'Monitoring source'}</strong>
          <span>
            {camera ? `Camera ID ${camera.id}` : 'Configure a camera to begin monitoring'}
          </span>
        </div>
        {camera && onInspect && (
          <button className="button" onClick={() => onInspect(camera.id)}>
            Inspect camera <span aria-hidden="true">↗</span>
          </button>
        )}
      </div>
    </section>
  )
})

export function RiskGauge({ risk, previous, fresh = true }) {
  const score = fresh && Number.isFinite(risk?.risk_score) ? risk.risk_score : null
  const level = score == null ? null : risk.risk_level
  const { fg } = riskColors(level)
  const delta = score != null && Number.isFinite(previous) ? score - previous : null
  return (
    <section className="panel risk-panel">
      <div className="panel-title">
        <span>Current risk</span>
        <span className="mono">0–100</span>
      </div>
      <div className="risk-gauge" style={{ '--risk-color': fg }}>
        <svg
          viewBox="0 0 220 150"
          role="img"
          aria-label={
            score == null ? 'No current risk score' : `Risk score ${score.toFixed(1)} of 100`
          }
        >
          <path className="risk-gauge__track" d="M 30 120 A 80 80 0 1 1 190 120" pathLength="100" />
          <path
            className="risk-gauge__value"
            d="M 30 120 A 80 80 0 1 1 190 120"
            pathLength="100"
            strokeDasharray={`${score == null ? 0 : Math.max(0, Math.min(100, score))} 100`}
          />
        </svg>
        <div className="risk-gauge__reading">
          <strong className="mono">{score == null ? '—' : Math.round(score)}</strong>
          <RiskBadge level={level} />
        </div>
      </div>
      <p className="risk-panel__trend mono">
        {delta == null
          ? 'Awaiting comparable windows'
          : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} from previous window`}
      </p>
      <div className="risk-scale">
        {['LOW', 'MODERATE', 'HIGH', 'CRITICAL'].map((entry) => (
          <span key={entry} style={{ color: riskColors(entry).fg }}>
            <i style={{ background: riskColors(entry).fg }} />
            {entry}
          </span>
        ))}
      </div>
      <p className="risk-panel__note">Crowd-risk estimate, not a validated incident prediction.</p>
    </section>
  )
}
