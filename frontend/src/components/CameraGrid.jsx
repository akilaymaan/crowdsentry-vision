import RiskBadge from './RiskBadge'
import { formatAgo, formatDensity, isFresh, riskColors } from '../lib/risk'
import './CameraGrid.css'

export default function CameraGrid({ cameras, selectedId, onSelect, freshnessSeconds }) {
  if (!cameras.length) {
    return (
      <section className="panel">
        <div className="panel-title">Cameras</div>
        <p className="empty">
          No cameras registered. Register cameras in the cameras table to begin
          monitoring.
        </p>
      </section>
    )
  }

  return (
    <section className="panel">
      <div className="panel-title">
        <span>Cameras</span>
        <span className="muted">{cameras.length}</span>
      </div>
      <div className="camera-grid">
        {cameras.map((camera, index) => (
          <CameraCard
            key={camera.id}
            index={index}
            camera={camera}
            selected={camera.id === selectedId}
            onSelect={onSelect}
            freshnessSeconds={freshnessSeconds}
          />
        ))}
      </div>
    </section>
  )
}

function CameraCard({ camera, index, selected, onSelect, freshnessSeconds }) {
  const observation = camera.latest_observation
  const risk = camera.latest_risk
  const live = isFresh(observation?.timestamp, freshnessSeconds)

  // A camera that stopped reporting keeps its last reading in the database, but showing
  // that as the current state would misrepresent the venue. Grey it out instead.
  const level = live ? risk?.risk_level : null
  const { fg } = riskColors(level)

  return (
    <button
      type="button"
      className={[
        'camera-card',
        selected && 'camera-card--selected',
        !live && 'camera-card--stale',
        level === 'CRITICAL' && live && 'camera-card--critical',
      ]
        .filter(Boolean)
        .join(' ')}
      style={{ '--card-accent': fg, '--i': index }}
      onClick={() => onSelect(camera.id)}
      aria-pressed={selected}
    >
      <span className="camera-card__accent" />

      <div className="camera-card__head">
        <div className="camera-card__ident">
          <span className="camera-card__name">{camera.name}</span>
          <span className="camera-card__location">{camera.location_name}</span>
        </div>
        <RiskBadge level={level} score={live ? risk?.risk_score : null} size="sm" />
      </div>

      <div className="camera-card__metrics">
        <Metric
          label="People"
          value={live && observation ? observation.person_count : '—'}
          emphasis
        />
        <Metric
          label="Density"
          value={live && observation ? formatDensity(observation.density) : '—'}
        />
        <Metric label="Area" value={`${Math.round(camera.area_sq_meters)} m²`} />
      </div>

      <div className="camera-card__foot">
        {!camera.is_active ? (
          <span className="camera-card__flag">inactive</span>
        ) : !camera.stream_configured ? (
          <span className="camera-card__flag">no feed configured</span>
        ) : live ? (
          <span className="camera-card__live">
            <span className="camera-card__live-dot" />
            updated {formatAgo(observation?.timestamp)}
          </span>
        ) : (
          <span className="camera-card__flag camera-card__flag--warn">
            no data · last {formatAgo(observation?.timestamp)}
          </span>
        )}
      </div>
    </button>
  )
}

function Metric({ label, value, emphasis = false }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className={`metric__value mono ${emphasis ? 'metric__value--lg' : ''}`}>
        {value}
      </span>
    </div>
  )
}
