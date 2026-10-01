import { useCallback } from 'react'
import { api } from '../api/client'
import { useResource } from '../hooks/useResource'
import { formatAgo, isFresh } from '../lib/risk'
import { CameraFeed, RiskGauge, SectionHeader } from './ConsoleUI'

export default function Monitoring({ camera, worker, freshnessSeconds, onInspect }) {
  const cameraId = camera?.id
  const loader = useCallback(
    (signal) => (cameraId != null ? api.camera(cameraId, signal) : Promise.resolve(null)),
    [cameraId],
  )
  const { data: detail, error } = useResource(loader)
  const observation = detail?.latest_observation
  const fresh = isFresh(camera?.latest_risk?.timestamp, freshnessSeconds)
  const telemetryFresh = isFresh(observation?.timestamp, freshnessSeconds)
  const metrics = [
    ['Flow speed', observation?.mean_flow_speed, 'm/s', 2],
    ['Stop ratio', observation?.stop_ratio == null ? null : observation.stop_ratio * 100, '%', 0],
    ['Flow variance', observation?.flow_direction_variance, '', 3],
    ['Optical entropy', observation?.optical_flow_entropy, '', 2],
    ['Baseline deviation', observation?.historical_deviation, 'σ', 2],
    ['Density change', observation?.density_rate_of_change, '/m²/s', 3],
  ]
  return (
    <div className="monitoring-layout">
      <CameraFeed
        camera={camera}
        worker={worker}
        freshnessSeconds={freshnessSeconds}
        onInspect={onInspect}
      />
      <RiskGauge risk={camera?.latest_risk} previous={camera?.previous_risk_score} fresh={fresh} />
      <section className="panel telemetry-panel">
        <SectionHeader
          title="Crowd dynamics"
          subtitle={
            error
              ? 'Feature telemetry unavailable'
              : observation
                ? `Feature window · ${formatAgo(observation.timestamp)}`
                : 'Waiting for a measured feature window'
          }
          action={<span className="eyebrow">MEASURED FEATURES</span>}
        />
        <div className="telemetry-grid">
          {metrics.map(([label, value, unit, digits]) => (
            <div key={label}>
              <span>{label}</span>
              <strong className="mono">
                {telemetryFresh && Number.isFinite(value) ? value.toFixed(digits) : '—'}
                <small>{unit}</small>
              </strong>
            </div>
          ))}
        </div>
        <p className="panel-note">
          {camera?.pixels_per_meter == null
            ? 'Uncalibrated source: speed in m/s is unavailable.'
            : 'Speed uses the camera’s ground-plane calibration.'}{' '}
          Features are measured, not inferred from the video placeholder.
        </p>
      </section>
    </div>
  )
}
