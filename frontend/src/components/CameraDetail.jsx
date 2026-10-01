import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import RiskBadge from './RiskBadge'
import { api } from '../api/client'
import { formatAgo, formatDensity, formatTime, riskColors } from '../lib/risk'
import './CameraDetail.css'

const RANGES = [
  { label: '15m', minutes: 15 },
  { label: '1h', minutes: 60 },
  { label: '6h', minutes: 360 },
  { label: '24h', minutes: 1440 },
]

// Fruin LOS E: the density at which movement degrades to shuffling. Drawn on the chart
// so a reading has a reference point rather than being a bare number.
const LOS_E_DENSITY = 1.08

export default function CameraDetail({ cameraId, cameras, onClose, onSelectCamera }) {
  const [range, setRange] = useState(RANGES[1])
  const [history, setHistory] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const camera = cameras.find((entry) => entry.id === cameraId)

  const load = useCallback(
    async (signal) => {
      setLoading(true)
      setError(null)
      try {
        const to = new Date()
        const from = new Date(to.getTime() - range.minutes * 60_000)
        const [historyData, detailData] = await Promise.all([
          api.cameraHistory(
            cameraId,
            { from: from.toISOString(), to: to.toISOString(), limit: 2000 },
            signal,
          ),
          api.camera(cameraId, signal),
        ])
        setHistory(historyData)
        setDetail(detailData)
      } catch (cause) {
        if (cause.name !== 'AbortError') setError(cause)
      } finally {
        setLoading(false)
      }
    },
    [cameraId, range],
  )

  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])

  // Escape closes, which is what anyone expects of an overlay.
  useEffect(() => {
    const onKey = (event) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  /**
   * Merge the two series onto a shared time axis.
   *
   * The API returns observations and risk scores separately because a window can produce
   * one without the other. Recharts needs a single row per x-value, so they are keyed by
   * timestamp here and missing values left undefined — which Recharts renders as a gap
   * rather than as zero, and a gap is the honest depiction of "no reading".
   */
  const series = useMemo(() => {
    if (!history) return []
    const byTime = new Map()

    for (const observation of history.observations) {
      const key = new Date(observation.timestamp).getTime()
      byTime.set(key, {
        time: key,
        density: observation.density,
        people: observation.person_count,
      })
    }
    for (const score of history.risk_scores) {
      const key = new Date(score.timestamp).getTime()
      const row = byTime.get(key) ?? { time: key }
      row.risk = score.risk_score
      row.level = score.risk_level
      byTime.set(key, row)
    }

    return [...byTime.values()].sort((a, b) => a.time - b.time)
  }, [history])

  const latestRisk = camera?.latest_risk ?? detail?.latest_risk
  const observation = detail?.latest_observation

  return (
    <div className="detail-backdrop" onClick={onClose} role="presentation">
      <aside
        className="detail"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`${camera?.name ?? 'Camera'} detail`}
      >
        <header className="detail__head">
          <div>
            <div className="detail__title-row">
              <h2 className="detail__title">{camera?.name ?? detail?.name ?? 'Camera'}</h2>
              <RiskBadge
                level={latestRisk?.risk_level}
                score={latestRisk?.risk_score}
                size="lg"
              />
            </div>
            <p className="detail__location">
              {camera?.location_name ?? detail?.location_name}
              {latestRisk && (
                <span className="muted"> · scored {formatAgo(latestRisk.timestamp)}</span>
              )}
            </p>
          </div>
          <button type="button" className="detail__close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <div className="detail__stats">
          <DetailStat label="People" value={observation?.person_count ?? '—'} />
          <DetailStat
            label="Density"
            value={observation ? formatDensity(observation.density) : '—'}
          />
          <DetailStat
            label="Flow speed"
            value={
              observation?.mean_flow_speed != null
                ? `${observation.mean_flow_speed.toFixed(2)} m/s`
                : 'uncalibrated'
            }
          />
          <DetailStat
            label="Stopped"
            value={
              observation?.stop_ratio != null
                ? `${Math.round(observation.stop_ratio * 100)}%`
                : '—'
            }
          />
          <DetailStat
            label="vs baseline"
            value={
              observation?.historical_deviation != null
                ? `${observation.historical_deviation > 0 ? '+' : ''}${observation.historical_deviation.toFixed(1)}σ`
                : 'no baseline'
            }
          />
          <DetailStat label="Open alerts" value={detail?.unacknowledged_alerts ?? 0} />
        </div>

        <div className="detail__chart-head">
          <span className="detail__chart-title">Density &amp; risk over time</span>
          <div className="range-picker">
            {RANGES.map((option) => (
              <button
                key={option.label}
                type="button"
                className={`range-picker__btn ${
                  option.label === range.label ? 'range-picker__btn--on' : ''
                }`}
                onClick={() => setRange(option)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="detail__chart">
          {loading && <p className="empty">Loading history…</p>}
          {error && <p className="empty">Could not load history: {error.message}</p>}
          {!loading && !error && series.length === 0 && (
            <p className="empty">
              No data in the last {range.label}. Start the processor for this camera.
            </p>
          )}
          {!loading && !error && series.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={series} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
                <defs>
                  <linearGradient id="densityFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="time"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  scale="time"
                  tickFormatter={formatTime}
                  stroke="var(--text-faint)"
                  fontSize={11}
                  minTickGap={40}
                />
                <YAxis
                  yAxisId="density"
                  stroke="var(--accent)"
                  fontSize={11}
                  width={52}
                  label={{
                    value: 'people/m²',
                    angle: -90,
                    position: 'insideLeft',
                    fill: 'var(--text-faint)',
                    fontSize: 10,
                    dy: 32,
                  }}
                />
                <YAxis
                  yAxisId="risk"
                  orientation="right"
                  domain={[0, 100]}
                  stroke="var(--moderate)"
                  fontSize={11}
                  width={38}
                />

                <ReferenceLine
                  yAxisId="density"
                  y={LOS_E_DENSITY}
                  stroke="var(--high)"
                  strokeDasharray="4 4"
                  strokeOpacity={0.7}
                  label={{
                    value: 'Fruin LOS E',
                    fill: 'var(--high)',
                    fontSize: 10,
                    position: 'insideTopRight',
                  }}
                />

                <Tooltip content={<ChartTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
                  iconType="plainline"
                />

                <Area
                  yAxisId="density"
                  type="monotone"
                  dataKey="density"
                  name="Density"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  fill="url(#densityFill)"
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="risk"
                  name="Risk score"
                  stroke="var(--moderate)"
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>

        {history?.truncated && (
          <p className="detail__note">
            Showing the most recent readings only — the range holds more than the row limit.
          </p>
        )}

        <footer className="detail__foot">
          <span className="muted">
            {camera?.pixels_per_meter
              ? `${camera.pixels_per_meter} px/m calibrated`
              : 'Not calibrated — speeds unavailable'}
          </span>
          <NeighbourLinks
            cameras={cameras}
            currentId={cameraId}
            onSelectCamera={onSelectCamera}
          />
        </footer>
      </aside>
    </div>
  )
}

function DetailStat({ label, value }) {
  return (
    <div className="detail-stat">
      <span className="detail-stat__label">{label}</span>
      <span className="detail-stat__value mono">{value}</span>
    </div>
  )
}

function NeighbourLinks({ cameras, currentId, onSelectCamera }) {
  const others = cameras.filter((camera) => camera.id !== currentId)
  if (!others.length) return null

  return (
    <div className="detail__neighbours">
      {others.map((camera) => {
        const { fg } = riskColors(camera.latest_risk?.risk_level)
        return (
          <button
            key={camera.id}
            type="button"
            className="detail__neighbour"
            style={{ '--n': fg }}
            onClick={() => onSelectCamera(camera.id)}
          >
            <span className="detail__neighbour-dot" />
            {camera.name}
          </button>
        )
      })}
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload

  return (
    <div className="chart-tip">
      <div className="chart-tip__time mono">{formatTime(label)}</div>
      {row.density != null && (
        <div className="chart-tip__row">
          <span style={{ color: 'var(--accent)' }}>Density</span>
          <span className="mono">{row.density.toFixed(3)} /m²</span>
        </div>
      )}
      {row.people != null && (
        <div className="chart-tip__row">
          <span className="muted">People</span>
          <span className="mono">{row.people}</span>
        </div>
      )}
      {row.risk != null && (
        <div className="chart-tip__row">
          <span style={{ color: 'var(--moderate)' }}>Risk</span>
          <span className="mono">
            {row.risk.toFixed(1)} {row.level && `· ${row.level}`}
          </span>
        </div>
      )}
    </div>
  )
}
