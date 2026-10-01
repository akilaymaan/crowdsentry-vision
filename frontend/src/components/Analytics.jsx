import { memo, useCallback, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { useResource } from '../hooks/useResource'
import { formatNumber, formatTime, riskColors, RISK_LEVELS } from '../lib/risk'
import { EmptyState, MetricCard, RangePicker, SectionHeader, Skeleton } from './ConsoleUI'

const RANGES = [
  { label: '1H', hours: 1 },
  { label: '6H', hours: 6 },
  { label: '24H', hours: 24 },
  { label: '7D', hours: 168 },
]

export const HistoryChart = memo(function HistoryChart({
  rows,
  field,
  label,
  unit = '',
  risk = false,
}) {
  const available = rows.filter((row) => Number.isFinite(row[field]))
  if (!available.length)
    return (
      <EmptyState title="No historical data">
        Continue monitoring to generate {label.toLowerCase()} intelligence.
      </EmptyState>
    )
  const latest = available.at(-1)
  const color = risk ? riskColors(latest.risk_level).fg : 'var(--chart-line)'
  const Chart = risk ? LineChart : AreaChart
  const Plot = risk ? Line : Area
  return (
    <div
      className="history-chart"
      role="img"
      aria-label={`${label} chart, ${available.length} measured windows`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <Chart data={rows} margin={{ top: 12, right: 12, bottom: 8, left: -18 }} accessibilityLayer>
          <CartesianGrid stroke="var(--border)" vertical={false} strokeDasharray="2 5" />
          <XAxis
            dataKey="time"
            type="number"
            domain={['dataMin', 'dataMax']}
            scale="time"
            tickFormatter={(value) =>
              new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }
            stroke="var(--text-faint)"
            tickLine={false}
            axisLine={false}
            minTickGap={45}
            fontSize={10}
          />
          <YAxis
            domain={risk ? [0, 100] : ['auto', 'auto']}
            stroke="var(--text-faint)"
            tickLine={false}
            axisLine={false}
            fontSize={10}
          />
          {risk &&
            RISK_LEVELS.map((level, index) => (
              <ReferenceArea
                key={level}
                y1={index * 25}
                y2={(index + 1) * 25}
                fill={riskColors(level).fg}
                fillOpacity={0.025}
              />
            ))}
          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value) => [`${formatNumber(value, 2)} ${unit}`, label]}
            contentStyle={{
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              fontSize: 12,
            }}
          />
          <Plot
            type="monotone"
            dataKey={field}
            stroke={color}
            fill={color}
            fillOpacity={0.07}
            strokeWidth={2}
            dot={available.length === 1 ? { r: 3 } : false}
            activeDot={{ r: 4 }}
            connectNulls={false}
            isAnimationActive={false}
          />
          {risk && (
            <ReferenceDot
              x={latest.time}
              y={latest[field]}
              r={4}
              fill={color}
              stroke="var(--surface)"
            />
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  )
})

export default function Analytics({ cameraId, compact = false, latestRisk }) {
  const [range, setRange] = useState(RANGES[0])
  const loader = useCallback(
    async (signal) => {
      if (cameraId == null) return { observations: [], risk_scores: [] }
      const to = new Date()
      const from = new Date(to.getTime() - range.hours * 3_600_000)
      const data = await api.cameraHistory(
        cameraId,
        { from: from.toISOString(), to: to.toISOString(), limit: 2000 },
        signal,
      )
      return { ...data, requestedStart: from.getTime() }
    },
    [cameraId, range],
  )
  const { data: history, loading, error, reload } = useResource(loader)
  const scores = useMemo(() => {
    const rows = (history?.risk_scores ?? []).map((row) => ({
      ...row,
      time: new Date(row.timestamp).getTime(),
    }))
    const time = new Date(latestRisk?.timestamp).getTime()
    if (
      Number.isFinite(time) &&
      time >= history?.requestedStart &&
      Number.isFinite(latestRisk?.risk_score) &&
      !rows.some((row) => row.time === time)
    )
      rows.push({ ...latestRisk, time })
    return rows.sort((left, right) => left.time - right.time)
  }, [history, latestRisk])
  const observations = useMemo(
    () =>
      (history?.observations ?? []).map((row) => ({
        ...row,
        time: new Date(row.timestamp).getTime(),
        stopped_percent: row.stop_ratio == null ? null : row.stop_ratio * 100,
      })),
    [history],
  )
  const peak = useMemo(
    () =>
      observations.reduce(
        (highest, row) => (highest == null || row.density > highest.density ? row : highest),
        null,
      ),
    [observations],
  )
  const distribution = useMemo(
    () =>
      RISK_LEVELS.map((level) => ({
        level,
        count: scores.filter((row) => row.risk_level === level).length,
      })),
    [scores],
  )

  return (
    <div className="analytics-view">
      <section className="panel">
        <SectionHeader
          title="Risk timeline"
          subtitle="Measured model scores · shaded quartiles are guides, not classification thresholds"
          action={
            <RangePicker options={RANGES} value={range} onChange={setRange} />
          }
        />
        {loading ? (
          <Skeleton label="Retrieving risk history" />
        ) : error ? (
          <div className="resource-error" role="alert">
            <p>Unable to retrieve history.</p>
            <button className="button" onClick={reload}>
              Retry history
            </button>
          </div>
        ) : (
          <HistoryChart rows={scores} field="risk_score" label="Risk score" risk />
        )}
        <div className="chart-foot">
          <span>{scores.length} scored windows</span>
          <span>
            {history?.truncated
              ? 'Recent 2,000 readings only · not the full range'
              : 'No interpolation across missing readings'}
          </span>
        </div>
      </section>
      {!compact && (
        <>
          <div className="analytics-metrics">
            <MetricCard
              label="Scored windows"
              value={loading || error ? null : scores.length}
              hint="In the returned history range"
            />
            <MetricCard
              label="Peak density"
              value={peak ? peak.density.toFixed(2) : null}
              unit="/m²"
              hint={peak ? `Measured at ${formatTime(peak.timestamp)}` : 'No measured density yet'}
            />
            <MetricCard
              label="Elevated windows"
              value={
                loading || error
                  ? null
                  : scores.filter((row) => ['HIGH', 'CRITICAL'].includes(row.risk_level)).length
              }
              hint="HIGH and CRITICAL classifications"
            />
          </div>
          {!loading && !error && !observations.length && !scores.length && (
            <section className="panel">
              <EmptyState title="Insufficient data">
                Continue monitoring to build historical crowd intelligence.
              </EmptyState>
            </section>
          )}
          <div className="dynamics-charts">
            {[
              ['density', 'Crowd density', 'people/m²'],
              ['mean_flow_speed', 'Flow speed', 'm/s'],
              ['stopped_percent', 'Stopped people', '%'],
              ['historical_deviation', 'Baseline comparison', 'σ'],
            ].map(([field, label, unit]) => (
              <section className="panel" key={field}>
                <SectionHeader
                  title={label}
                  subtitle={
                    field === 'mean_flow_speed'
                      ? 'Calibrated sources only'
                      : 'Observed feature windows'
                  }
                />
                {loading ? (
                  <Skeleton />
                ) : error ? (
                  <p className="empty">History unavailable.</p>
                ) : (
                  <HistoryChart rows={observations} field={field} label={label} unit={unit} />
                )}
              </section>
            ))}
          </div>
          <section className="panel">
            <SectionHeader
              title="Risk distribution"
              subtitle="Classification counts in the returned history · not incident counts"
            />
            <div className="distribution">
              {distribution.map(({ level, count }) => (
                <div key={level}>
                  <span style={{ color: riskColors(level).fg }}>{level}</span>
                  <div>
                    <i
                      style={{
                        width: `${scores.length ? (count / scores.length) * 100 : 0}%`,
                        background: riskColors(level).fg,
                      }}
                    />
                  </div>
                  <strong className="mono">{loading || error ? '—' : count}</strong>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
