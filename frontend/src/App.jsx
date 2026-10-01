import { Suspense, lazy, useCallback, useState } from 'react'
import { Activity, Crosshair, RefreshCw, Users, Video } from 'lucide-react'

import AlertBanner from './components/AlertBanner'
import AlertCenter from './components/AlertCenter'
import CameraGrid from './components/CameraGrid'
import CrowdMap from './components/CrowdMap'
import ConsoleShell from './components/ConsoleShell'
import { usePage } from './hooks/usePage'
import { EmptyState, MetricCard, Skeleton, Status } from './components/ConsoleUI'
import Monitoring from './components/Monitoring'
import SystemStatus from './components/SystemStatus'
import { useDashboard } from './hooks/useDashboard'
import { useSystemStatus } from './hooks/useSystemStatus'
import { formatAgo, isFresh, riskColors } from './lib/risk'
import './App.css'

// Recharts is ~400 kB and only the detail view needs it. Loading it lazily keeps it out
// of the initial bundle, so the dashboard itself paints without waiting for a charting
// library most viewers never open.
const CameraDetail = lazy(() => import('./components/CameraDetail'))
const Analytics = lazy(() => import('./components/Analytics'))

const TITLES = {
  overview: ['Command center', 'Real-time crowd intelligence across your monitored sources.'],
  cameras: ['Camera network', 'Source availability, measured crowd conditions, and live risk.'],
  analytics: [
    'Crowd intelligence',
    'Explore measured patterns across historical observation windows.',
  ],
  alerts: ['Alert center', 'Review crowd-risk events and coordinate your response.'],
  system: ['System status', 'Infrastructure readiness and realtime processing diagnostics.'],
}

export default function App() {
  const {
    cameras,
    alerts,
    summary,
    derived,
    loading,
    error,
    feedStatus,
    lastEventAt,
    acknowledging,
    acknowledge,
    reload,
  } = useDashboard()
  const system = useSystemStatus()
  const [page, navigate] = usePage()
  const [selectedId, setSelectedId] = useState(null)
  const [focusedId, setFocusedId] = useState(null)
  const select = useCallback((id) => setSelectedId(id), [])
  const clearSelection = useCallback(() => setSelectedId(null), [])
  const freshnessSeconds = summary?.freshness_seconds ?? 120
  const camera = cameras.find((entry) => entry.id === focusedId) ?? cameras[0]
  const worker = system.data?.workers?.find((entry) => entry.camera_id === camera?.id)
  const freshCameras = cameras.filter(
    (entry) => entry.is_active && isFresh(entry.latest_observation?.timestamp, freshnessSeconds),
  )
  const highest = freshCameras
    .filter((entry) => isFresh(entry.latest_risk?.timestamp, freshnessSeconds))
    .reduce(
      (risk, entry) =>
        !risk || entry.latest_risk.risk_score > risk.risk_score ? entry.latest_risk : risk,
      null,
    )
  const meanDensity = freshCameras.length
    ? freshCameras.reduce((sum, entry) => sum + entry.latest_observation.density, 0) /
      freshCameras.length
    : null
  const [title, subtitle] = TITLES[page]

  return (
    <ConsoleShell
      page={page}
      navigate={navigate}
      feedStatus={feedStatus}
      lastEventAt={lastEventAt}
      alertCount={alerts.length}
      system={system.data}
    >
      <div className="page-head">
        <div>
          <div className="eyebrow">CROWDSENTRY / OPERATIONS</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <div className="page-head__tools">
          <Status
            label={
              derived.reporting > 0 && feedStatus === 'live'
                ? 'LIVE TELEMETRY'
                : 'AWAITING TELEMETRY'
            }
            state={derived.reporting > 0 && feedStatus === 'live' ? 'connected' : 'unknown'}
          />
          <button className="button icon-button" aria-label="Refresh dashboard" onClick={reload}>
            <RefreshCw size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
      {loading ? (
        <div className="boot-panels">
          <Skeleton label="Connecting to CrowdSentry" />
          <Skeleton label="Synchronizing monitoring sources" />
        </div>
      ) : error ? (
        <section className="panel resource-error" role="alert">
          <h2>Cannot reach the backend</h2>
          <p>
            The API is currently unavailable or rejected this request. Network failures retry
            automatically.
          </p>
          <button className="button" onClick={reload}>
            Retry
          </button>
        </section>
      ) : (
        <>
          {page === 'overview' && (
            <>
              <div className="kpi-grid">
                <MetricCard
                  label="Active cameras"
                  value={derived.activeCameras}
                  icon={Video}
                  hint={`${derived.reporting}/${derived.totalCameras} reporting · ${derived.stale} not reporting`}
                />
                <MetricCard
                  label="People detected"
                  value={derived.reporting ? derived.totalPeople : null}
                  icon={Users}
                  hint="Across fresh reporting sources"
                />
                <MetricCard
                  label="Highest risk"
                  value={highest ? Math.round(highest.risk_score) : null}
                  unit="/100"
                  icon={Activity}
                  hint={highest?.risk_level ?? 'No current risk classification'}
                />
                <MetricCard
                  label="Average density"
                  value={meanDensity == null ? null : meanDensity.toFixed(2)}
                  unit="/m²"
                  icon={Crosshair}
                  hint="Mean across reporting cameras"
                />
              </div>
              <div className="overview-toolbar">
                <div>
                  <span className="eyebrow">SOURCE MONITOR</span>
                  <label className="sr-only" htmlFor="overview-source">
                    Monitoring source
                  </label>
                  <select
                    id="overview-source"
                    value={camera?.id ?? ''}
                    onChange={(event) => setFocusedId(Number(event.target.value))}
                  >
                    {!cameras.length && <option value="">No cameras</option>}
                    {cameras.map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {entry.name}
                      </option>
                    ))}
                  </select>
                </div>
                <span className="muted">
                  Last event · {lastEventAt ? formatAgo(lastEventAt) : 'Awaiting signal'}
                </span>
              </div>
              <Monitoring
                camera={camera}
                worker={worker}
                freshnessSeconds={freshnessSeconds}
                onInspect={select}
              />
              <div className="overview-secondary">
                <Suspense
                  fallback={
                    <section className="panel">
                      <Skeleton label="Preparing timeline" />
                    </section>
                  }
                >
                  <Analytics cameraId={camera?.id} latestRisk={camera?.latest_risk} compact />
                </Suspense>
                <section className="panel risk-distribution">
                  <div className="panel-title">
                    Network risk state <span>FRESH CAMERAS</span>
                  </div>
                  {Object.entries(derived.counts).map(([level, count]) => (
                    <div key={level}>
                      <span style={{ color: riskColors(level).fg }}>{level}</span>
                      <div>
                        <i
                          style={{
                            width: `${derived.reporting ? (count / derived.reporting) * 100 : 0}%`,
                            background: riskColors(level).fg,
                          }}
                        />
                      </div>
                      <strong className="mono">{count}</strong>
                    </div>
                  ))}
                  <div className="network-density">
                    <span>Peak density</span>
                    <strong className="mono">
                      {freshCameras.length
                        ? Math.max(
                            ...freshCameras.map((entry) => entry.latest_observation.density),
                          ).toFixed(2)
                        : '—'}
                      <small> /m²</small>
                    </strong>
                  </div>
                  <p className="panel-note">
                    {derived.stale} active source{derived.stale === 1 ? '' : 's'} without fresh
                    measurements.
                  </p>
                </section>
              </div>
              <div className="section-label">
                <h2>Live alerts</h2>
                <button className="text-button" onClick={() => navigate('alerts')}>
                  Open alert center ↗
                </button>
              </div>
              <AlertBanner
                alerts={alerts}
                onAcknowledge={acknowledge}
                acknowledging={acknowledging}
                onSelectCamera={select}
              />
              <div className="app__main">
                <CameraGrid
                  cameras={cameras}
                  selectedId={selectedId}
                  onSelect={select}
                  freshnessSeconds={freshnessSeconds}
                />
                <CrowdMap
                  cameras={cameras}
                  selectedId={selectedId}
                  onSelect={select}
                  freshnessSeconds={freshnessSeconds}
                />
              </div>
            </>
          )}
          {page === 'cameras' && (
            <>
              <div className="inventory-summary">
                <span>{cameras.length} registered sources</span>
                <span>{derived.reporting} reporting</span>
                <span>{derived.stale} not reporting</span>
              </div>
              <CameraGrid
                cameras={cameras}
                selectedId={selectedId}
                onSelect={select}
                freshnessSeconds={freshnessSeconds}
              />
            </>
          )}
          {page === 'analytics' && (
            <>
              {!cameras.length ? (
                <section className="panel">
                  <EmptyState title="No monitoring sources">
                    Register a camera to build historical crowd intelligence.
                  </EmptyState>
                </section>
              ) : (
                <>
                  <div className="overview-toolbar">
                    <label className="select-label">
                      Analyze camera
                      <select
                        value={camera?.id ?? ''}
                        onChange={(event) => setFocusedId(Number(event.target.value))}
                      >
                        {cameras.map((entry) => (
                          <option key={entry.id} value={entry.id}>
                            {entry.name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <span className="muted">Measured history · no synthetic data</span>
                  </div>
                  <Suspense fallback={<Skeleton label="Preparing analytics" />}>
                    <Analytics
                      key={camera?.id}
                      cameraId={camera?.id}
                      latestRisk={camera?.latest_risk}
                    />
                  </Suspense>
                </>
              )}
            </>
          )}
          {page === 'alerts' && (
            <AlertCenter cameras={cameras} onSelectCamera={select} onAcknowledged={reload} />
          )}
          {page === 'system' && (
            <SystemStatus system={system} feedStatus={feedStatus} lastEventAt={lastEventAt} />
          )}
        </>
      )}
      {selectedId != null && (
        <Suspense fallback={<div className="detail-loading">Preparing camera intelligence…</div>}>
          <CameraDetail
            key={selectedId}
            cameraId={selectedId}
            cameras={cameras}
            onClose={clearSelection}
            onSelectCamera={select}
            freshnessSeconds={freshnessSeconds}
            worker={system.data?.workers?.find((entry) => entry.camera_id === selectedId)}
            onAcknowledged={reload}
            onCameraChanged={() => {
              reload()
              system.reload()
            }}
          />
        </Suspense>
      )}
    </ConsoleShell>
  )
}
