import { Suspense, lazy, useCallback, useState } from 'react'

import AlertBanner from './components/AlertBanner'
import CameraGrid from './components/CameraGrid'
import CrowdMap from './components/CrowdMap'
import Header from './components/Header'
import { API_BASE } from './api/client'
import { useDashboard } from './hooks/useDashboard'
import './App.css'

// Recharts is ~400 kB and only the detail view needs it. Loading it lazily keeps it out
// of the initial bundle, so the dashboard itself paints without waiting for a charting
// library most viewers never open.
const CameraDetail = lazy(() => import('./components/CameraDetail'))

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

  const [selectedId, setSelectedId] = useState(null)

  const select = useCallback((id) => setSelectedId(id), [])
  const clearSelection = useCallback(() => setSelectedId(null), [])

  const freshnessSeconds = summary?.freshness_seconds ?? 120

  if (loading) {
    return (
      <div className="boot">
        <div className="boot__spinner" />
        <p>Connecting to CrowdSentry…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="boot boot--error">
        <h2>Cannot reach the backend</h2>
        <p className="muted">{error.message}</p>
        <pre className="boot__hint">
{`cd backend
docker compose up -d          # database
uvicorn app.main:app --reload # API${API_BASE ? ` at ${API_BASE}` : ' (same origin)'}`}
        </pre>
        <button type="button" className="boot__retry" onClick={reload}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="app">
      <Header
        derived={derived}
        summary={summary}
        feedStatus={feedStatus}
        lastEventAt={lastEventAt}
      />

      <AlertBanner
        alerts={alerts}
        onAcknowledge={acknowledge}
        acknowledging={acknowledging}
        onSelectCamera={select}
      />

      <div className="app__main">
        <CrowdMap
          cameras={cameras}
          selectedId={selectedId}
          onSelect={select}
          freshnessSeconds={freshnessSeconds}
        />
        <CameraGrid
          cameras={cameras}
          selectedId={selectedId}
          onSelect={select}
          freshnessSeconds={freshnessSeconds}
        />
      </div>

      {selectedId != null && (
        <Suspense fallback={<div className="detail-loading">Loading chart…</div>}>
          <CameraDetail
            cameraId={selectedId}
            cameras={cameras}
            onClose={clearSelection}
            onSelectCamera={select}
          />
        </Suspense>
      )}
    </div>
  )
}
