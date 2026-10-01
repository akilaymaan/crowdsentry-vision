import { useCallback, useState } from 'react'
import { api } from '../api/client'
import { useResource } from '../hooks/useResource'
import { EmptyState, Skeleton } from './ConsoleUI'
import RiskBadge from './RiskBadge'
import { formatTime } from '../lib/risk'

export default function AlertCenter({
  cameras,
  onSelectCamera,
  onAcknowledged,
  cameraId: fixedCameraId,
}) {
  const [filter, setFilter] = useState('all')
  const [cameraId, setCameraId] = useState('')
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(new Set())
  const [actionError, setActionError] = useState(null)
  const loader = useCallback(
    (signal) =>
      api.alerts(
        {
          acknowledged: filter === 'all' ? undefined : filter === 'acknowledged',
          cameraId: fixedCameraId ?? (cameraId ? Number(cameraId) : undefined),
          offset,
        },
        signal,
      ),
    [filter, cameraId, fixedCameraId, offset],
  )
  const { data, loading, error, reload } = useResource(loader)

  async function acknowledge(id) {
    setBusy((current) => new Set(current).add(id))
    setActionError(null)
    try {
      await api.acknowledgeAlert(id)
      onAcknowledged?.(id)
      reload()
    } catch {
      setActionError('Acknowledgement failed. The alert remains open; please try again.')
    } finally {
      setBusy((current) => {
        const next = new Set(current)
        next.delete(id)
        return next
      })
    }
  }

  return (
    <section className="panel alert-center">
      <div className="alert-controls">
        <div className="tabs" aria-label="Acknowledgement filter">
          {[
            ['all', 'All'],
            ['open', 'Unacknowledged'],
            ['acknowledged', 'Acknowledged'],
          ].map(([value, label]) => (
            <button
              key={value}
              aria-pressed={filter === value}
              onClick={() => {
                setFilter(value)
                setOffset(0)
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {fixedCameraId == null && (
          <label className="select-label">
            Camera
            <select
              value={cameraId}
              onChange={(event) => {
                setCameraId(event.target.value)
                setOffset(0)
              }}
            >
              <option value="">All cameras</option>
              {cameras.map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="button" onClick={reload}>
          Refresh alerts
        </button>
      </div>
      {actionError && (
        <p className="inline-error" role="alert">
          {actionError}
        </p>
      )}
      {loading ? (
        <Skeleton label="Retrieving alert records" />
      ) : error ? (
        <div className="resource-error" role="alert">
          <p>Alert records are unavailable.</p>
          <button className="button" onClick={reload}>
            Retry alerts
          </button>
        </div>
      ) : !data?.alerts?.length ? (
        <EmptyState title="No alerts">
          No crowd-risk alerts match this filter. This does not establish that unobserved areas are
          safe.
        </EmptyState>
      ) : (
        <div className="alert-records">
          {data.alerts.map((alert) => (
            <article key={alert.id} className="alert-record">
              <div>
                <RiskBadge
                  level={
                    alert.risk_level ??
                    (alert.message?.startsWith('CRITICAL') ? 'CRITICAL' : 'HIGH')
                  }
                  score={alert.risk_score}
                />
                <time dateTime={alert.timestamp}>
                  {new Date(alert.timestamp).toLocaleDateString()} · {formatTime(alert.timestamp)}
                </time>
              </div>
              <button className="text-button" onClick={() => onSelectCamera(alert.camera_id)}>
                {alert.camera_name}
              </button>
              <p>{alert.message}</p>
              <div>
                <span className="eyebrow">
                  {alert.acknowledged ? 'ACKNOWLEDGED' : 'REQUIRES REVIEW'}
                </span>
                {!alert.acknowledged && (
                  <button
                    className="button"
                    disabled={busy.has(alert.id)}
                    onClick={() => acknowledge(alert.id)}
                  >
                    {busy.has(alert.id) ? 'Acknowledging…' : 'Acknowledge'}
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
      <div className="chart-foot">
        <span>
          {data ? `${data.total} matching alerts · newest first` : 'Authoritative alert records'}
        </span>
        <div className="pagination">
          <button
            className="button"
            disabled={offset === 0 || loading}
            onClick={() => setOffset(Math.max(0, offset - 50))}
          >
            Previous
          </button>
          <button
            className="button"
            disabled={loading || !data || offset + 50 >= data.total}
            onClick={() => setOffset(offset + 50)}
          >
            Next
          </button>
        </div>
      </div>
    </section>
  )
}
