import { Activity, Database, Radio, Server } from 'lucide-react'
import { formatAgo, formatNumber } from '../lib/risk'
import { EmptyState, SectionHeader, Skeleton, Status } from './ConsoleUI'

export default function SystemStatus({ system, feedStatus, lastEventAt }) {
  const data = system.data
  const states = [
    ['API', data?.api, Server],
    ['MongoDB readiness', data?.database, Database],
    ['Realtime processor', data?.processor, Activity],
    ['WebSocket', feedStatus === 'live' ? 'Connected' : feedStatus, Radio],
  ]
  return (
    <div className="system-view">
      <div className="system-services">
        {states.map(([label, value, Icon]) => (
          <section className="panel service-card" key={label}>
            <Icon size={21} strokeWidth={1.5} aria-hidden="true" />
            <h2>{label}</h2>
            <Status
              label={value ?? 'Checking'}
              state={
                ['Operational', 'Connected', 'Running'].includes(value) ? 'connected' : 'unknown'
              }
            />
          </section>
        ))}
      </div>
      <section className="panel">
        <SectionHeader
          title="Camera workers"
          subtitle={
            data
              ? `Last checked ${formatAgo(data.checkedAt)} · checks every 15 seconds`
              : 'Checking the realtime processor'
          }
          action={
            <button className="button" onClick={system.reload}>
              Recheck system
            </button>
          }
        />
        {system.loading ? (
          <Skeleton label="Checking worker status" />
        ) : data?.workers == null ? (
          <EmptyState title="Worker status unavailable">
            The processor endpoint has not returned a valid status.
          </EmptyState>
        ) : !data.workers.length ? (
          <EmptyState title="No camera workers">
            No realtime camera workers are currently registered.
          </EmptyState>
        ) : (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">Realtime camera worker measurements</caption>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>State</th>
                  <th>Frames</th>
                  <th>Windows</th>
                  <th>Alerts raised</th>
                  <th>Last window</th>
                </tr>
              </thead>
              <tbody>
                {data.workers.map((worker) => (
                  <tr key={worker.camera_id}>
                    <td>
                      <strong>{worker.name}</strong>
                      <small>ID {worker.camera_id}</small>
                    </td>
                    <td>
                      <Status
                        label={worker.state ?? 'Unknown'}
                        state={worker.state === 'running' ? 'connected' : 'unknown'}
                      />
                    </td>
                    <td className="mono">{formatNumber(worker.frames_processed)}</td>
                    <td className="mono">{formatNumber(worker.windows_processed)}</td>
                    <td className="mono">{formatNumber(worker.alerts_raised)}</td>
                    <td>{formatAgo(worker.last_window_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <div className="system-notes">
        <section className="panel">
          <SectionHeader
            title="Processing architecture"
            subtitle="Configured project pipeline · not a model-health probe"
          />
          <ol className="pipeline-list">
            {[
              'Video source ingestion',
              'YOLO person detection + ByteTrack',
              'Crowd feature aggregation',
              'XGBoost risk classification',
              'MongoDB persistence + WebSocket broadcast',
            ].map((step, index) => (
              <li key={step}>
                <span className="mono">0{index + 1}</span>
                {step}
              </li>
            ))}
          </ol>
        </section>
        <section className="panel">
          <SectionHeader title="Operational boundaries" />
          <div className="system-copy">
            <p>
              Database connectivity is derived from the readiness probe, not inferred from the
              WebSocket.
            </p>
            <p>
              A running processor does not imply a working video source. Check each worker state.
            </p>
            <p>
              Risk classification is trained on simulated pedestrian data and is not a validated
              incident predictor.
            </p>
            <p>
              Last WebSocket event:{' '}
              <strong>{lastEventAt ? formatAgo(lastEventAt) : 'No event received'}</strong>
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
