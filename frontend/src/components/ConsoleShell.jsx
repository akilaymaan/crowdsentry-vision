import { useEffect, useState } from 'react'
import {
  Activity,
  Bell,
  ChartNoAxesCombined,
  ChevronRight,
  Grid2X2,
  Radio,
  Shield,
  Video,
} from 'lucide-react'
import { formatAgo } from '../lib/risk'
import { Status } from './ConsoleUI'

const PAGES = [
  ['overview', 'Overview', Grid2X2],
  ['cameras', 'Cameras', Video],
  ['analytics', 'Analytics', ChartNoAxesCombined],
  ['alerts', 'Alerts', Bell],
  ['system', 'System', Activity],
]

export default function ConsoleShell({
  page,
  navigate,
  feedStatus,
  lastEventAt,
  alertCount,
  system,
  children,
}) {
  const [clock, setClock] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  const connected = feedStatus === 'live'
  const socketLabel =
    {
      live: 'Connected',
      connecting: 'Connecting',
      reconnecting: 'Reconnecting',
      offline: 'Reconnecting',
      unauthorized: 'Unauthorized',
    }[feedStatus] ?? 'Unknown'
  const active = system?.workers?.filter((worker) => worker.state === 'running').length
  const title = PAGES.find(([id]) => id === page)?.[1]

  return (
    <div className="console-shell">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(event) => {
          event.preventDefault()
          document.getElementById('main-content')?.focus()
        }}
      >
        Skip to content
      </a>
      <aside className="sidebar" aria-label="Application sidebar">
        <a className="brand" href="#overview" onClick={() => navigate('overview')}>
          <span className="brand__mark">
            <Shield size={23} strokeWidth={1.5} aria-hidden="true" />
          </span>
          <span>
            <strong>CrowdSentry</strong>
            <small>CROWD INTELLIGENCE</small>
          </span>
        </a>
        <div className="sidebar__workspace">
          <span className="workspace-symbol">
            <Radio size={15} aria-hidden="true" />
          </span>
          <div>
            Operations workspace<small>Real-time monitoring</small>
          </div>
          <ChevronRight size={14} aria-hidden="true" />
        </div>
        <span className="sidebar__label">WORKSPACE</span>
        <nav className="navigation" aria-label="Main navigation">
          {PAGES.map(([id, label, Icon]) => (
            <a
              key={id}
              href={`#${id}`}
              aria-label={label}
              aria-current={page === id ? 'page' : undefined}
              onClick={() => navigate(id)}
            >
              <Icon size={18} strokeWidth={1.6} aria-hidden="true" />
              <span>{label}</span>
              {id === 'alerts' && alertCount > 0 && <small aria-hidden="true">{alertCount}</small>}
            </a>
          ))}
        </nav>
        <div className="sidebar__foot">
          <span className="sidebar__label">CONNECTIONS</span>
          <div>
            WebSocket <Status label={socketLabel} state={connected ? 'connected' : 'unknown'} />
          </div>
          <div>
            API{' '}
            <Status
              label={system?.api ?? 'Checking'}
              state={system?.api === 'Operational' ? 'connected' : 'unknown'}
            />
          </div>
          <div>
            Processor{' '}
            <Status
              label={system?.processor ?? 'Checking'}
              state={system?.processor === 'Running' ? 'connected' : 'unknown'}
            />
          </div>
          <p>
            Person detection. Crowd dynamics.
            <br />
            No facial or identity recognition.
          </p>
        </div>
      </aside>
      <div className="console-main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} aria-hidden="true" />
            <strong>{title}</strong>
          </div>
          <div className="topbar__tools">
            <Status
              label={connected ? 'CONNECTED' : socketLabel.toUpperCase()}
              state={connected ? 'connected' : 'unknown'}
            />
            <time className="mono" dateTime={clock.toISOString()}>
              {clock.toLocaleTimeString([], { hour12: false })}
            </time>
            <button
              className="notification-button"
              onClick={() => navigate('alerts')}
              aria-label={`Open alert center, ${alertCount} open alerts`}
            >
              <Bell size={17} aria-hidden="true" />
              {alertCount > 0 && <span>{alertCount}</span>}
            </button>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          {children}
        </main>
        <footer className="console-footer">
          <span>
            <Radio size={12} aria-hidden="true" /> WebSocket · {socketLabel}
          </span>
          <span>
            Processing ·{' '}
            {active == null ? 'Unknown' : `${active} active worker${active === 1 ? '' : 's'}`}
          </span>
          <span className="mono">
            Last event · {lastEventAt ? formatAgo(lastEventAt) : 'Awaiting telemetry'}
          </span>
        </footer>
      </div>
    </div>
  )
}
