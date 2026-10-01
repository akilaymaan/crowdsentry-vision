import { useMemo } from 'react'

import { formatAgo, isFresh, riskColors } from '../lib/risk'
import './CrowdMap.css'

const VIEW_W = 1000
const VIEW_H = 620
const PAD = 90
const METRES_PER_DEG_LAT = 111_320

/**
 * Camera positions plotted from latitude/longitude.
 *
 * Deliberately hand-drawn SVG rather than Leaflet or MapLibre. At venue scale — the
 * cameras here span a few hundred metres — street tiles add no information, cost a
 * network dependency (and for most providers an API key), and would need offline tiles
 * for a control room with no internet. What this view actually needs is relative
 * position and risk colour, which is geometry, not cartography. Swapping in Leaflet later
 * means replacing this one component; nothing else knows about the projection.
 */
function project(cameras) {
  if (!cameras.length) return { points: [], scaleMetres: null }

  const lats = cameras.map((c) => c.latitude)
  const lons = cameras.map((c) => c.longitude)
  const minLat = Math.min(...lats)
  const maxLat = Math.max(...lats)
  const minLon = Math.min(...lons)
  const maxLon = Math.max(...lons)

  // Longitude degrees shrink toward the poles; without this correction an east-west
  // spread is drawn wider than it is on the ground.
  const midLat = (minLat + maxLat) / 2
  const lonScale = Math.cos((midLat * Math.PI) / 180)

  const spanLat = maxLat - minLat
  const spanLon = (maxLon - minLon) * lonScale

  // A single camera, or several at the same spot, has zero span: fall back to a
  // nominal window so the divisions below stay finite.
  const span = Math.max(spanLat, spanLon, 1e-6)

  const usableW = VIEW_W - PAD * 2
  const usableH = VIEW_H - PAD * 2
  const size = Math.min(usableW, usableH)

  const points = cameras.map((camera) => {
    const nx = spanLon > 1e-9 ? ((camera.longitude - minLon) * lonScale) / span : 0.5
    const ny = spanLat > 1e-9 ? (camera.latitude - minLat) / span : 0.5
    return {
      camera,
      x: PAD + (usableW - size) / 2 + nx * size,
      // SVG y grows downward, latitude grows north: flip so north is up.
      y: PAD + (usableH - size) / 2 + (1 - ny) * size,
    }
  })

  const scaleMetres = span * METRES_PER_DEG_LAT
  return { points, scaleMetres }
}

export default function CrowdMap({ cameras, selectedId, onSelect, freshnessSeconds }) {
  const { points, scaleMetres } = useMemo(() => project(cameras), [cameras])

  return (
    <section className="panel map-panel">
      <div className="panel-title">
        <span>Live crowd map</span>
        {scaleMetres != null && (
          <span className="muted mono">~{Math.round(scaleMetres)} m across</span>
        )}
      </div>

      <div className="map">
        {points.length === 0 ? (
          <p className="empty">No cameras to plot.</p>
        ) : (
          <svg
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            className="map__svg"
            role="img"
            aria-label="Camera positions coloured by risk level"
          >
            <defs>
              <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                <path
                  d="M 50 0 L 0 0 0 50"
                  fill="none"
                  stroke="var(--border)"
                  strokeWidth="1"
                  opacity="0.5"
                />
              </pattern>
              <radialGradient id="vignette">
                <stop offset="60%" stopColor="transparent" />
                <stop offset="100%" stopColor="rgba(0,0,0,0.35)" />
              </radialGradient>
            </defs>

            <rect width={VIEW_W} height={VIEW_H} fill="url(#grid)" />
            <rect width={VIEW_W} height={VIEW_H} fill="url(#vignette)" />

            {points.map(({ camera, x, y }, index) => (
              <Marker
                key={camera.id}
                index={index}
                camera={camera}
                x={x}
                y={y}
                selected={camera.id === selectedId}
                onSelect={onSelect}
                freshnessSeconds={freshnessSeconds}
              />
            ))}
          </svg>
        )}
      </div>
    </section>
  )
}

function Marker({ camera, index, x, y, selected, onSelect, freshnessSeconds }) {
  const observation = camera.latest_observation
  const live = isFresh(observation?.timestamp, freshnessSeconds)
  const level = live ? camera.latest_risk?.risk_level : null
  const { fg } = riskColors(level)

  const people = live ? (observation?.person_count ?? 0) : 0
  // Area-proportional sizing: radius by sqrt so a marker twice the radius represents
  // four times the people, which is how the eye reads a disc.
  const radius = 13 + Math.min(20, Math.sqrt(people) * 3.2)
  const alarming = live && (level === 'HIGH' || level === 'CRITICAL')

  return (
    <g
      className={`marker ${selected ? 'marker--selected' : ''} ${live ? '' : 'marker--stale'}`}
      style={{ '--marker': fg, '--mi': index }}
      transform={`translate(${x} ${y})`}
      onClick={() => onSelect(camera.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect(camera.id)
        }
      }}
      aria-label={`${camera.name}, ${level ?? 'no data'}, ${people} people`}
    >
      <g className="marker__pop">
        {alarming && <circle className="marker__halo" r={radius + 10} />}
        {selected && <circle className="marker__select-ring" r={radius + 6} />}

        <circle className="marker__disc" r={radius} />
        <circle className="marker__core" r={radius * 0.42} />

        <text className="marker__count mono" y={-radius - 12} textAnchor="middle">
          {live ? people : '—'}
        </text>
        <text className="marker__label" y={radius + 20} textAnchor="middle">
          {camera.name}
        </text>
        <text className="marker__sub" y={radius + 35} textAnchor="middle">
          {live ? (level ?? 'no score') : `stale · ${formatAgo(observation?.timestamp)}`}
        </text>
      </g>
    </g>
  )
}
