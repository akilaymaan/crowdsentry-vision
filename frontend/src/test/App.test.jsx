/**
 * Render smoke tests for the dashboard.
 *
 * These mount the real component tree against payloads shaped exactly like the API's,
 * so a renamed field or a crash on a null value fails here rather than showing a blank
 * panel in the browser.
 */
// Explicit React import: the plugin's automatic JSX runtime does not cover this file
// in the vitest pipeline, so JSX here compiles to React.createElement.
import React from 'react'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '../App'

const NOW = new Date().toISOString()
const LONG_AGO = new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString()

const CAMERAS = [
  {
    id: 1,
    name: 'CAM-01-NORTH-GATE',
    location_name: 'Stadium North Gate',
    latitude: 12.9789,
    longitude: 77.5998,
    area_sq_meters: 2.2,
    pixels_per_meter: 38,
    stream_configured: true,
    is_active: true,
    created_at: NOW,
    latest_risk: { timestamp: NOW, risk_score: 71.7, risk_level: 'CRITICAL' },
    latest_observation: { timestamp: NOW, person_count: 4, density: 1.602 },
  },
  {
    id: 2,
    name: 'CAM-02-CONCOURSE',
    location_name: 'Upper Concourse',
    latitude: 12.9793,
    longitude: 77.6004,
    area_sq_meters: 6,
    pixels_per_meter: 52,
    stream_configured: true,
    is_active: true,
    created_at: NOW,
    latest_risk: { timestamp: NOW, risk_score: 60.5, risk_level: 'HIGH' },
    latest_observation: { timestamp: NOW, person_count: 4, density: 0.587 },
  },
  {
    id: 3,
    name: 'CAM-03-METRO-EXIT',
    location_name: 'Metro Exit Plaza',
    latitude: 12.9761,
    longitude: 77.6032,
    area_sq_meters: 40,
    pixels_per_meter: null,
    stream_configured: false,
    is_active: true,
    created_at: NOW,
    // Reported hours ago: must be treated as stale, not as current.
    latest_risk: { timestamp: LONG_AGO, risk_score: 18.2, risk_level: 'LOW' },
    latest_observation: { timestamp: LONG_AGO, person_count: 99, density: 0.088 },
  },
]

const SUMMARY = {
  generated_at: NOW,
  total_cameras: 3,
  active_cameras: 3,
  reporting_cameras: 2,
  stale_cameras: 1,
  total_people: 8,
  mean_density: 1.09,
  peak_density: 1.602,
  cameras_by_risk_level: { LOW: 0, MODERATE: 0, HIGH: 1, CRITICAL: 1 },
  highest_risk_level: 'CRITICAL',
  unacknowledged_alerts: 1,
  freshness_seconds: 120,
}

const ALERTS = {
  total: 1,
  count: 1,
  alerts: [
    {
      id: 9,
      camera_id: 1,
      timestamp: NOW,
      risk_score: 71.7,
      message:
        'CRITICAL crowd risk detected near Stadium North Gate - risk 72/100, density 1.60 people/m2',
      acknowledged: false,
      camera_name: 'CAM-01-NORTH-GATE',
      camera_location: 'Stadium North Gate',
    },
  ],
}

function mockApi(overrides = {}) {
  return vi.fn(async (url) => {
    const path = String(url)
    const body = path.includes('/api/dashboard/summary')
      ? (overrides.summary ?? SUMMARY)
      : path.includes('/api/alerts')
        ? (overrides.alerts ?? ALERTS)
        : path.match(/\/api\/cameras\/\d+\/history/)
          ? (overrides.history ?? { camera_id: 1, camera_name: 'CAM-01-NORTH-GATE', observations: [], risk_scores: [], truncated: false })
          : path.match(/\/api\/cameras\/\d+$/)
            ? (overrides.detail ?? { ...CAMERAS[0], unacknowledged_alerts: 1 })
            : (overrides.cameras ?? CAMERAS)
    return { ok: true, status: 200, json: async () => body }
  })
}

beforeEach(() => {
  globalThis.__MockWebSocket.instances = []
  globalThis.fetch = mockApi()
})

/* Camera names appear twice by design — once on a card, once as a map marker — so every
 * lookup has to say which one it means.
 *
 * These scope to the container `render` returns rather than to `document`. Querying the
 * document finds the *first* matching node, which during a full run can belong to a
 * previous test's container: the lookup then succeeds against detached DOM and clicking
 * it silently does nothing. */
function renderApp() {
  const { container } = render(<App />)

  const grid = () => container.querySelector('.camera-grid')
  const mapSvg = () => container.querySelector('.map__svg')

  async function findCard(name) {
    await waitFor(() => expect(grid()).toBeTruthy())
    return await waitFor(() => within(grid()).getByText(name).closest('button'))
  }

  return { container, grid, mapSvg, findCard }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('dashboard', () => {
  it('renders the header, cameras, map and alerts from the API', async () => {
    const { findCard, mapSvg } = renderApp()

    expect(await screen.findByText('CrowdSentry')).toBeInTheDocument()

    // A card per camera...
    for (const camera of CAMERAS) {
      expect(await findCard(camera.name)).toBeTruthy()
    }
    // ...and a marker per camera on the map.
    expect(screen.getByRole('img', { name: /camera positions/i })).toBeInTheDocument()
    for (const camera of CAMERAS) {
      expect(within(mapSvg()).getByText(camera.name)).toBeInTheDocument()
    }

    // Alert banner shows the open alert and its acknowledge control.
    expect(await screen.findByText(/CRITICAL crowd risk detected/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /acknowledge/i })).toBeInTheDocument()

    // Risk badges reflect each camera's level.
    expect(within(await findCard('CAM-01-NORTH-GATE')).getByText('CRITICAL')).toBeInTheDocument()
    expect(within(await findCard('CAM-02-CONCOURSE')).getByText('HIGH')).toBeInTheDocument()
  })

  it('counts only fresh cameras in the header', async () => {
    const { findCard } = renderApp()
    await findCard('CAM-01-NORTH-GATE')

    // Two cameras reported recently; the third reported hours ago. Its 99 people must
    // not reach the headline count, which would otherwise read 107.
    expect(await screen.findByText('2/3')).toBeInTheDocument()
    expect(screen.getByText('8')).toBeInTheDocument()
    expect(screen.getByText('1 not reporting')).toBeInTheDocument()
  })

  it('shows a stale camera as having no current data', async () => {
    const { findCard } = renderApp()
    const stale = await findCard('CAM-03-METRO-EXIT')

    expect(stale).toHaveClass('camera-card--stale')
    expect(within(stale).getByText(/no data/i)).toBeInTheDocument()
    // Its last known headcount must not be presented as current.
    expect(within(stale).queryByText('99')).not.toBeInTheDocument()
  })

  it('updates a camera in place from a websocket event, without refetching', async () => {
    const { findCard } = renderApp()
    await findCard('CAM-02-CONCOURSE')

    const callsBefore = globalThis.fetch.mock.calls.length
    const socket = globalThis.__MockWebSocket.instances.at(-1)

    socket.emit({
      type: 'risk_score',
      camera_id: 2,
      camera_name: 'CAM-02-CONCOURSE',
      location_name: 'Upper Concourse',
      timestamp: new Date().toISOString(),
      risk_score: 88.4,
      risk_level: 'CRITICAL',
      person_count: 17.4,
      density: 2.9,
      top_feature: 'density',
    })

    const card = await findCard('CAM-02-CONCOURSE')
    // 17.4 is a per-frame mean; the card shows a headcount.
    await waitFor(() => expect(within(card).getByText('17')).toBeInTheDocument())
    expect(within(card).getByText('2.90/m²')).toBeInTheDocument()
    expect(within(card).getByText('CRITICAL')).toBeInTheDocument()

    // A LOW/MODERATE event needs no refetch at all; this one is CRITICAL so it
    // triggers exactly one debounced alert re-read, not a full reload.
    const newCalls = globalThis.fetch.mock.calls
      .slice(callsBefore)
      .map(([url]) => String(url))
    expect(newCalls.filter((url) => url.includes('/api/cameras'))).toHaveLength(0)
  })

  it('acknowledging an alert removes it from the queue immediately', async () => {
    const user = userEvent.setup()
    renderApp()

    await screen.findByText(/CRITICAL crowd risk detected/)
    await user.click(screen.getByRole('button', { name: /acknowledge/i }))

    await waitFor(() =>
      expect(screen.queryByText(/CRITICAL crowd risk detected/)).not.toBeInTheDocument(),
    )
    expect(
      globalThis.fetch.mock.calls.some(([url]) =>
        String(url).includes('/api/alerts/9/acknowledge'),
      ),
    ).toBe(true)
  })

  it('shows the all-clear when there are no alerts', async () => {
    globalThis.fetch = mockApi({ alerts: { total: 0, count: 0, alerts: [] } })
    renderApp()

    expect(await screen.findByText(/no active alerts/i)).toBeInTheDocument()
  })

  it('opens the detail view when a camera is clicked', async () => {
    const user = userEvent.setup()
    const { findCard } = renderApp()

    await user.click(await findCard('CAM-01-NORTH-GATE'))

    // CameraDetail is a lazy chunk that pulls in recharts; the first import pays a
    // transform cost that a short timeout fails on a cold cache.
    const dialog = await screen.findByRole('dialog', {}, { timeout: 20000 })
    expect(within(dialog).getByText(/density & risk over time/i)).toBeInTheDocument()
  })

  it('explains itself when the backend is unreachable', async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    })
    renderApp()

    expect(await screen.findByText(/cannot reach the backend/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })

  it('renders with no cameras at all', async () => {
    globalThis.fetch = mockApi({ cameras: [] })
    renderApp()

    expect(await screen.findByText(/no cameras registered/i)).toBeInTheDocument()
  })

  it('survives a camera that has never been scored', async () => {
    globalThis.fetch = mockApi({
      cameras: [{ ...CAMERAS[0], latest_risk: null, latest_observation: null }],
    })
    const { findCard } = renderApp()

    expect(await findCard('CAM-01-NORTH-GATE')).toBeTruthy()
    expect(screen.getAllByText('NO DATA').length).toBeGreaterThan(0)
  })
})
