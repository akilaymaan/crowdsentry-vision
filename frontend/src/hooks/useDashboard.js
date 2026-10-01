import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../api/client'
import { RISK_LEVELS, highestLevel, isAlerting, isFresh } from '../lib/risk'
import { useLiveFeed } from './useLiveFeed'

/** Background reconciliation, in case a websocket event was missed while offline. */
const RECONCILE_MS = 30_000

/**
 * All dashboard state in one place.
 *
 * Cameras are loaded once and then updated **in place** from websocket events — no
 * refetch per event. The header counts are derived from that same camera state rather
 * than re-read from the server, so a card and the header can never disagree.
 *
 * The exception is alerts, which are not pushed over the socket: they are created
 * server-side when a window scores HIGH or CRITICAL, so a risk event at those levels is
 * the signal to re-read the queue.
 */
export function useDashboard() {
  const [cameras, setCameras] = useState([])
  const [alerts, setAlerts] = useState([])
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [acknowledging, setAcknowledging] = useState(() => new Set())

  // A ticking clock held in state, so "12s ago" and staleness stay honest without new
  // data arriving. Kept as state rather than calling Date.now() inside the memo below:
  // reading a clock during render makes the result depend on when React happens to
  // re-render, which is exactly the kind of instability memoisation is meant to avoid.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])

  // If the page loads while the API is restarting (deploy, crash, dev bounce), the
  // first fetch fails at the network level. Retry quietly on a timer rather than
  // stranding the operator on the error screen — mirrors the socket's reconnect loop.
  const retryTimer = useRef(null)
  const loadController = useRef(null)
  const loadVersion = useRef(0)
  useEffect(() => () => clearTimeout(retryTimer.current), [])

  const refreshAlerts = useCallback(async (signal) => {
    try {
      const data = await api.openAlerts(signal)
      if (!signal?.aborted) setAlerts(data.alerts ?? [])
    } catch (cause) {
      if (cause.name !== 'AbortError') console.warn('alert refresh failed', cause)
    }
  }, [])

  const refreshSummary = useCallback(async (signal) => {
    try {
      const data = await api.summary(signal)
      if (!signal?.aborted) setSummary(data)
    } catch (cause) {
      if (cause.name !== 'AbortError') console.warn('summary refresh failed', cause)
    }
  }, [])

  // The camera list itself is not pushed over the socket either: a camera registered
  // or retired while the dashboard is open only reaches it here. The fields this
  // replaces (latest_risk, latest_observation) mirror exactly what websocket events
  // write, so a refresh cannot resurrect data the events already superseded.
  const refreshCameras = useCallback(async (signal) => {
    try {
      const data = await api.cameras(signal)
      if (!signal?.aborted) setCameras(data)
    } catch (cause) {
      if (cause.name !== 'AbortError') console.warn('camera refresh failed', cause)
    }
  }, [])

  /* Initial load ---------------------------------------------------------- */

  const load = useCallback(async function loadData(signal = loadController.current?.signal) {
    if (signal?.aborted) return
    const version = ++loadVersion.current
    clearTimeout(retryTimer.current)
    try {
      const [summaryData, cameraData, alertData] = await Promise.all([
        api.summary(signal),
        api.cameras(signal),
        api.openAlerts(signal),
      ])
      if (signal?.aborted || version !== loadVersion.current) return
      clearTimeout(retryTimer.current)
      setError(null)
      setSummary(summaryData)
      setCameras(cameraData)
      setAlerts(alertData.alerts ?? [])
    } catch (cause) {
      if (!signal?.aborted && version === loadVersion.current && cause.name !== 'AbortError') {
        setError(cause)
        // status 0 = network-level failure: the API process is down or restarting.
        // Retry on a timer; the retry cancels itself the moment a load succeeds.
        if (cause.status === 0) {
          clearTimeout(retryTimer.current)
          retryTimer.current = setTimeout(() => loadData(), 4000)
        }
      }
    } finally {
      if (!signal?.aborted && version === loadVersion.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    loadController.current = controller
    queueMicrotask(() => load(controller.signal))
    return () => {
      controller.abort()
      clearTimeout(retryTimer.current)
    }
  }, [load])

  /* Live updates ---------------------------------------------------------- */

  // Debounce alert refreshes: a burst of CRITICAL windows across cameras should cost
  // one request, not one per event.
  const alertRefreshTimer = useRef(null)
  const queueAlertRefresh = useCallback(() => {
    clearTimeout(alertRefreshTimer.current)
    alertRefreshTimer.current = setTimeout(() => refreshAlerts(), 600)
  }, [refreshAlerts])

  const handleEvent = useCallback(
    (event) => {
      if (event.type === 'hello') {
        if (Array.isArray(event.cameras) && event.cameras.length) {
          setCameras((current) => (current.length ? current : event.cameras))
        }
        return
      }

      if (event.type !== 'risk_score') return

      setCameras((current) =>
        current.map((camera) =>
          camera.id === event.camera_id &&
          (!camera.latest_risk?.timestamp ||
            new Date(event.timestamp) > new Date(camera.latest_risk.timestamp))
            ? {
                ...camera,
                previous_risk_score: camera.latest_risk?.risk_score,
                latest_risk: {
                  timestamp: event.timestamp,
                  risk_score: event.risk_score,
                  risk_level: event.risk_level,
                },
                latest_observation: {
                  timestamp: event.timestamp,
                  // person_count is a per-frame mean on the wire; the card shows a
                  // headcount, so round rather than displaying "3.51 people".
                  person_count: Math.round(event.person_count ?? 0),
                  density: event.density,
                },
                top_feature: event.top_feature ?? null,
              }
            : camera,
        ),
      )

      if (isAlerting(event.risk_level)) queueAlertRefresh()
    },
    [queueAlertRefresh],
  )

  const { status: feedStatus, lastEventAt } = useLiveFeed(handleEvent)

  useEffect(() => () => clearTimeout(alertRefreshTimer.current), [])

  /* Periodic reconciliation ------------------------------------------------ */

  useEffect(() => {
    const controller = new AbortController()
    let busy = false
    const id = setInterval(async () => {
      if (busy) return
      busy = true
      await Promise.all([
        refreshSummary(controller.signal),
        refreshAlerts(controller.signal),
        refreshCameras(controller.signal),
      ])
      busy = false
    }, RECONCILE_MS)
    return () => {
      controller.abort()
      clearInterval(id)
    }
  }, [refreshSummary, refreshAlerts, refreshCameras])

  /* Acknowledging ---------------------------------------------------------- */

  const acknowledge = useCallback(
    async (alertId) => {
      setAcknowledging((current) => new Set(current).add(alertId))
      // Optimistic: drop it from the queue immediately so the button feels instant.
      setAlerts((current) => current.filter((alert) => alert.id !== alertId))
      try {
        await api.acknowledgeAlert(alertId)
      } catch (cause) {
        console.warn('acknowledge failed', cause)
        // Put it back — the operator needs to know it is still open.
        await refreshAlerts()
      } finally {
        setAcknowledging((current) => {
          const next = new Set(current)
          next.delete(alertId)
          return next
        })
      }
    },
    [refreshAlerts],
  )

  /* Derived header figures -------------------------------------------------- */

  const freshnessSeconds = summary?.freshness_seconds ?? 120

  const derived = useMemo(() => {
    const counts = Object.fromEntries(RISK_LEVELS.map((level) => [level, 0]))
    let totalPeople = 0
    let reporting = 0
    const liveLevels = []

    for (const camera of cameras) {
      const observation = camera.latest_observation
      const fresh = isFresh(observation?.timestamp, freshnessSeconds, now)
      if (!fresh) continue

      reporting += 1
      totalPeople += observation?.person_count ?? 0

      const level = camera.latest_risk?.risk_level
      if (level && isFresh(camera.latest_risk.timestamp, freshnessSeconds, now)) {
        counts[level] = (counts[level] ?? 0) + 1
        liveLevels.push(level)
      }
    }

    const activeCameras = cameras.filter((camera) => camera.is_active).length

    return {
      counts,
      totalPeople,
      reporting,
      stale: Math.max(0, activeCameras - reporting),
      totalCameras: cameras.length,
      activeCameras,
      highest: highestLevel(liveLevels),
    }
    // `now` is intentionally a dependency: freshness decays with wall-clock time, so a
    // camera that stops reporting must drop out of these counts without new data.
  }, [cameras, freshnessSeconds, now])

  return {
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
    reload: () => {
      setError(null)
      load()
    },
  }
}
