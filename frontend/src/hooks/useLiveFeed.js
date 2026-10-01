import { useEffect, useRef, useState } from 'react'

import { liveSocketUrl } from '../api/client'

/**
 * Subscribe to /ws/live.
 *
 * Reconnects with exponential backoff, because a monitoring dashboard is expected to be
 * left open across backend restarts and should recover on its own rather than needing a
 * page refresh.
 *
 * `onEvent` is kept in a ref rather than being an effect dependency: it is almost always
 * an inline closure over component state, so depending on it would tear down and rebuild
 * the socket on every render.
 */
export function useLiveFeed(onEvent) {
  const [status, setStatus] = useState('connecting')
  const [lastEventAt, setLastEventAt] = useState(null)

  const handlerRef = useRef(onEvent)
  useEffect(() => {
    handlerRef.current = onEvent
  }, [onEvent])

  useEffect(() => {
    let socket = null
    let retries = 0
    let retryTimer = null
    // Set during cleanup so an in-flight reconnect cannot resurrect a closed feed.
    let cancelled = false

    // A plain function declaration rather than a useCallback: it refers to itself via
    // scheduleRetry, and a self-capturing useCallback reads its own binding while that
    // binding is still being initialised.
    function connect() {
      if (cancelled) return

      try {
        socket = new WebSocket(liveSocketUrl())
      } catch {
        setStatus('offline')
        scheduleRetry()
        return
      }

      setStatus(retries === 0 ? 'connecting' : 'reconnecting')

      socket.onopen = () => {
        if (cancelled) return
        retries = 0
        setStatus('live')
      }

      socket.onmessage = (message) => {
        let payload
        try {
          payload = JSON.parse(message.data)
        } catch {
          return
        }
        // Keepalives exist to hold the connection open through proxies; they are not
        // data and must not look like activity to the UI.
        if (payload.type === 'keepalive' || payload.type === 'pong') return
        setLastEventAt(Date.now())
        handlerRef.current?.(payload)
      }

      // onerror is not handled separately: onclose always follows it, and carries the
      // information worth acting on.
      socket.onclose = (event) => {
        if (cancelled) return
        // 4401 = the backend rejected the API key at handshake. Retrying can never
        // fix a bad credential, so stop instead of hammering a closed door.
        if (event.code === 4401) {
          setStatus('unauthorized')
          return
        }
        setStatus('offline')
        scheduleRetry()
      }
    }

    function scheduleRetry() {
      if (cancelled) return
      // 1s, 2s, 4s ... capped at 15s. Fast enough to feel instant after a quick backend
      // restart, slow enough not to hammer a server that is genuinely down.
      const delay = Math.min(1000 * 2 ** retries, 15000)
      retries += 1
      clearTimeout(retryTimer)
      retryTimer = setTimeout(connect, delay)
    }

    connect()

    return () => {
      // React 18 StrictMode mounts effects twice in development; without this guard the
      // first socket is left dangling and every event arrives twice.
      cancelled = true
      clearTimeout(retryTimer)
      if (socket && socket.readyState <= WebSocket.OPEN) socket.close()
      socket = null
    }
  }, [])

  return { status, lastEventAt }
}
