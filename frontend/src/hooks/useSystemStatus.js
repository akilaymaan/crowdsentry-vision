import { api } from '../api/client'
import { useResource } from './useResource'

async function checkSystem(signal) {
  const results = await Promise.allSettled([
    api.health(signal),
    api.readiness(signal),
    api.processorStatus(signal),
  ])
  const [health, ready, processor] = results
  return {
    checkedAt: new Date().toISOString(),
    api:
      health.status === 'fulfilled' && health.value.status === 'ok' ? 'Operational' : 'Unavailable',
    database:
      ready.status === 'fulfilled' && ready.value.status === 'ready' ? 'Connected' : 'Unavailable',
    processor:
      processor.status === 'fulfilled' && typeof processor.value.running === 'boolean'
        ? processor.value.running
          ? 'Running'
          : 'Stopped'
        : 'Unknown',
    workers:
      processor.status === 'fulfilled' && Array.isArray(processor.value.cameras)
        ? processor.value.cameras.map(
            ({
              camera_id,
              name,
              state,
              frames_processed,
              windows_processed,
              last_window_at,
              alerts_raised,
            }) => ({
              camera_id,
              name,
              state,
              frames_processed,
              windows_processed,
              last_window_at,
              alerts_raised,
            }),
          )
        : null,
  }
}

export function useSystemStatus() {
  return useResource(checkSystem, 15_000)
}
