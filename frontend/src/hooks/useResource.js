import { useEffect, useState } from 'react'

export function useResource(loader, interval = 30_000) {
  const [state, setState] = useState({ data: null, error: null, loader: null, revision: null })
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer
    async function refresh() {
      try {
        const data = await loader(controller.signal)
        if (!controller.signal.aborted) setState({ data, error: null, loader, revision })
      } catch (error) {
        if (!controller.signal.aborted)
          setState((current) => ({
            data: current.loader === loader ? current.data : null,
            error,
            loader,
            revision,
          }))
      } finally {
        if (!controller.signal.aborted && interval) timer = setTimeout(refresh, interval)
      }
    }
    refresh()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [loader, interval, revision])

  const current = state.loader === loader && state.revision === revision
  return {
    data: current ? state.data : null,
    error: current ? state.error : null,
    loading: !current,
    reload: () => setRevision((value) => value + 1),
  }
}
