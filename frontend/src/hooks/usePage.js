import { useEffect, useState } from 'react'

const PAGES = new Set(['overview', 'cameras', 'analytics', 'alerts', 'system'])
function readPage() {
  const page = window.location.hash.slice(1)
  return PAGES.has(page) ? page : 'overview'
}

export function usePage() {
  const [page, setPage] = useState(readPage)
  useEffect(() => {
    const update = () => setPage(readPage())
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  return [
    page,
    (value) => {
      window.location.hash = value
      setPage(value)
    },
  ]
}
