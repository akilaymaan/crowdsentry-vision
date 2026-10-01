import '@testing-library/jest-dom/vitest'

// jsdom has no WebSocket. The dashboard opens one on mount, so without a stub every
// test would throw before rendering anything.
class MockWebSocket {
  static instances = []
  static OPEN = 1

  constructor(url) {
    this.url = url
    this.readyState = 0
    MockWebSocket.instances.push(this)
  }

  close() {
    this.readyState = 3
  }

  /** Drive the hook from a test: pretend the server sent this frame. */
  emit(payload) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  open() {
    this.readyState = 1
    this.onopen?.()
  }
}

globalThis.WebSocket = MockWebSocket
globalThis.__MockWebSocket = MockWebSocket

// jsdom has no ResizeObserver, which Recharts' ResponsiveContainer needs. A stub is
// enough: the tests assert on text and structure, not on measured layout.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
