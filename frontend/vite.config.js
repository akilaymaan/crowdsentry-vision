import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Test config lives here rather than in a separate vitest.config.js: a second config
// file shadows this one entirely, and the tests would then run without the React
// plugin.
export default defineConfig({
  plugins: [react()],
  // Vitest transforms modules through its own pipeline, which defaults to the classic
  // JSX runtime and fails with "React is not defined". Stating the automatic runtime
  // explicitly makes both the app build and the tests use the same transform.
  esbuild: { jsx: 'automatic', jsxImportSource: 'react' },
  server: {
    // '::' is a dual-stack bind: without it vite resolves 'localhost' to ::1 only, and
    // any browser connection that races to 127.0.0.1 gets connection-refused — which
    // surfaces as a spurious "cannot reach the backend" on page loads.
    host: '::',
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
    // The detail-view test pays a one-off transform cost for the lazy recharts chunk
    // on a cold cache -- comfortably over the 5 s default, so the limit is raised
    // here rather than per-test.
    testTimeout: 30000,
  },
})
