import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Local dev proxies the FastAPI surface so the app can run against
// `uvicorn uptime_api.app:app` with no CORS changes on the backend.
const apiProxy = Object.fromEntries(
  ['/facilities', '/inventory', '/assessments', '/healthz'].map((path) => [
    path,
    { target: 'http://localhost:8000' },
  ]),
)

export default defineConfig({
  plugins: [react()],
  server: { proxy: apiProxy },
})
