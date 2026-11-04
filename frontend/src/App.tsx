/**
 * App shell — Phase 0 stub. The three MVP screens (facility/topology view,
 * evidence review, assessment report) land here in Phase 2 against the
 * frozen API contract; this stub only proves the build and dev proxy.
 */

export default function App() {
  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', margin: '3rem auto', maxWidth: '40rem' }}>
      <h1>AI-Uptime Console</h1>
      <p>Phase 0 scaffold — facility onboarding, topology, and assessment views arrive in later phases.</p>
      <p>
        Backend liveness: <code>GET /healthz</code> (dev proxy → <code>http://localhost:8000</code>).
      </p>
    </main>
  )
}
