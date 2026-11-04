/**
 * App shell — the Phase 2 onboarding scanner: pick or create a facility,
 * intake inventory, confirm topology, read the scored assessment report.
 * Every step talks to the real backend APIs; there is no mock data path.
 */

import { useCallback, useEffect, useState } from 'react'
import { ApiError, createFacility, listFacilities, listVersions } from './api'
import type { Facility } from './types'
import { ErrorNote } from './steps/ErrorNote'
import { InventoryStep } from './steps/InventoryStep'
import { ReportStep } from './steps/ReportStep'
import { TopologyStep } from './steps/TopologyStep'
import './styles.css'

type Screen = 'inventory' | 'topology' | 'report'

const SEGMENTS: Array<{ value: string; label: string }> = [
  { value: 'cold_chain', label: 'Cold chain' },
  { value: 'aidc', label: 'AIDC data center' },
  { value: 'hospital', label: 'Hospital' },
]

function FacilityPicker({
  onSelected,
}: {
  onSelected: (facility: Facility) => void
}) {
  const [facilities, setFacilities] = useState<Facility[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [segment, setSegment] = useState('cold_chain')

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setFacilities(await listFacilities())
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  async function submit() {
    setError(null)
    try {
      const created = await createFacility({
        id: null,
        name: name.trim(),
        segment,
      })
      onSelected(created)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    }
  }

  return (
    <section>
      <h2>Choose a facility</h2>
      {loading && <p className="loading">Loading facilities…</p>}
      <ErrorNote message={error} onRetry={load} />
      {!loading && facilities.length === 0 && (
        <p className="empty">
          No facilities yet — create the first one below to start the onboarding scan.
        </p>
      )}
      {facilities.length > 0 && (
        <ul className="facility-list">
          {facilities.map((facility) => (
            <li key={facility.id}>
              <button type="button" onClick={() => onSelected(facility)}>
                {facility.name}{' '}
                <span className="muted">
                  ({facility.id} · {facility.segment})
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <h3>New facility</h3>
      <div className="actions">
        <input
          value={name}
          placeholder="facility name, e.g. Riverside Cold Storage"
          onChange={(event) => setName(event.target.value)}
        />
        <select value={segment} onChange={(event) => setSegment(event.target.value)}>
          {SEGMENTS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          disabled={name.trim() === ''}
          onClick={() => void submit()}
        >
          Create
        </button>
      </div>
    </section>
  )
}

export default function App() {
  const [facility, setFacility] = useState<Facility | null>(null)
  const [screen, setScreen] = useState<Screen>('inventory')
  const [hasVersions, setHasVersions] = useState(false)

  async function selectFacility(selected: Facility) {
    setFacility(selected)
    setScreen('inventory')
    try {
      const versions = await listVersions(selected.id)
      setHasVersions(versions.length > 0)
      setScreen(versions.length > 0 ? 'topology' : 'inventory')
    } catch {
      // Conservative default: show the inventory step; its own error surface
      // reports backend problems when actions run.
      setHasVersions(false)
    }
  }

  if (facility === null) {
    return (
      <main>
        <h1>AI-Uptime Console</h1>
        <FacilityPicker onSelected={(selected) => void selectFacility(selected)} />
      </main>
    )
  }

  return (
    <main>
      <header className="topbar">
        <button
          type="button"
          className="link"
          onClick={() => {
            setFacility(null)
            setHasVersions(false)
          }}
        >
          ← facilities
        </button>
        <h1>{facility.name}</h1>
        <nav className="steps">
          {(['inventory', 'topology', 'report'] as const).map((step) => (
            <button
              key={step}
              type="button"
              className={screen === step ? 'active' : undefined}
              onClick={() => setScreen(step)}
            >
              {step}
            </button>
          ))}
        </nav>
      </header>
      {screen === 'inventory' && (
        <InventoryStep
          facility={facility}
          hasVersions={hasVersions}
          onImported={() => {
            setHasVersions(true)
            setScreen('topology')
          }}
          onSkipToTopology={() => setScreen('topology')}
        />
      )}
      {screen === 'topology' && (
        <TopologyStep
          facilityId={facility.id}
          onCommitted={() => setScreen('report')}
        />
      )}
      {screen === 'report' && (
        <ReportStep facilityId={facility.id} onBack={() => setScreen('topology')} />
      )}
    </main>
  )
}
