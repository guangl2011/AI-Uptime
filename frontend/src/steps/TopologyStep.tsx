/**
 * Step 2 — topology confirm/edit: review the parsed nodes, declare which
 * assets serve which services, set N / N+1 / 2N redundancy, and commit a
 * new immutable graph version. Every edit carries technician-entry
 * provenance, so the assessment stays traceable to who claimed what.
 */

import { useEffect, useState } from 'react'
import { ApiError, commitGraphVersion, getGraph } from '../api'
import type { GraphSnapshot, RedundancyConfig } from '../types'
import { ErrorNote } from './ErrorNote'

const REDUNDANCY_OPTIONS: RedundancyConfig[] = ['N', 'N+1', '2N']

interface EdgeDraft {
  fromNode: string
  toNode: string
  redundancy: RedundancyConfig
  memberGroup: string
}

interface Props {
  facilityId: string
  onCommitted: (versionNo: number) => void
}

function emptyEdge(): EdgeDraft {
  return { fromNode: '', toNode: '', redundancy: 'N', memberGroup: '' }
}

function draftValid(draft: EdgeDraft, keys: Set<string>): boolean {
  return (
    draft.fromNode !== '' &&
    draft.toNode !== '' &&
    keys.has(draft.fromNode) &&
    keys.has(draft.toNode) &&
    draft.fromNode !== draft.toNode
  )
}

function commitPayload(
  snapshot: GraphSnapshot,
  newServiceNames: string[],
  drafts: EdgeDraft[],
): { author: string; notes: string; nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> } {
  const serviceNodes = newServiceNames.map((name, index) => ({
    node_key: name,
    node_type: 'service',
    name,
    provenance: {
      source_kind: 'technician_entry',
      source_system: 'scanner_ui',
      source_ref: `operator-defined service #${index + 1}`,
    },
  }))
  const edges = drafts
    .filter((draft) => draftValid(draft, nodeKeysOf(snapshot, newServiceNames)))
    .map((draft) => ({
      from_node: draft.fromNode,
      to_node: draft.toNode,
      redundancy_config: draft.redundancy,
      member_group: draft.memberGroup === '' ? null : draft.memberGroup,
      provenance: {
        source_kind: 'technician_entry',
        source_system: 'scanner_ui',
        source_ref: `operator-declared dependency ${draft.fromNode}->${draft.toNode}`,
      },
    }))
  return {
    author: 'scanner_ui',
    notes: 'topology confirmation from the onboarding scanner',
    nodes: [...snapshot.nodes, ...serviceNodes],
    edges,
  }
}

function nodeKeysOf(snapshot: GraphSnapshot, newServiceNames: string[]): Set<string> {
  const keys = new Set(snapshot.nodes.map((node) => node.node_key))
  for (const name of newServiceNames) keys.add(name)
  return keys
}

export function TopologyStep({ facilityId, onCommitted }: Props) {
  const [snapshot, setSnapshot] = useState<GraphSnapshot | null>(null)
  const [drafts, setDrafts] = useState<EdgeDraft[]>([])
  const [newServices, setNewServices] = useState<string[]>([])
  const [serviceInput, setServiceInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [committing, setCommitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getGraph(facilityId)
      .then((loaded) => {
        if (!cancelled) {
          setSnapshot(loaded)
          if (loaded.edges.length === 0) setDrafts([emptyEdge()])
          else {
            setDrafts(
              loaded.edges.map((edge) => ({
                fromNode: edge.from_node,
                toNode: edge.to_node,
                redundancy: edge.redundancy_config ?? 'N',
                memberGroup: edge.member_group ?? '',
              })),
            )
          }
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof ApiError ? cause.message : String(cause))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [facilityId])

  if (loading) return <p className="loading">Loading graph…</p>
  if (snapshot === null) {
    return <ErrorNote message={error ?? 'graph unavailable'} />
  }

  const keys = nodeKeysOf(snapshot, newServices)
  const allAssets = snapshot.nodes.filter((node) => node.node_type === 'asset')
  const services = snapshot.nodes.filter((node) => node.node_type === 'service')
  const serviceNames = services.map((node) => node.node_key)
  const anyDraftValid = drafts.some((draft) => draftValid(draft, keys))

  function updateDraft(index: number, patch: Partial<EdgeDraft>) {
    setDrafts(drafts.map((draft, i) => (i === index ? { ...draft, ...patch } : draft)))
  }

  function addService() {
    const name = serviceInput.trim()
    const valid = /^[A-Za-z0-9_.-]{1,120}$/.test(name)
    if (!valid || keys.has(name)) {
      setError(
        `service key "${name}" must be 1-120 chars of letters/digits/._- and unique`,
      )
      return
    }
    setError(null)
    setNewServices([...newServices, name])
    setServiceInput('')
  }

  async function commit() {
    if (snapshot === null) return
    setCommitting(true)
    setError(null)
    try {
      const version = await commitGraphVersion(
        facilityId,
        commitPayload(snapshot, newServices, drafts),
      )
      onCommitted(version.version_no)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setCommitting(false)
    }
  }

  return (
    <section>
      <h2>Topology review — version {snapshot.version_no}</h2>
      <p>
        Confirm which assets serve which services. Redundancy: N fails on the first
        fault, N+1 absorbs one, 2N absorbs a full path. Group redundant suppliers with
        the same member-group name.
      </p>

      <h3>Assets from inventory ({allAssets.length})</h3>
      <ul className="node-list">
        {allAssets.map((node) => (
          <li key={node.node_key}>
            <code>{node.node_key}</code> {node.asset_type ?? 'unknown type'}
          </li>
        ))}
      </ul>

      <h3>Services ({[...serviceNames, ...newServices].length})</h3>
      {serviceNames.length === 0 && newServices.length === 0 && (
        <p className="empty">
          No services defined yet — add at least one (e.g. “SVC-POWER” or
          “SVC-REFRIG-CR1”).
        </p>
      )}
      <ul className="node-list">
        {serviceNames.map((name) => (
          <li key={name}>
            <code>{name}</code> (from inventory)
          </li>
        ))}
        {newServices.map((name) => (
          <li key={name}>
            <code>{name}</code> (new)
          </li>
        ))}
      </ul>
      <div className="actions">
        <input
          value={serviceInput}
          placeholder="new service key, e.g. SVC-POWER"
          onChange={(event) => setServiceInput(event.target.value)}
        />
        <button type="button" onClick={addService}>
          Add service
        </button>
      </div>

      <h3>Dependencies</h3>
      <table>
        <thead>
          <tr>
            <th>Supplier</th>
            <th>Serves</th>
            <th>Redundancy</th>
            <th>Member group</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {drafts.map((draft, index) => (
            <tr key={index}>
              <td>
                <select
                  value={draft.fromNode}
                  onChange={(event) => updateDraft(index, { fromNode: event.target.value })}
                >
                  <option value="">—</option>
                  {[...keys].sort().map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <select
                  value={draft.toNode}
                  onChange={(event) => updateDraft(index, { toNode: event.target.value })}
                >
                  <option value="">—</option>
                  {[...serviceNames, ...newServices].map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <select
                  value={draft.redundancy}
                  onChange={(event) =>
                    updateDraft(index, {
                      redundancy: event.target.value as RedundancyConfig,
                    })
                  }
                >
                  {REDUNDANCY_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </td>
              <td>
                <input
                  value={draft.memberGroup}
                  placeholder="e.g. power_sources"
                  onChange={(event) => updateDraft(index, { memberGroup: event.target.value })}
                />
              </td>
              <td>
                <button
                  type="button"
                  className="link"
                  onClick={() => setDrafts(drafts.filter((_, i) => i !== index))}
                >
                  remove
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="actions">
        <button type="button" onClick={() => setDrafts([...drafts, emptyEdge()])}>
          Add dependency
        </button>
        <button
          type="button"
          disabled={!anyDraftValid || committing}
          onClick={() => void commit()}
        >
          {committing ? 'Committing…' : 'Commit topology version'}
        </button>
      </div>
      <ErrorNote message={error} />
    </section>
  )
}
