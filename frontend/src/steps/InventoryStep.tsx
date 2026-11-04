/**
 * Step 1 — inventory intake: paste-or-upload CSV/JSON, dry-run preview,
 * then commit the import as the facility's first graph version. Ambiguous
 * rows are shown with their reasons — the parser abstains, the UI shows it.
 */

import { useState } from 'react'
import { ApiError, importInventory, previewInventory } from '../api'
import type { Facility, InventoryPreview, UnresolvedRow } from '../types'
import { ErrorNote } from './ErrorNote'

type ImportOutcome = { versionNo: number; nodesCreated: number; unresolved: UnresolvedRow[] }

interface Props {
  facility: Facility
  hasVersions: boolean
  onImported: (outcome: ImportOutcome) => void
  onSkipToTopology: () => void
}

function readFile(file: File): Promise<string> {
  return file.text()
}

export function InventoryStep({ facility, hasVersions, onImported, onSkipToTopology }: Props) {
  const [content, setContent] = useState('')
  const [filename, setFilename] = useState('inventory.csv')
  const [preview, setPreview] = useState<InventoryPreview | null>(null)
  const [busy, setBusy] = useState<'preview' | 'import' | null>(null)
  const [error, setError] = useState<string | null>(null)

  if (hasVersions) {
    return (
      <section>
        <h2>Inventory</h2>
        <p>
          {facility.name} already has a committed graph version. Imports never overwrite
          topology — continue to topology review instead.
        </p>
        <button type="button" onClick={onSkipToTopology}>
          Go to topology review
        </button>
      </section>
    )
  }

  async function handleFileChange(file: File | undefined) {
    if (file === undefined) return
    setError(null)
    setPreview(null)
    setFilename(file.name)
    try {
      setContent(await readFile(file))
    } catch (cause) {
      setError(`could not read ${file.name}: ${String(cause)}`)
    }
  }

  async function runPreview() {
    setBusy('preview')
    setError(null)
    try {
      setPreview(await previewInventory(content, filename))
    } catch (cause) {
      setPreview(null)
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setBusy(null)
    }
  }

  async function runImport() {
    setBusy('import')
    setError(null)
    try {
      const imported = await importInventory(facility.id, content, filename)
      onImported({
        versionNo: imported.version_no,
        nodesCreated: imported.nodes_created,
        unresolved: imported.unresolved,
      })
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setBusy(null)
    }
  }

  return (
    <section>
      <h2>Inventory — {facility.name}</h2>
      <p>
        Paste or upload an equipment inventory (CSV or JSON). The parse is a dry run —
        nothing is committed until you import, and ambiguous rows are flagged, never guessed.
      </p>
      <label className="file-label">
        Upload file
        <input
          type="file"
          accept=".csv,.json,text/csv,application/json"
          onChange={(event) => void handleFileChange(event.target.files?.[0])}
        />
      </label>
      <textarea
        rows={10}
        value={content}
        placeholder={'asset_id,asset_type,name\nCOMP-1,compressor,Compressor 1'}
        onChange={(event) => setContent(event.target.value)}
      />
      <div className="actions">
        <button
          type="button"
          disabled={content.trim() === '' || busy !== null}
          onClick={() => void runPreview()}
        >
          {busy === 'preview' ? 'Parsing…' : 'Preview parse'}
        </button>
        <button
          type="button"
          disabled={
            preview === null || preview.parseable_nodes === 0 || busy !== null
          }
          onClick={() => void runImport()}
        >
          {busy === 'import' ? 'Importing…' : `Import as version 1`}
        </button>
      </div>
      <ErrorNote message={error} />
      {preview !== null && <PreviewTable preview={preview} />}
    </section>
  )
}

function PreviewTable({ preview }: { preview: InventoryPreview }) {
  const empty = preview.parseable_nodes === 0 && preview.unresolved.length === 0
  return (
    <div className="preview">
      <h3>Preview of {preview.filename}</h3>
      {empty && <p className="empty">No rows found in this file.</p>}
      <p>
        {preview.parseable_nodes} parseable row(s)
        {preview.unresolved.length > 0 &&
          `, ${preview.unresolved.length} need attention below`}
      </p>
      {preview.unresolved.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Reason</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {preview.unresolved.map((row) => (
              <tr key={`${row.row}-${row.reason}`}>
                <td>{row.row}</td>
                <td>
                  <code>{row.reason}</code>
                </td>
                <td>{row.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
