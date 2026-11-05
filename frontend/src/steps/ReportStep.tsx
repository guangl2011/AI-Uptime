/**
 * Step 3 — assessment report: run the scanner against the pinned version,
 * then read back the stored report. Every finding renders its evidence
 * citations; data gaps render as gaps, never as bare scores.
 */

import { useCallback, useEffect, useState } from 'react'
import { ApiError, getAssessment, listAssessments, runAssessment } from '../api'
import type {
  AssessmentReport,
  AssessmentSummary,
  EvidenceCitation,
  Finding,
  RiskForecast,
  ServiceAssessment,
} from '../types'
import { ErrorNote } from './ErrorNote'

interface Props {
  facilityId: string
  onBack: () => void
}

const FINDING_LABELS: Record<Finding['kind'], string> = {
  single_point_of_failure: 'Single point of failure',
  missing_redundancy: 'Missing redundancy',
  unserved_service: 'Unserved service',
  life_safety_isolation: 'Life-safety isolation',
}

function CitationList({ citations }: { citations: EvidenceCitation[] }) {
  if (citations.length === 0) {
    return <span className="empty">no citations</span>
  }
  return (
    <ul className="citations">
      {citations.map((citation) => (
        <li key={citation.record_id}>
          <code>{citation.kind}</code> — {citation.claim} (provenance #{citation.record_id})
        </li>
      ))}
    </ul>
  )
}

function riskClass(riskLevel: string): string {
  return `risk-${riskLevel}`
}

/** Life-safety marker: a hospital service whose supply must stay isolated. */
function ServiceName({ service }: { service: ServiceAssessment }) {
  return (
    <>
      <code>{service.service_id}</code>{' '}
      {service.life_safety && <span className="badge life_safety">life safety</span>}
    </>
  )
}

/** Forecasts are informational estimates with uncertainty bounds — the flag
 *  exists so the module can be evaluated; nothing executes on these values. */
function Forecasts({ forecasts }: { forecasts: RiskForecast[] }) {
  if (forecasts.length === 0) {
    return null
  }
  return (
    <>
      <h3>Risk forecasts (WP2, feature-flagged)</h3>
      <p className="note">
        Informational estimates with uncertainty bounds — reported, not gated, and
        nothing in the system acts on them. No equipment control exists in this product.
      </p>
      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>p(fault, {forecasts[0].horizon_days}d)</th>
            <th>95% band</th>
            <th>Redundancy</th>
            <th>Basis</th>
          </tr>
        </thead>
        <tbody>
          {forecasts.map((forecast) => (
            <tr key={forecast.asset_key}>
              <td>
                <code>{forecast.asset_key}</code>
              </td>
              {forecast.status === 'abstained' ? (
                <td colSpan={2} className="note">
                  abstained — {forecast.abstention_reason}
                </td>
              ) : (
                <>
                  <td>{forecast.p_fault.toFixed(3)}</td>
                  <td>
                    [{forecast.lower.toFixed(3)}, {forecast.upper.toFixed(3)}]
                  </td>
                </>
              )}
              <td>{forecast.redundancy_class ?? '—'}</td>
              <td>{forecast.basis}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function ReportView({ report }: { report: AssessmentReport }) {
  return (
    <div>
      <p>
        Pinned to graph version <strong>{report.graph_version}</strong> — this report is
        stored and reproducible byte-for-byte.
      </p>

      <h3>Findings (ranked)</h3>
      {report.findings.length === 0 ? (
        <p className="empty">No findings — every service tolerates a first fault.</p>
      ) : (
        <ol className="findings">
          {report.findings.map((finding) => (
            <li key={`${finding.kind}-${finding.subject}`}>
              <span className={`badge ${finding.kind}`}>
                {FINDING_LABELS[finding.kind]}
              </span>{' '}
              <strong>{finding.title}</strong>
              <p>{finding.detail}</p>
              <CitationList citations={finding.evidence} />
            </li>
          ))}
        </ol>
      )}

      <h3>Services</h3>
      <table>
        <thead>
          <tr>
            <th>Service</th>
            <th>Redundancy</th>
            <th>Score</th>
            <th>First fault</th>
            <th>SPOFs</th>
          </tr>
        </thead>
        <tbody>
          {report.services.map((service) => (
            <tr key={service.service_id}>
              <td>
                <ServiceName service={service} />
              </td>
              <td>{service.redundancy}</td>
              <td>
                <span className={riskClass(service.criticality.risk_level)}>
                  {service.criticality.score}
                </span>{' '}
                {service.criticality.drivers.join('; ')}
              </td>
              <td>{service.exposure.first_fault_tolerated ? 'tolerated' : 'interrupts'}</td>
              <td>
                {service.single_points_of_failure.length === 0
                  ? '—'
                  : service.single_points_of_failure.map((spof) => spof.asset_id).join(', ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Assets</h3>
      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>Score</th>
            <th>Exposes</th>
            <th>Absorbed</th>
          </tr>
        </thead>
        <tbody>
          {report.assets.map((asset) => (
            <tr key={asset.node_key}>
              <td>
                <code>{asset.node_key}</code>
              </td>
              <td>
                <span className={riskClass(asset.criticality.risk_level)}>
                  {asset.criticality.score}
                </span>{' '}
                {asset.criticality.drivers.join('; ')}
              </td>
              <td>{asset.exposed_services.join(', ') || '—'}</td>
              <td>{asset.tolerated_services.join(', ') || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Data gaps</h3>
      {report.data_gaps.length === 0 ? (
        <p className="empty">None.</p>
      ) : (
        <ul>
          {report.data_gaps.map((gap) => (
            <li key={gap}>{gap}</li>
          ))}
        </ul>
      )}
      {report.open_questions.length > 0 && (
        <>
          <h3>Open questions</h3>
          <ul>
            {report.open_questions.map((question) => (
              <li key={question}>{question}</li>
            ))}
          </ul>
        </>
      )}

      <Forecasts forecasts={report.forecasts} />
    </div>
  )
}

export function ReportStep({ facilityId, onBack }: Props) {
  const [report, setReport] = useState<AssessmentReport | null>(null)
  const [history, setHistory] = useState<AssessmentSummary[]>([])
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadHistory = useCallback(async () => {
    setError(null)
    try {
      setHistory(await listAssessments(facilityId))
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    }
  }, [facilityId])

  useEffect(() => {
    void loadHistory().finally(() => setLoading(false))
  }, [loadHistory])

  async function run() {
    setRunning(true)
    setError(null)
    try {
      setReport(await runAssessment(facilityId, 'scanner_ui'))
      await loadHistory()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setRunning(false)
    }
  }

  async function loadStored(assessmentId: number) {
    setError(null)
    try {
      setReport(await getAssessment(facilityId, assessmentId))
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    }
  }

  return (
    <section>
      <h2>Assessment report</h2>
      <div className="actions">
        <button type="button" onClick={onBack}>
          Back to topology
        </button>
        <button type="button" disabled={running} onClick={() => void run()}>
          {running ? 'Assessing…' : 'Run new assessment'}
        </button>
      </div>
      <ErrorNote message={error} />

      {loading && <p className="loading">Loading assessment history…</p>}
      {!loading && history.length === 0 && report === null && (
        <p className="empty">
          No assessments yet. Commit a topology version, then run the first assessment.
        </p>
      )}
      {!loading && history.length > 0 && (
        <div className="history">
          <h3>Stored assessments</h3>
          <ul>
            {history.map((summary) => (
              <li key={summary.id}>
                <button
                  type="button"
                  className="link"
                  onClick={() => void loadStored(summary.id)}
                >
                  #{summary.id}
                </button>{' '}
                — graph v{summary.graph_version_no} ({new Date(summary.created_at).toISOString()})
              </li>
            ))}
          </ul>
        </div>
      )}
      {report !== null && <ReportView report={report} />}
    </section>
  )
}
