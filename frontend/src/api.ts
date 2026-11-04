/**
 * Typed fetch client for the AI-Uptime backend. All calls hit the FastAPI
 * surface through the dev proxy; every error surfaces as an ApiError with
 * the server's detail — nothing is swallowed or guessed.
 */

import type {
  AssessmentReport,
  AssessmentSummary,
  Facility,
  GraphSnapshot,
  GraphVersionSummary,
  InventoryImport,
  InventoryPreview,
} from './types'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (cause) {
    throw new ApiError(
      0,
      `backend unreachable at ${path} — is uvicorn running on :8000? (${String(cause)})`,
    )
  }
  if (!response.ok) {
    const detail = await extractDetail(response)
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

async function extractDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof body.detail === 'string'
    ) {
      return body.detail
    }
  } catch {
    // Non-JSON body — fall through to the generic status text.
  }
  return `${response.status} ${response.statusText}`
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export async function listFacilities(): Promise<Facility[]> {
  return request<Facility[]>('/facilities')
}

export async function createFacility(payload: {
  id: string | null
  name: string
  segment: string
}): Promise<Facility> {
  return request<Facility>('/facilities', jsonInit('POST', payload))
}

export async function listVersions(facilityId: string): Promise<GraphVersionSummary[]> {
  return request<GraphVersionSummary[]>(`/facilities/${facilityId}/graph/versions`)
}

export async function getGraph(
  facilityId: string,
  version?: number,
): Promise<GraphSnapshot> {
  const suffix = version === undefined ? '' : `?version=${version}`
  return request<GraphSnapshot>(`/facilities/${facilityId}/graph${suffix}`)
}

export async function commitGraphVersion(
  facilityId: string,
  payload: {
    author: string
    notes: string | null
    nodes: Array<Record<string, unknown>>
    edges: Array<Record<string, unknown>>
  },
): Promise<GraphVersionSummary> {
  return request<GraphVersionSummary>(
    `/facilities/${facilityId}/graph/versions`,
    jsonInit('POST', payload),
  )
}

export async function previewInventory(
  content: string,
  filename: string,
): Promise<InventoryPreview> {
  const form = new FormData()
  form.append('file', new File([content], filename))
  return request<InventoryPreview>('/inventory/preview', { method: 'POST', body: form })
}

export async function importInventory(
  facilityId: string,
  content: string,
  filename: string,
): Promise<InventoryImport> {
  const form = new FormData()
  form.append('file', new File([content], filename))
  return request<InventoryImport>(`/facilities/${facilityId}/inventory`, {
    method: 'POST',
    body: form,
  })
}

export async function runAssessment(
  facilityId: string,
  author: string,
): Promise<AssessmentReport> {
  return request<AssessmentReport>(
    `/facilities/${facilityId}/assessments`,
    jsonInit('POST', { author }),
  )
}

export async function listAssessments(facilityId: string): Promise<AssessmentSummary[]> {
  return request<AssessmentSummary[]>(`/facilities/${facilityId}/assessments`)
}

export async function getAssessment(
  facilityId: string,
  assessmentId: number,
): Promise<AssessmentReport> {
  return request<AssessmentReport>(
    `/facilities/${facilityId}/assessments/${assessmentId}`,
  )
}
