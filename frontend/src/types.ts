/**
 * API payload types, mirroring the FastAPI contracts (graph/schemas.py,
 * assessor/schemas.py). The backend is the source of truth — these are
 * hand-kept in sync and validated by tsc against real responses in dev.
 */

export type Segment = 'cold_chain' | 'aidc' | 'hospital'

export type NodeType = 'asset' | 'service'

export type RedundancyConfig = 'N' | 'N+1' | '2N'

export type SourceKind =
  | 'inventory_import'
  | 'technician_entry'
  | 'oem_doc'
  | 'alarm'
  | 'work_order'
  | 'sensor'

export type RiskLevel = 'low' | 'elevated' | 'high'

export type FindingKind =
  | 'single_point_of_failure'
  | 'missing_redundancy'
  | 'unserved_service'

export interface Facility {
  id: string
  name: string
  segment: Segment
}

export interface Provenance {
  id: number
  source_kind: SourceKind
  source_system: string
  source_ref: string
}

export interface GraphNode {
  node_key: string
  node_type: NodeType
  asset_type: string | null
  name: string
  location: string | null
  attributes: Record<string, unknown>
  provenance: Provenance
}

export interface GraphEdge {
  from_node: string
  to_node: string
  failure_mode: string | null
  redundancy_config: RedundancyConfig | null
  member_group: string | null
  provenance: Provenance
}

export interface GraphSnapshot {
  facility_id: string
  version_no: number
  author: string
  notes: string | null
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphVersionSummary {
  version_no: number
  author: string
  notes: string | null
  node_count: number
  edge_count: number
}

export interface UnresolvedRow {
  row: number
  reason: string
  detail: string
}

export interface InventoryPreview {
  filename: string
  parseable_nodes: number
  unresolved: UnresolvedRow[]
}

export interface InventoryImport {
  facility_id: string
  version_no: number
  nodes_created: number
  unresolved: UnresolvedRow[]
}

export interface EvidenceCitation {
  record_id: number
  kind: SourceKind
  claim: string
}

export interface CriticalityScore {
  score: number
  risk_level: RiskLevel
  drivers: string[]
}

export interface SpofEntry {
  asset_id: string
  reason: string
}

export interface Exposure {
  first_fault_tolerated: boolean
  concurrent_faults_tolerated: number
  unprotected_assets: string[]
}

export interface ServiceAssessment {
  service_id: string
  name: string
  redundancy: RedundancyConfig
  criticality: CriticalityScore
  exposure: Exposure
  single_points_of_failure: SpofEntry[]
  evidence: EvidenceCitation[]
  data_gaps: string[]
}

export interface AssetAssessment {
  node_key: string
  name: string
  asset_type: string | null
  criticality: CriticalityScore
  exposed_services: string[]
  tolerated_services: string[]
  evidence: EvidenceCitation[]
}

export interface Finding {
  kind: FindingKind
  subject: string
  title: string
  detail: string
  affected_services: string[]
  evidence: EvidenceCitation[]
}

export interface AssessmentReport {
  facility_id: string
  graph_version: number
  evidence_window: { from: string | null; to: string | null } | null
  services: ServiceAssessment[]
  assets: AssetAssessment[]
  findings: Finding[]
  data_gaps: string[]
  open_questions: string[]
}

export interface AssessmentSummary {
  id: number
  facility_id: string
  graph_version_no: number
  created_at: string
}
