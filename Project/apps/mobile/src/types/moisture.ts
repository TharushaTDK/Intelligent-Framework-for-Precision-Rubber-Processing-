/**
 * TypeScript contracts for Component 4 — “How’s the Moisture?” (Acoustic Quality Assessment).
 * Matches the backend Pydantic models from backend/app/schemas/thud_test.py.
 */

export type ProcessingStatus = 'COMPLETED';
export type TechnicalStatus = 'TECHNICAL_PASSED' | 'TECHNICAL_REJECTED';
export type QualityStatus = 'VALID' | 'INVALID';
export type DecisionSource = 'TECHNICAL_RULES' | 'ML_CLASSIFIER';
export type ModelStatus = 'PRELIMINARY_DEMO';

export interface AcousticMeasurementResponse {
  measurement_id: string;
  created_at: string;
  filename: string;
  sheet_id: string | null;
  condition_id: string | null;
  source_sha256: string;
  processing_status: ProcessingStatus;
  technical_status: TechnicalStatus;
  quality_status: QualityStatus;
  classifier_status: string;
  decision_source: DecisionSource;
  rejection_reasons: string[];
  warnings: string[];
  rule_version: string;
  model_version: string | null;
  model_status: ModelStatus;
  invalid_probability_uncalibrated: number | null;
  candidate_count: number;
  onset_seconds: number | null;
  segment_duration_seconds: number | null;
  peak_percent_full_scale: number | null;
  near_full_scale_fraction: number | null;
  moisture_percent: null;
  guidance: string;
}

export interface AcousticHealthResponse {
  component: string;
  model_available: boolean;
  model_version: string | null;
  status: 'PRELIMINARY_DEMO' | 'MODEL_UNAVAILABLE' | string;
}

export interface AcousticDeviceStatusResponse {
  configured: boolean;
  capture_path: string;
}

export interface CaptureQueryParams {
  sheet_id?: string;
  condition_id?: string;
}
