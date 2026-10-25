// field names match backend/app/schemas/coagulation.py

export type CoagulationOutput = 'water_acid' | 'coagulation_time';

export type RecommendationStatus = 'recommended' | 'in_progress' | 'completed' | 'cancelled';

export type StageKey = 'liquid' | 'thickening' | 'setting' | 'firm';

export interface CoagulationRecommendationCreate {
  // map point, the backend finds the DRC there
  latitude: number;
  longitude: number;
  // weather hour from /location-drc, so the batch uses the DRC the farmer saw
  observation_timestamp?: string;
  latex_volume_L: number;
  outputs: CoagulationOutput[];
}

export interface LocationDrcResponse {
  latitude: number;
  longitude: number;
  location: string;
  drc: number;
  drc_source: string;
  observation_timestamp: string;
}

export interface WaterAcidResult {
  water_amount_L: number;
  prepared_1pct_formic_acid_solution_mL: number;
}

export interface CoagulationStage {
  key: StageKey;
  number: number;
  label: string;
  description: string;
}

export interface CoagulationStep {
  number: number;
  title: string;
  detail: string;
}

export interface CoagulationAlert {
  alert_at: string;
  type: 'progress' | 'due' | 'overdue';
  message: string;
  is_past: boolean;
}

export interface CoagulationRecommendationResponse {
  id: number;
  estate_id: string;
  latex_volume_L: number;
  drc: number;
  drc_source: string;
  outputs: CoagulationOutput[];
  water_acid: WaterAcidResult | null;
  coagulation_time_hours: number | null;
  outside_training_range: boolean;
  notes: string[];
  water_acid_model_version: string;
  coagulation_time_model_version: string | null;
  status: RecommendationStatus;
  created_at: string;
  started_at: string | null;
  expected_completion_at: string | null;
  completed_at: string | null;
  actual_coagulation_time_hours: number | null;
  acid_used_mL: number | null;
  steps: CoagulationStep[];
  time_remaining_hours: number | null;
  is_due: boolean | null;
  next_alert_at: string | null;
  alerts: CoagulationAlert[];
  progress_pct: number | null;
  stage: CoagulationStage | null;
}

export interface CoagulationHistoryResponse {
  items: CoagulationRecommendationResponse[];
}

export interface AcidStockResponse {
  stock_mL: number | null;
  updated_at: string | null;
  average_batch_acid_mL: number | null;
  batches_left: number | null;
  low_stock: boolean;
  message: string;
}

export interface CoagulationPreviewRequest {
  drc: number;
  latex_volume_L: number;
  outputs: CoagulationOutput[];
}

export interface CoagulationPreviewResponse {
  drc: number;
  latex_volume_L: number;
  water_acid: WaterAcidResult | null;
  coagulation_time_hours: number | null;
  outside_training_range: boolean;
  notes: string[];
}
