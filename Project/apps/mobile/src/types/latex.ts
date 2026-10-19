// field names match backend/app/schemas/coagulation.py

export type CoagulationOutput = 'water_acid' | 'coagulation_time';

export type RecommendationStatus = 'recommended' | 'in_progress' | 'completed' | 'cancelled';

export interface CoagulationRecommendationCreate {
  estate_id: string;
  latex_volume_L: number;
  outputs: CoagulationOutput[];
  drc?: number;
}

export interface WaterAcidResult {
  water_amount_L: number;
  prepared_1pct_formic_acid_solution_mL: number;
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
}

export interface CoagulationHistoryResponse {
  items: CoagulationRecommendationResponse[];
}
