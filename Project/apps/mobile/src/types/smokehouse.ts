/**
 * TypeScript mirror of the Component 3 Smokehouse API contract.
 *
 * Field names match the FastAPI/Pydantic schemas exactly
 * (backend/app/schemas/smokehouse.py). This file defines shapes
 * only — no ML/business logic is implemented here. All inference
 * and guidance text originate from the backend response.
 *
 * Timestamps are ISO 8601 strings, as returned by FastAPI's JSON
 * serialization of Python datetime values.
 */

export type LayerCode = 'L1' | 'L2' | 'L3';

export type ReadingSource = 'manual' | 'iot';

export type BatchStatus = 'active' | 'completed' | 'cancelled';

export type FireRiskLabel = 'Normal' | 'Warning' | 'Critical';

// ============================================================
// LAYER SCHEMAS
// ============================================================

export interface SmokehouseLayerCreate {
  layer_code: LayerCode;
  number_of_sheets: number;
  layer_area_m2: number;
}

export interface SmokehouseLayerResponse {
  id: number;
  layer_code: LayerCode;
  number_of_sheets: number;
  layer_area_m2: number;
}

// ============================================================
// BATCH SCHEMAS
// ============================================================

export interface SmokehouseBatchCreate {
  batch_code: string;
  smokehouse_id: string;
  started_at?: string;
  layers: SmokehouseLayerCreate[];
}

export interface SmokehouseBatchResponse {
  id: number;
  batch_code: string;
  smokehouse_id: string;
  started_at: string;
  ended_at: string | null;
  status: BatchStatus;
  layers: SmokehouseLayerResponse[];
}

// ============================================================
// READING SCHEMAS
// ============================================================

export interface SmokehouseReadingCreate {
  batch_code: string;
  layer_code: LayerCode;
  temperature_c: number;
  humidity_pct: number;
  recorded_at?: string;
  source?: ReadingSource;
}

export interface SmokehouseReadingResponse {
  id: number;
  layer_id: number;
  recorded_at: string;
  temperature_c: number;
  humidity_pct: number;
  source: ReadingSource;
}

// ============================================================
// PREDICTION SCHEMAS
// ============================================================

export interface FireRiskProbabilities {
  normal: number;
  warning: number;
  critical: number;
}

export interface SmokehousePredictionResponse {
  prediction_id: number;
  reading_id: number;
  predicted_at: string;
  remaining_drying_time_hours: number;
  fire_risk_label: FireRiskLabel;
  probabilities: FireRiskProbabilities;
  guidance: string | null;
  regression_model_version: string;
  classification_model_version: string;
}

export interface SmokehouseReadingPredictionResponse {
  reading: SmokehouseReadingResponse;
  prediction: SmokehousePredictionResponse;
  missing_numeric_features: string[];
  historical_reading_count: number;
}

// ============================================================
// LATEST STATUS / HISTORY SCHEMAS
// ============================================================

export interface SmokehouseLatestStatusResponse {
  batch_code: string;
  smokehouse_id: string;
  layer_code: LayerCode;
  latest_reading: SmokehouseReadingResponse;
  latest_prediction: SmokehousePredictionResponse | null;
}

export interface SmokehouseHistoryItemResponse {
  reading: SmokehouseReadingResponse;
  prediction: SmokehousePredictionResponse | null;
}

export interface SmokehouseLayerHistoryResponse {
  batch_code: string;
  smokehouse_id: string;
  layer_code: LayerCode;
  items: SmokehouseHistoryItemResponse[];
}
