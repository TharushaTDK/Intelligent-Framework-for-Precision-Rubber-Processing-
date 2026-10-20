// field names match the Component 1 backend:
//   backend/app/schemas/weather_provider.py   (weather snapshot)
//   backend/app/schemas/weather_tapping.py    (DRC, tapping, price)
//   backend/app/schemas/advisory_record.py    (saved advisories)
//   backend/app/api/routes/weather_tapping.py (model catalog)
// timestamps are ISO strings with an explicit offset, e.g. 2026-10-20T10:00:00+05:30

// ---------------- weather snapshot ----------------

export interface WeatherSnapshotRequest {
  latitude: number;
  longitude: number;
  observation_timestamp: string;
}

export interface RainfallWindowConvention {
  feature: string;
  window: string;
  verification: 'unverified_runtime_convention';
}

export interface WeatherDataQuality {
  historical_window_start: string;
  historical_window_end: string;
  forecast_window_end: string;
  missing_hourly_values: string[];
  rainfall_window_conventions: RainfallWindowConvention[];
  grid_cell_latitude: number | null;
  grid_cell_longitude: number | null;
  // true when the hour is in the future, so every value is a forecast
  observation_in_future: boolean;
  forecast_hours_in_history_window: number;
}

export interface WeatherSnapshotResponse {
  observation_timestamp: string;
  observation_timezone: string;
  temperature_c: number;
  relative_humidity_pct: number;
  wind_speed_kmh: number;
  rainfall_last_6h_mm: number;
  rainfall_last_3days_mm: number;
  forecast_rain_next_6h_mm: number;
  provider: string;
  data_quality: WeatherDataQuality;
}

// ---------------- prediction requests ----------------
// every input except the timestamp is optional: anything left out is
// treated as "not observed" and filled by the model's training median

export interface WeatherInputs {
  temperature_c?: number;
  relative_humidity_pct?: number;
  rainfall_last_6h_mm?: number;
  forecast_rain_next_6h_mm?: number;
  rainfall_last_3days_mm?: number;
  wind_speed_kmh?: number;
}

export interface DrcPredictionRequest extends WeatherInputs {
  observation_timestamp: string;
  soil_moisture_pct?: number;
  soil_ph?: number;
  estate_id?: string;
  tree_count?: number;
  tapping_system?: string;
  days_since_last_tap?: number;
  previous_collection_yield_litres?: number;
}

export interface TappingSuitabilityRequest extends WeatherInputs {
  observation_timestamp: string;
  soil_moisture_pct?: number;
  soil_ph?: number;
  tree_count?: number;
  tapping_system?: string;
  days_since_last_tap?: number;
}

export interface PriceEstimationRequest {
  observation_timestamp: string;
  latex_drc_pct?: number;
  temperature_c?: number;
  relative_humidity_pct?: number;
  rainfall_last_6h_mm?: number;
  soil_moisture_pct?: number;
  soil_ph?: number;
  tree_count?: number;
  tapping_system?: string;
}

// ---------------- prediction responses ----------------

interface ModelIdentifierFields {
  model_name: string;
  model_class: string;
  runtime_artifact_sha256: string;
  original_artifact_sha256: string;
  dataset_provenance: string;
  observation_month: number;
  observation_day_of_year: number;
}

export interface DrcPredictionResponse extends ModelIdentifierFields {
  target_name: 'latex_drc_pct';
  unit: 'percent';
  predicted_latex_drc_pct: number;
}

export type TappingLabel = 'High' | 'Low' | 'Moderate';

export interface TappingClassProbabilities {
  high: number;
  low: number;
  moderate: number;
}

export interface TappingSuitabilityResponse extends ModelIdentifierFields {
  target_name: 'tapping_suitability_proxy';
  predicted_label: TappingLabel;
  class_order: TappingLabel[];
  probabilities: TappingClassProbabilities;
  is_rule_derived_proxy: boolean;
  research_warning: string;
}

export interface PriceEstimationResponse extends ModelIdentifierFields {
  target_name: 'price_lkr_per_kg';
  unit: 'LKR per kg';
  predicted_price_lkr_per_kg: number;
}

// ---------------- saved advisories (read by Component 2) ----------------

export type LocationSource = 'map' | 'gps' | 'typed';

export interface AdvisoryRecordCreate {
  latitude: number;
  longitude: number;
  location_source?: LocationSource | null;
  observation_timestamp: string;
  forecast_based: boolean;
  temperature_c: number;
  relative_humidity_pct: number;
  wind_speed_kmh: number;
  rainfall_last_6h_mm: number;
  rainfall_last_3days_mm: number;
  forecast_rain_next_6h_mm: number;
  predicted_latex_drc_pct: number;
  drc_model_name: string;
  drc_runtime_artifact_sha256: string;
  tapping_label?: TappingLabel | null;
  tapping_probability_high?: number | null;
  tapping_probability_low?: number | null;
  tapping_probability_moderate?: number | null;
  predicted_price_lkr_per_kg?: number | null;
}

export interface AdvisoryRecordResponse extends AdvisoryRecordCreate {
  id: number;
  created_at: string;
}

// ---------------- model catalog ----------------

export interface ModelFeatureContract {
  name: string;
  target: string;
  model_class: string;
  runtime_artifact_sha256: string;
  original_artifact_sha256: string;
  feature_count: number;
  features_in_pipeline_order: string[];
  classes_in_probability_order: string[] | null;
  dataset_provenance: string;
  target_warning: string | null;
}

export interface ModelCatalogResponse {
  component: string;
  observation_timezone: string;
  known_limitations: string[];
  models: ModelFeatureContract[];
}
