import type {
  AdvisoryRecordCreate,
  DrcPredictionResponse,
  LatexCondition,
  PriceEstimationResponse,
  TappingClassProbabilities,
  TappingLabel,
  TappingSuitabilityResponse,
  WeatherInputs,
  WeatherSnapshotResponse,
} from '../../types/plantAdvisory';
import type { FieldLocation } from './fieldLocation';

// The weather values go to the models exactly as the backend returned
// them - same names and units, nothing converted or rounded.
export function weatherInputs(weather: WeatherSnapshotResponse): WeatherInputs {
  return {
    temperature_c: weather.temperature_c,
    relative_humidity_pct: weather.relative_humidity_pct,
    rainfall_last_6h_mm: weather.rainfall_last_6h_mm,
    forecast_rain_next_6h_mm: weather.forecast_rain_next_6h_mm,
    rainfall_last_3days_mm: weather.rainfall_last_3days_mm,
    wind_speed_kmh: weather.wind_speed_kmh,
  };
}

// What the farmer sees in the collection cup. It matters most for the DRC:
// watery latex has less rubber in it, so it is also worth less.
// null = not checked; the backend then assumes normal latex.
export const LATEX_CONDITION_OPTIONS: { key: LatexCondition | null; label: string; caption: string }[] = [
  { key: null, label: 'Not checked', caption: 'Normal latex will be assumed' },
  { key: 'Normal', label: 'Normal', caption: 'White and milky, flows evenly' },
  { key: 'Watery', label: 'Watery', caption: 'Thin and runny, often after rain' },
  { key: 'Early coagulation', label: 'Early coagulation', caption: 'Lumps or clots already forming' },
];

// What the result screen shows. Built only from backend responses.
export interface AdvisoryResult {
  latitude: number;
  longitude: number;
  locationSource: FieldLocation['source'];
  observationTimestamp: string;
  forecastBased: boolean;
  latexCondition: LatexCondition | null;
  drcPct: number;
  tapping: { label: TappingLabel; classOrder: TappingLabel[]; probabilities: TappingClassProbabilities } | null;
  tappingError: string | null;
  priceLkrPerKg: number | null;
  priceError: string | null;
}

export function buildAdvisoryResult(
  location: FieldLocation,
  weather: WeatherSnapshotResponse,
  latexCondition: LatexCondition | null,
  drc: DrcPredictionResponse,
  tapping: TappingSuitabilityResponse | null,
  tappingError: string | null,
  price: PriceEstimationResponse | null,
  priceError: string | null,
): AdvisoryResult {
  return {
    latitude: location.latitude,
    longitude: location.longitude,
    locationSource: location.source,
    observationTimestamp: weather.observation_timestamp,
    forecastBased: weather.data_quality.observation_in_future,
    latexCondition,
    drcPct: drc.predicted_latex_drc_pct,
    tapping: tapping
      ? { label: tapping.predicted_label, classOrder: tapping.class_order, probabilities: tapping.probabilities }
      : null,
    tappingError,
    priceLkrPerKg: price ? price.predicted_price_lkr_per_kg : null,
    priceError,
  };
}

// The record saved for Component 2 (DRC with its location and time).
export function buildAdvisoryRecord(
  location: FieldLocation,
  weather: WeatherSnapshotResponse,
  drc: DrcPredictionResponse,
  tapping: TappingSuitabilityResponse | null,
  price: PriceEstimationResponse | null,
): AdvisoryRecordCreate {
  return {
    latitude: location.latitude,
    longitude: location.longitude,
    location_source: location.source,
    observation_timestamp: weather.observation_timestamp,
    forecast_based: weather.data_quality.observation_in_future,
    temperature_c: weather.temperature_c,
    relative_humidity_pct: weather.relative_humidity_pct,
    wind_speed_kmh: weather.wind_speed_kmh,
    rainfall_last_6h_mm: weather.rainfall_last_6h_mm,
    rainfall_last_3days_mm: weather.rainfall_last_3days_mm,
    forecast_rain_next_6h_mm: weather.forecast_rain_next_6h_mm,
    predicted_latex_drc_pct: drc.predicted_latex_drc_pct,
    drc_model_name: drc.model_name,
    drc_runtime_artifact_sha256: drc.runtime_artifact_sha256,
    tapping_label: tapping ? tapping.predicted_label : null,
    tapping_probability_high: tapping ? tapping.probabilities.high : null,
    tapping_probability_low: tapping ? tapping.probabilities.low : null,
    tapping_probability_moderate: tapping ? tapping.probabilities.moderate : null,
    predicted_price_lkr_per_kg: price ? price.predicted_price_lkr_per_kg : null,
  };
}

// The result travels to the next screen as a route parameter (a string).
// Reading it back checks the shape, so a stale or broken link shows an
// empty state instead of crashing.
export function parseAdvisoryResult(raw: string | string[] | undefined): AdvisoryResult | null {
  if (typeof raw !== 'string') {
    return null;
  }
  try {
    const value = JSON.parse(raw) as AdvisoryResult;
    if (
      typeof value.latitude !== 'number' ||
      typeof value.longitude !== 'number' ||
      typeof value.drcPct !== 'number' ||
      typeof value.observationTimestamp !== 'string'
    ) {
      return null;
    }
    return value;
  } catch {
    return null;
  }
}
