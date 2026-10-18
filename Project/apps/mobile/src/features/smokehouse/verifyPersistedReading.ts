import type {
  SmokehouseLatestStatusResponse,
  SmokehouseReadingPredictionResponse,
} from '../../types/smokehouse';

/**
 * Compares a freshly-submitted reading/prediction response against the
 * backend's persisted "latest status" for the same layer.
 *
 * This is NOT model validation and does NOT recompute any ML output.
 * It only checks that the values the mobile app just received match
 * what the backend says it actually persisted, using the exact values
 * already returned by each endpoint's JSON.
 */

export type VerificationStatus = 'verified' | 'mismatch' | 'unavailable';

export interface VerificationResult {
  status: VerificationStatus;
  mismatchedFields: string[];
}

export function verifyPersistedReading(
  posted: SmokehouseReadingPredictionResponse,
  latest: SmokehouseLatestStatusResponse,
): VerificationResult {
  const mismatchedFields: string[] = [];

  const latestReading = latest.latest_reading;
  const latestPrediction = latest.latest_prediction;

  if (latestReading.id !== posted.reading.id) {
    mismatchedFields.push('reading.id');
  }

  if (!latestPrediction) {
    mismatchedFields.push('prediction (missing from latest status)');
  } else {
    if (latestPrediction.prediction_id !== posted.prediction.prediction_id) {
      mismatchedFields.push('prediction.prediction_id');
    }

    if (latestPrediction.fire_risk_label !== posted.prediction.fire_risk_label) {
      mismatchedFields.push('prediction.fire_risk_label');
    }

    if (
      latestPrediction.remaining_drying_time_hours !==
      posted.prediction.remaining_drying_time_hours
    ) {
      mismatchedFields.push('prediction.remaining_drying_time_hours');
    }

    if (latestPrediction.probabilities.normal !== posted.prediction.probabilities.normal) {
      mismatchedFields.push('prediction.probabilities.normal');
    }

    if (latestPrediction.probabilities.warning !== posted.prediction.probabilities.warning) {
      mismatchedFields.push('prediction.probabilities.warning');
    }

    if (latestPrediction.probabilities.critical !== posted.prediction.probabilities.critical) {
      mismatchedFields.push('prediction.probabilities.critical');
    }

    if (latestPrediction.guidance !== posted.prediction.guidance) {
      mismatchedFields.push('prediction.guidance');
    }

    if (
      latestPrediction.regression_model_version !==
      posted.prediction.regression_model_version
    ) {
      mismatchedFields.push('prediction.regression_model_version');
    }

    if (
      latestPrediction.classification_model_version !==
      posted.prediction.classification_model_version
    ) {
      mismatchedFields.push('prediction.classification_model_version');
    }
  }

  return {
    status: mismatchedFields.length === 0 ? 'verified' : 'mismatch',
    mismatchedFields,
  };
}
