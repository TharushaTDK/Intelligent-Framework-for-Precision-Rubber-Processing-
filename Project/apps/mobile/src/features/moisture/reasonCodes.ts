/**
 * Human-readable translations for Component 4 acoustic technical checks
 * and preliminary quality classifier rejection reasons and warnings.
 */

export const REJECTION_REASON_DESCRIPTIONS: Record<string, string> = {
  FILE_OR_FORMAT_ERROR:
    'Audio format error: The WAV file could not be parsed or is not a supported mono PCM WAV (16-, 24-, or 32-bit).',
  SILENT_RECORDING:
    'Silent recording: The microphone signal remained below the minimum audible energy threshold.',
  NO_EVENT:
    'No impact detected: No clear strike transient was identified in the audio waveform.',
  NO_EVENT_DETECTED:
    'No clear strike event was detected. Repeat one clean strike using the standard setup.',
  CAPTURE_TOO_SHORT:
    'Capture too short: The recorded audio length was insufficient for event segmentation.',
  MULTIPLE_EVENT_CANDIDATES:
    'Multiple sound-event candidates detected. Extra impacts or other sharp sounds may be present. Repeat one clean strike.',
  INCOMPLETE_RESPONSE:
    'Incomplete analysis window: The recording does not contain the full required segment around the detected event.',
  BACKGROUND_INTERVAL_CONTAMINATED:
    'Early strike: The impact occurred during the initial ambient noise calibration window (first 200 ms).',
  EXCESSIVE_NEAR_FULL_SCALE_SAMPLES:
    'Too many samples are near digital full scale. Review the recording level and standard setup; this check does not identify the cause.',
  ML_RECORDING_QUALITY_REJECTION:
    'Acoustic quality rejection: The learned classifier determined the recording does not meet quality standards for standardized analysis.',
};

export const WARNING_DESCRIPTIONS: Record<string, string> = {
  HIGH_PEAK_REVIEW:
    'High amplitude warning: Peak amplitude reached 99% or higher of full scale. Review for potential clipping.',
};

/**
 * Returns a user-friendly explanation for a rejection reason code.
 */
export function getReadableRejectionReason(code: string): string {
  if (REJECTION_REASON_DESCRIPTIONS[code]) {
    return REJECTION_REASON_DESCRIPTIONS[code];
  }
  // Human-readable fallback for unanticipated codes
  const formatted = code
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
  return `Quality check failed (${formatted}). Repeat the strike following standard setup.`;
}

/**
 * Returns a user-friendly explanation for a warning code.
 */
export function getReadableWarning(code: string): string {
  if (WARNING_DESCRIPTIONS[code]) {
    return WARNING_DESCRIPTIONS[code];
  }
  return code.replace(/_/g, ' ');
}
