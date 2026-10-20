import type { LocationSource } from '../../types/plantAdvisory';

// Sri Lanka has no daylight saving, so a fixed offset is safe
const SRI_LANKA_OFFSET_MINUTES = 330;

export interface FieldLocation {
  latitude: number;
  longitude: number;
  source: LocationSource;
  // GPS accuracy in metres, when it came from the device
  accuracy: number | null;
}

// Reads typed coordinates. Returns the location, or a message per field.
export function parseCoordinates(
  latitudeText: string,
  longitudeText: string,
): { location: FieldLocation | null; errors: { latitude?: string; longitude?: string } } {
  const errors: { latitude?: string; longitude?: string } = {};
  const latitude = Number(latitudeText.trim());
  const longitude = Number(longitudeText.trim());

  if (!latitudeText.trim() || !Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
    errors.latitude = 'Enter a latitude between -90 and 90.';
  }
  if (!longitudeText.trim() || !Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    errors.longitude = 'Enter a longitude between -180 and 180.';
  }
  if (errors.latitude || errors.longitude) {
    return { location: null, errors };
  }
  return { location: { latitude: round6(latitude), longitude: round6(longitude), source: 'typed', accuracy: null }, errors };
}

export function fromGps(latitude: number, longitude: number, accuracy: number | null): FieldLocation {
  return {
    latitude: round6(latitude),
    longitude: round6(longitude),
    source: 'gps',
    accuracy: accuracy != null && Number.isFinite(accuracy) ? Math.round(accuracy) : null,
  };
}

export function fromMap(latitude: number, longitude: number): FieldLocation {
  return { latitude: round6(latitude), longitude: round6(longitude), source: 'map', accuracy: null };
}

// 6 decimals is about 10 cm - plenty for weather, and easy to read
function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

// "2026-10-20T10:42:00+05:30" for the given moment, in Sri Lanka time.
// The backend needs an explicit offset and floors it to the hour.
export function sriLankaTimestamp(moment: Date = new Date()): string {
  const shifted = new Date(moment.getTime() + SRI_LANKA_OFFSET_MINUTES * 60000);
  return `${shifted.toISOString().slice(0, 19)}+05:30`;
}

// ---------------- advisory time ----------------
// Tapping usually starts early in the morning, so the future options
// are 6 AM. Both stay well inside the backend's 7-day forecast limit.

export type AdvisoryTimeKey = 'now' | 'tomorrow_6am' | 'day_after_6am';

export const ADVISORY_TIME_OPTIONS: { key: AdvisoryTimeKey; label: string; caption: string }[] = [
  { key: 'now', label: 'Now', caption: 'Current weather' },
  { key: 'tomorrow_6am', label: 'Tomorrow, 6 AM', caption: 'Forecast for the morning tapping' },
  { key: 'day_after_6am', label: 'Day after tomorrow, 6 AM', caption: 'Forecast, less certain' },
];

// Timestamp (Sri Lanka time) for the chosen option.
export function advisoryTimestamp(key: AdvisoryTimeKey, now: Date = new Date()): string {
  if (key === 'now') {
    return sriLankaTimestamp(now);
  }
  const daysAhead = key === 'tomorrow_6am' ? 1 : 2;
  // today's date as it is in Sri Lanka, then move forward whole days
  const shifted = new Date(now.getTime() + SRI_LANKA_OFFSET_MINUTES * 60000);
  shifted.setUTCDate(shifted.getUTCDate() + daysAhead);
  return `${shifted.toISOString().slice(0, 10)}T06:00:00+05:30`;
}

// "10:00 AM, 20 Oct" from a backend timestamp, read in Sri Lanka time
export function formatObservationHour(timestamp: string): string {
  const moment = new Date(timestamp);
  if (Number.isNaN(moment.getTime())) {
    return timestamp;
  }
  const shifted = new Date(moment.getTime() + SRI_LANKA_OFFSET_MINUTES * 60000);
  const hours = shifted.getUTCHours();
  const minutes = String(shifted.getUTCMinutes()).padStart(2, '0');
  const suffix = hours >= 12 ? 'PM' : 'AM';
  const hour12 = hours % 12 === 0 ? 12 : hours % 12;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${hour12}:${minutes} ${suffix}, ${shifted.getUTCDate()} ${months[shifted.getUTCMonth()]}`;
}
