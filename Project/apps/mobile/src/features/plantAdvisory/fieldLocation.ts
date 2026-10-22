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
// The date and time pickers work in the phone's own time zone, but the
// advisory is always for Sri Lanka time. So the picked date and time are
// kept as a plain wall-clock value: the numbers on the screen are read as
// Sri Lanka time, whatever zone the phone (or emulator) is set to.

// same limits as the backend and the dashboard
export const FORECAST_LIMIT_DAYS = 7;
export const HISTORY_LIMIT_DAYS = 89;

const DAY_MS = 24 * 3600 * 1000;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// The current Sri Lanka wall-clock time, as a Date for the pickers.
export function sriLankaNow(now: Date = new Date()): Date {
  const shifted = new Date(now.getTime() + SRI_LANKA_OFFSET_MINUTES * 60000);
  return new Date(
    shifted.getUTCFullYear(),
    shifted.getUTCMonth(),
    shifted.getUTCDate(),
    shifted.getUTCHours(),
    shifted.getUTCMinutes(),
  );
}

// "2026-10-21T13:00:00+05:30" from a picked wall-clock value
export function advisoryTimestamp(picked: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return (
    `${picked.getFullYear()}-${pad(picked.getMonth() + 1)}-${pad(picked.getDate())}` +
    `T${pad(picked.getHours())}:${pad(picked.getMinutes())}:00+05:30`
  );
}

// Earliest and latest wall-clock values the pickers should offer.
export function advisoryTimeBounds(now: Date = new Date()): { min: Date; max: Date } {
  const current = sriLankaNow(now);
  return {
    min: new Date(current.getTime() - HISTORY_LIMIT_DAYS * DAY_MS),
    max: new Date(current.getTime() + FORECAST_LIMIT_DAYS * DAY_MS),
  };
}

// Error message when the picked time is outside what the weather
// service covers, or null when it's fine.
export function checkAdvisoryTime(picked: Date, now: Date = new Date()): string | null {
  const when = new Date(advisoryTimestamp(picked)).getTime();
  if (when > now.getTime() + FORECAST_LIMIT_DAYS * DAY_MS) {
    return `Weather is only available up to ${FORECAST_LIMIT_DAYS} days ahead. Pick an earlier time.`;
  }
  if (when < now.getTime() - HISTORY_LIMIT_DAYS * DAY_MS) {
    return `Weather history only goes back about ${HISTORY_LIMIT_DAYS} days. Pick a later time.`;
  }
  return null;
}

// "Wed, 21 Oct 2026" and "1:00 PM" for the picker buttons
export function formatPickedDate(picked: Date): string {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return `${days[picked.getDay()]}, ${picked.getDate()} ${MONTHS[picked.getMonth()]} ${picked.getFullYear()}`;
}

export function formatPickedTime(picked: Date): string {
  const hours = picked.getHours();
  const minutes = String(picked.getMinutes()).padStart(2, '0');
  const hour12 = hours % 12 === 0 ? 12 : hours % 12;
  return `${hour12}:${minutes} ${hours >= 12 ? 'PM' : 'AM'}`;
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
  return `${hour12}:${minutes} ${suffix}, ${shifted.getUTCDate()} ${MONTHS[shifted.getUTCMonth()]}`;
}
