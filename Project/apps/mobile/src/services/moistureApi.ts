import type {
  AcousticDeviceStatusResponse,
  AcousticHealthResponse,
  AcousticMeasurementResponse,
  CaptureQueryParams,
} from '../types/moisture';

/**
 * Component 4 API Base URL.
 *
 * Defaults to port 8001 matching the developer's tested backend setup.
 * Android emulator forwards to the host via:
 *   adb reverse tcp:8001 tcp:8001
 * Can be overridden via EXPO_PUBLIC_ACOUSTIC_API_BASE_URL.
 */
export const MOISTURE_API_BASE_URL =
  (process.env.EXPO_PUBLIC_ACOUSTIC_API_BASE_URL?.trim() || 'http://127.0.0.1:8001').replace(/\/+$/, '');

const THUD_TEST_PREFIX = '/api/v1/thud-test';

// Timeout configuration
const CAPTURE_TIMEOUT_MS = 60000; // 60s for ESP32 capture + backend segmentation/ML
const HEALTH_TIMEOUT_MS = 10000; // 10s for health/device status
const LOOKUP_TIMEOUT_MS = 15000; // 15s for saved measurement retrieval

export class MoistureApiError extends Error {
  status: number;
  isTimeout: boolean;
  detail?: unknown;

  constructor(message: string, status: number, isTimeout = false, detail?: unknown) {
    super(message);
    this.name = 'MoistureApiError';
    this.status = status;
    this.isTimeout = isTimeout;
    this.detail = detail;
  }
}

/**
 * Parse FastAPI JSON error details (handles string or array of validation errors).
 */
async function parseErrorMessage(response: Response): Promise<{ message: string; detail?: unknown }> {
  try {
    const body = (await response.json()) as { detail?: unknown };

    if (typeof body.detail === 'string') {
      return { message: body.detail, detail: body.detail };
    }

    if (Array.isArray(body.detail)) {
      const messages = body.detail
        .map((entry) =>
          entry && typeof entry === 'object' && 'msg' in entry
            ? String((entry as { msg: unknown }).msg)
            : null,
        )
        .filter((msg): msg is string => Boolean(msg));

      if (messages.length > 0) {
        return { message: messages.join(' '), detail: body.detail };
      }
    }
    return { message: response.statusText || `Request failed with status ${response.status}`, detail: body };
  } catch {
    // Response was not JSON
    return { message: response.statusText || `Request failed with status ${response.status}` };
  }
}

/**
 * Common bounded fetch wrapper with AbortController and guaranteed resource cleanup.
 */
async function fetchWithTimeout<T>(
  url: string,
  options: RequestInit = {},
  timeoutMs: number,
  timeoutMessage: string,
  consumeResponse: (response: Response) => Promise<T>,
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return await consumeResponse(response);
  } catch (error: unknown) {
    if (controller.signal.aborted) {
      throw new MoistureApiError(timeoutMessage, 408, true);
    }
    if (error instanceof MoistureApiError) throw error;
    if (error instanceof SyntaxError) {
      throw new MoistureApiError('The backend returned an unreadable JSON response.', 502);
    }
    // Network failure (server down, wrong port, no ADB reverse)
    throw new MoistureApiError(
      'Could not reach the backend. Check if FastAPI is running on port 8001 and ADB reverse (adb reverse tcp:8001 tcp:8001) or network route is configured.',
      0,
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Retrieve backend health and quality model status.
 * GET /api/v1/thud-test/health
 */
export async function getAcousticHealth(): Promise<AcousticHealthResponse> {
  const url = `${MOISTURE_API_BASE_URL}${THUD_TEST_PREFIX}/health`;
  return fetchWithTimeout<AcousticHealthResponse>(
    url,
    { method: 'GET', headers: { Accept: 'application/json' } },
    HEALTH_TIMEOUT_MS,
    'Model health check timed out.',
    async (response) => {
      if (!response.ok) {
        const { message, detail } = await parseErrorMessage(response);
        throw new MoistureApiError(message, response.status, false, detail);
      }

      return (await response.json()) as AcousticHealthResponse;
    },
  );
}

/**
 * Retrieve recorder device configuration status.
 * Note: configured=true means recorder address is configured on the backend;
 * it does not confirm live connectivity until capture.
 * GET /api/v1/thud-test/device
 */
export async function getAcousticDeviceStatus(): Promise<AcousticDeviceStatusResponse> {
  const url = `${MOISTURE_API_BASE_URL}${THUD_TEST_PREFIX}/device`;
  return fetchWithTimeout<AcousticDeviceStatusResponse>(
    url,
    { method: 'GET', headers: { Accept: 'application/json' } },
    HEALTH_TIMEOUT_MS,
    'Device status check timed out.',
    async (response) => {
      if (!response.ok) {
        const { message, detail } = await parseErrorMessage(response);
        throw new MoistureApiError(message, response.status, false, detail);
      }

      return (await response.json()) as AcousticDeviceStatusResponse;
    },
  );
}

/**
 * Trigger one fresh ESP32 acoustic recording through FastAPI.
 * POST /api/v1/thud-test/capture
 *
 * Optional query parameters: sheet_id, condition_id.
 * No body is sent. Never automatically retried.
 */
export async function captureMeasurement(
  params?: CaptureQueryParams,
): Promise<AcousticMeasurementResponse> {
  const searchParams = new URLSearchParams();

  if (params?.sheet_id) {
    const trimmedSheetId = params.sheet_id.trim();
    if (trimmedSheetId.length > 0) {
      if (trimmedSheetId.length > 64) {
        throw new MoistureApiError('Sheet ID must be 64 characters or fewer.', 422);
      }
      searchParams.append('sheet_id', trimmedSheetId);
    }
  }

  if (params?.condition_id) {
    const trimmedConditionId = params.condition_id.trim();
    if (trimmedConditionId.length > 0) {
      if (trimmedConditionId.length > 64) {
        throw new MoistureApiError('Condition ID must be 64 characters or fewer.', 422);
      }
      searchParams.append('condition_id', trimmedConditionId);
    }
  }

  const queryString = searchParams.toString();
  const url = `${MOISTURE_API_BASE_URL}${THUD_TEST_PREFIX}/capture${queryString ? `?${queryString}` : ''}`;

  return fetchWithTimeout<AcousticMeasurementResponse>(
    url,
    {
      method: 'POST',
      headers: { Accept: 'application/json' },
    },
    CAPTURE_TIMEOUT_MS,
    'Recording capture timed out after 60 seconds. The strike may have completed and saved on the server. Check saved results or verify device and backend state before trying again.',
    async (response) => {
      if (!response.ok) {
        const { message, detail } = await parseErrorMessage(response);

        if (response.status === 409) {
          throw new MoistureApiError(
            'A recording is already running on the device. Please wait for its result before starting another.',
            409,
            false,
            detail,
          );
        }

        if (response.status === 502) {
          throw new MoistureApiError(
            message || 'Could not communicate with the ESP32 recorder. Check device power and Wi-Fi connection.',
            502,
            false,
            detail,
          );
        }

        if (response.status === 503) {
          throw new MoistureApiError(
            message || 'Recorder device or quality model is not configured on the backend.',
            503,
            false,
            detail,
          );
        }

        throw new MoistureApiError(message, response.status, false, detail);
      }

      return (await response.json()) as AcousticMeasurementResponse;
    },
  );
}

/**
 * Retrieve one saved measurement by UUID.
 * GET /api/v1/thud-test/measurements/{measurement_id}
 */
export async function getMeasurementById(
  measurementId: string,
): Promise<AcousticMeasurementResponse> {
  const trimmedId = measurementId.trim();
  if (!trimmedId) {
    throw new MoistureApiError('Please enter a measurement ID.', 400);
  }

  // Basic UUID format validation to provide friendly immediate feedback
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(trimmedId)) {
    throw new MoistureApiError(
      'Invalid measurement ID format. Please provide a standard 36-character UUID (e.g. 12345678-1234-1234-1234-123456789abc).',
      422,
    );
  }

  const url = `${MOISTURE_API_BASE_URL}${THUD_TEST_PREFIX}/measurements/${encodeURIComponent(trimmedId)}`;
  return fetchWithTimeout<AcousticMeasurementResponse>(
    url,
    { method: 'GET', headers: { Accept: 'application/json' } },
    LOOKUP_TIMEOUT_MS,
    'Measurement lookup timed out.',
    async (response) => {
      if (!response.ok) {
        const { message, detail } = await parseErrorMessage(response);
        if (response.status === 404) {
          throw new MoistureApiError('Measurement was not found. Please verify the ID.', 404, false, detail);
        }
        throw new MoistureApiError(message, response.status, false, detail);
      }

      return (await response.json()) as AcousticMeasurementResponse;
    },
  );
}

/**
 * User-friendly error message formatter.
 */
export function toReadableMoistureError(error: unknown): string {
  if (error instanceof MoistureApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred while communicating with the acoustic service.';
}
