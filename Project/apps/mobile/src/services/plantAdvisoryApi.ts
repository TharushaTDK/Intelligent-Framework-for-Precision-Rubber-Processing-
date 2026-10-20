import type {
  AdvisoryRecordCreate,
  AdvisoryRecordResponse,
  DrcPredictionRequest,
  DrcPredictionResponse,
  ModelCatalogResponse,
  PriceEstimationRequest,
  PriceEstimationResponse,
  TappingSuitabilityRequest,
  TappingSuitabilityResponse,
  WeatherSnapshotRequest,
  WeatherSnapshotResponse,
} from '../types/plantAdvisory';

// emulator reaches the backend through: adb reverse tcp:8000 tcp:8000
export const PLANT_ADVISORY_API_BASE_URL = 'http://127.0.0.1:8000';

const WEATHER_TAPPING_PREFIX = '/api/v1/weather-tapping';

// the weather call can retry the provider a few times on the backend,
// so give it room, but don't let a stuck request spin forever
const REQUEST_TIMEOUT_MS = 30000;

export class PlantAdvisoryApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'PlantAdvisoryApiError';
    this.status = status;
  }
}

// FastAPI sends either a plain string or a list of validation errors
async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };

    if (typeof body.detail === 'string') {
      return body.detail;
    }

    if (Array.isArray(body.detail)) {
      const messages = body.detail
        .map((entry) => {
          if (!entry || typeof entry !== 'object' || !('msg' in entry)) {
            return null;
          }
          const loc = 'loc' in entry && Array.isArray(entry.loc)
            ? entry.loc.filter((part: unknown) => part !== 'body').join('.')
            : '';
          const msg = String((entry as { msg: unknown }).msg);
          return loc ? `${loc}: ${msg}` : msg;
        })
        .filter((msg): msg is string => Boolean(msg));

      if (messages.length > 0) {
        return messages.join('\n');
      }
    }
  } catch {
    // body was not JSON, use the status text below
  }

  return response.statusText || `Request failed with status ${response.status}`;
}

async function request<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${PLANT_ADVISORY_API_BASE_URL}${WEATHER_TAPPING_PREFIX}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...init?.headers,
      },
    });

    if (!response.ok) {
      throw new PlantAdvisoryApiError(await parseErrorMessage(response), response.status);
    }

    return (await response.json()) as TResponse;
  } finally {
    clearTimeout(timer);
  }
}

function post<TResponse>(path: string, body: unknown): Promise<TResponse> {
  return request<TResponse>(path, { method: 'POST', body: JSON.stringify(body) });
}

// ---------------- endpoints ----------------

export function getModelCatalog(): Promise<ModelCatalogResponse> {
  return request<ModelCatalogResponse>('/models');
}

export function getWeatherSnapshot(payload: WeatherSnapshotRequest): Promise<WeatherSnapshotResponse> {
  return post<WeatherSnapshotResponse>('/weather-snapshot', payload);
}

export function estimateDrc(payload: DrcPredictionRequest): Promise<DrcPredictionResponse> {
  return post<DrcPredictionResponse>('/drc-estimation', payload);
}

export function getTappingAdvisory(payload: TappingSuitabilityRequest): Promise<TappingSuitabilityResponse> {
  return post<TappingSuitabilityResponse>('/tapping-advisory', payload);
}

export function estimatePrice(payload: PriceEstimationRequest): Promise<PriceEstimationResponse> {
  return post<PriceEstimationResponse>('/price-estimation', payload);
}

// saves an advisory so Component 2 can use the DRC and location
export function saveAdvisoryRecord(payload: AdvisoryRecordCreate): Promise<AdvisoryRecordResponse> {
  return post<AdvisoryRecordResponse>('/advisory-records', payload);
}

export function toReadableMessage(error: unknown): string {
  if (error instanceof PlantAdvisoryApiError) {
    return error.message;
  }
  if (error instanceof Error && error.name === 'AbortError') {
    return 'The backend took too long to respond. Please try again.';
  }
  return 'Could not reach the backend. Please check it is running and try again.';
}
