import type {
  AcidStockResponse,
  CoagulationHistoryResponse,
  CoagulationPreviewRequest,
  CoagulationPreviewResponse,
  CoagulationRecommendationCreate,
  CoagulationRecommendationResponse,
  LocationDrcResponse,
  RecommendationStatus,
} from '../types/latex';

// emulator reaches the backend through: adb reverse tcp:8000 tcp:8000
export const LATEX_API_BASE_URL = 'http://127.0.0.1:8000';

const COAGULATION_PREFIX = '/api/v1/coagulation';

export class LatexApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'LatexApiError';
    this.status = status;
  }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };

    if (typeof body.detail === 'string') {
      return body.detail;
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
        return messages.join(' ');
      }
    }
  } catch {
    // body was not JSON, use the status text below
  }

  return response.statusText || `Request failed with status ${response.status}`;
}

async function request<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const response = await fetch(`${LATEX_API_BASE_URL}${COAGULATION_PREFIX}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new LatexApiError(await parseErrorMessage(response), response.status);
  }

  return (await response.json()) as TResponse;
}

export function createRecommendation(
  payload: CoagulationRecommendationCreate,
): Promise<CoagulationRecommendationResponse> {
  return request<CoagulationRecommendationResponse>('/recommendations', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// DRC for a map point, worked out from the weather there now
export function getLocationDrc(latitude: number, longitude: number): Promise<LocationDrcResponse> {
  return request<LocationDrcResponse>('/location-drc', {
    method: 'POST',
    body: JSON.stringify({ latitude, longitude }),
  });
}

export function getRecommendation(id: number): Promise<CoagulationRecommendationResponse> {
  return request<CoagulationRecommendationResponse>(`/recommendations/${id}`);
}

// batches from every location, newest first
export function listRecommendations(
  limit = 50,
  status?: RecommendationStatus,
): Promise<CoagulationHistoryResponse> {
  const filter = status ? `&status=${status}` : '';
  return request<CoagulationHistoryResponse>(`/recommendations?limit=${limit}${filter}`);
}

// start, complete or cancel the batch timer
export function updateBatchStatus(
  id: number,
  status: 'in_progress' | 'completed' | 'cancelled',
): Promise<CoagulationRecommendationResponse> {
  return request<CoagulationRecommendationResponse>(`/recommendations/${id}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}

export function getAcidStock(): Promise<AcidStockResponse> {
  return request<AcidStockResponse>('/acid-stock');
}

// farmer prepared more solution, added on top of the stock
export function addAcidStock(addedMl: number): Promise<AcidStockResponse> {
  return request<AcidStockResponse>('/acid-stock/add', {
    method: 'POST',
    body: JSON.stringify({ added_mL: addedMl }),
  });
}

// corrects the total, e.g. after a spill
export function setAcidStock(stockMl: number): Promise<AcidStockResponse> {
  return request<AcidStockResponse>('/acid-stock', {
    method: 'PUT',
    body: JSON.stringify({ stock_mL: stockMl }),
  });
}

// what-if check, nothing is saved
export function previewRecommendation(payload: CoagulationPreviewRequest): Promise<CoagulationPreviewResponse> {
  return request<CoagulationPreviewResponse>('/predict-preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// adds the time to a batch that was first asked only for water and acid
export function addCoagulationTime(id: number): Promise<CoagulationRecommendationResponse> {
  return request<CoagulationRecommendationResponse>(`/recommendations/${id}/coagulation-time`, {
    method: 'POST',
  });
}

export function toReadableMessage(error: unknown): string {
  if (error instanceof LatexApiError) {
    return error.message;
  }
  return 'Could not reach the backend. Please check it is running and try again.';
}
