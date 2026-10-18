import type {
  LayerCode,
  SmokehouseBatchCreate,
  SmokehouseBatchListResponse,
  SmokehouseBatchResponse,
  SmokehouseLatestStatusResponse,
  SmokehouseLayerHistoryResponse,
  SmokehouseReadingCreate,
  SmokehouseReadingPredictionResponse,
} from '../types/smokehouse';

/**
 * Development-only API base URL.
 *
 * The Android emulator reaches the host FastAPI backend through:
 *   adb -s emulator-5554 reverse tcp:8000 tcp:8000
 * which forwards the emulator's own 127.0.0.1:8000 to the host's
 * 127.0.0.1:8000. This must be re-created whenever the emulator
 * or ADB server restarts.
 */
export const SMOKEHOUSE_API_BASE_URL = 'http://127.0.0.1:8000';

const SMOKEHOUSE_PREFIX = '/api/v1/smokehouse';

export class SmokehouseApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'SmokehouseApiError';
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
    // Response body was not JSON; fall through to the status text.
  }

  return response.statusText || `Request failed with status ${response.status}`;
}

async function request<TResponse>(
  path: string,
  init?: RequestInit,
): Promise<TResponse> {
  const response = await fetch(`${SMOKEHOUSE_API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response);
    throw new SmokehouseApiError(message, response.status);
  }

  return (await response.json()) as TResponse;
}

// ============================================================
// BATCH
// ============================================================

export function createSmokehouseBatch(
  payload: SmokehouseBatchCreate,
): Promise<SmokehouseBatchResponse> {
  return request<SmokehouseBatchResponse>(`${SMOKEHOUSE_PREFIX}/batches`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// ============================================================
// READING + ML INFERENCE
// ============================================================

export function createSmokehouseReading(
  payload: SmokehouseReadingCreate,
): Promise<SmokehouseReadingPredictionResponse> {
  return request<SmokehouseReadingPredictionResponse>(
    `${SMOKEHOUSE_PREFIX}/readings`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
  );
}

// ============================================================
// BATCHES FOR A SMOKEHOUSE
// ============================================================

export function listSmokehouseBatches(
  smokehouseId: string,
): Promise<SmokehouseBatchListResponse> {
  return request<SmokehouseBatchListResponse>(
    `${SMOKEHOUSE_PREFIX}/smokehouses/${encodeURIComponent(smokehouseId)}/batches`,
  );
}

// ============================================================
// LATEST STATUS
// ============================================================

export function getSmokehouseLatestStatus(
  batchCode: string,
  layerCode: LayerCode,
): Promise<SmokehouseLatestStatusResponse> {
  return request<SmokehouseLatestStatusResponse>(
    `${SMOKEHOUSE_PREFIX}/batches/${encodeURIComponent(
      batchCode,
    )}/layers/${encodeURIComponent(layerCode)}/latest`,
  );
}

// ============================================================
// HISTORY (contract only — not used in UI yet)
// ============================================================

export function getSmokehouseLayerHistory(
  batchCode: string,
  layerCode: LayerCode,
): Promise<SmokehouseLayerHistoryResponse> {
  return request<SmokehouseLayerHistoryResponse>(
    `${SMOKEHOUSE_PREFIX}/batches/${encodeURIComponent(
      batchCode,
    )}/layers/${encodeURIComponent(layerCode)}/history`,
  );
}
