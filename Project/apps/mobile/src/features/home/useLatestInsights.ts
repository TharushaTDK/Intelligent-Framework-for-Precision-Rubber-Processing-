import { useCallback, useEffect, useRef, useState } from 'react';

import { colors } from '../../constants/theme';
import { KNOWN_SMOKEHOUSE_IDS } from '../../constants/knownSmokehouses';
import { listRecommendations } from '../../services/latexApi';
import { getAcousticDeviceStatus, getAcousticHealth } from '../../services/moistureApi';
import { PLANT_ADVISORY_API_BASE_URL } from '../../services/plantAdvisoryApi';
import { getSmokehouseLatestStatus, listSmokehouseBatches } from '../../services/smokehouseApi';
import type { AdvisoryRecordResponse } from '../../types/plantAdvisory';
import type { LayerCode, SmokehousePredictionResponse } from '../../types/smokehouse';

/**
 * Read-only summary of the most recent result from each component,
 * shown on the Home screen. It only calls existing GET endpoints and
 * never creates or changes anything on the backend.
 */

export type InsightKey = 'plant' | 'latex' | 'smokehouse' | 'moisture';

export type Insight =
  | { status: 'loading' }
  | { status: 'empty' }
  | { status: 'offline' }
  | {
      status: 'ready';
      value: string;
      caption: string;
      detail: string;
      when: string | null;
      tone: string;
    };

export type Insights = Record<InsightKey, Insight>;

const LOADING: Insights = {
  plant: { status: 'loading' },
  latex: { status: 'loading' },
  smokehouse: { status: 'loading' },
  moisture: { status: 'loading' },
};

const REQUEST_TIMEOUT_MS = 12000;
const LAYERS: LayerCode[] = ['L1', 'L2', 'L3'];

function withTimeout<T>(promise: Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout')), REQUEST_TIMEOUT_MS);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function fetchLatestAdvisory(): Promise<AdvisoryRecordResponse | null> {
  const response = await fetch(
    `${PLANT_ADVISORY_API_BASE_URL}/api/v1/weather-tapping/advisory-records?limit=1`,
  );
  if (!response.ok) {
    throw new Error(`status ${response.status}`);
  }
  const body = (await response.json()) as { records?: AdvisoryRecordResponse[] };
  return body.records?.[0] ?? null;
}

export function formatRelativeTime(iso: string | null | undefined): string | null {
  if (!iso) {
    return null;
  }
  const time = new Date(iso).getTime();
  if (Number.isNaN(time)) {
    return null;
  }
  const minutes = Math.round((Date.now() - time) / 60000);
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days} day${days === 1 ? '' : 's'} ago`;
  return new Date(time).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

function tappingTone(label: string | null | undefined): string {
  if (label === 'High') return colors.success;
  if (label === 'Moderate') return colors.warning;
  if (label === 'Low') return colors.critical;
  return colors.primary;
}

function fireTone(label: string): string {
  if (label === 'Critical') return colors.critical;
  if (label === 'Warning') return colors.warning;
  return colors.success;
}

function plantInsight(record: AdvisoryRecordResponse | null): Insight {
  if (!record) {
    return { status: 'empty' };
  }
  const label = record.tapping_label ?? null;
  const parts = [`DRC ${record.predicted_latex_drc_pct.toFixed(1)}%`];
  if (record.predicted_price_lkr_per_kg != null) {
    parts.push(`Rs ${Math.round(record.predicted_price_lkr_per_kg)}/kg`);
  }
  return {
    status: 'ready',
    value: label ? `${label}` : 'Saved',
    caption: label ? 'Tapping suitability' : 'Advisory saved',
    detail: parts.join(' · '),
    when: formatRelativeTime(record.created_at),
    tone: tappingTone(label),
  };
}

async function loadLatex(): Promise<Insight> {
  const history = await listRecommendations(1);
  const latest = history.items[0];
  if (!latest) {
    return { status: 'empty' };
  }
  const parts: string[] = [];
  if (latest.water_acid) {
    parts.push(`${Math.round(latest.water_acid.prepared_1pct_formic_acid_solution_mL)} mL acid`);
  }
  if (latest.coagulation_time_hours != null) {
    parts.push(`${latest.coagulation_time_hours.toFixed(1)} h to set`);
  }
  if (parts.length === 0) {
    parts.push(`${latest.latex_volume_L} L latex · DRC ${latest.drc.toFixed(1)}%`);
  }
  return {
    status: 'ready',
    value: latest.water_acid ? `${latest.water_acid.water_amount_L.toFixed(1)} L` : `${latest.latex_volume_L} L`,
    caption: latest.water_acid ? 'Water to add' : 'Latex batch',
    detail: parts.join(' · '),
    when: formatRelativeTime(latest.created_at),
    tone: latest.outside_training_range ? colors.warning : colors.success,
  };
}

async function loadSmokehouse(): Promise<Insight> {
  const lists = await Promise.allSettled(KNOWN_SMOKEHOUSE_IDS.map((id) => listSmokehouseBatches(id)));
  const fulfilled = lists.filter(
    (result): result is PromiseFulfilledResult<Awaited<ReturnType<typeof listSmokehouseBatches>>> =>
      result.status === 'fulfilled',
  );
  if (fulfilled.length === 0) {
    throw new Error('smokehouse unreachable');
  }
  const batches = fulfilled
    .flatMap((result) => result.value.batches)
    .sort((a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime());
  const batch = batches.find((item) => item.status === 'active') ?? batches[0];
  if (!batch) {
    return { status: 'empty' };
  }

  const statuses = await Promise.allSettled(
    LAYERS.map((layer) => getSmokehouseLatestStatus(batch.batch_code, layer)),
  );
  let latest: SmokehousePredictionResponse | null = null;
  for (const result of statuses) {
    const prediction = result.status === 'fulfilled' ? result.value.latest_prediction : null;
    if (
      prediction &&
      (!latest || new Date(prediction.predicted_at).getTime() > new Date(latest.predicted_at).getTime())
    ) {
      latest = prediction;
    }
  }

  if (!latest) {
    return {
      status: 'ready',
      value: batch.status === 'active' ? 'Active' : 'Completed',
      caption: `Batch ${batch.batch_code}`,
      detail: `${batch.smokehouse_id} · no readings yet`,
      when: formatRelativeTime(batch.started_at),
      tone: colors.primary,
    };
  }

  return {
    status: 'ready',
    value: latest.fire_risk_label,
    caption: 'Fire risk',
    detail: `~${latest.remaining_drying_time_hours.toFixed(1)} h drying left · ${batch.smokehouse_id}`,
    when: formatRelativeTime(latest.predicted_at),
    tone: fireTone(latest.fire_risk_label),
  };
}

async function loadMoisture(): Promise<Insight> {
  const [health, device] = await Promise.allSettled([getAcousticHealth(), getAcousticDeviceStatus()]);
  if (health.status !== 'fulfilled') {
    throw new Error('acoustic service unreachable');
  }
  const deviceReady = device.status === 'fulfilled' && device.value.configured;
  return {
    status: 'ready',
    value: health.value.model_available ? 'Ready' : 'No model',
    caption: 'Acoustic checker',
    detail: deviceReady ? 'Capture device connected' : 'Capture device not connected',
    when: null,
    tone: health.value.model_available ? (deviceReady ? colors.success : colors.warning) : colors.critical,
  };
}

function settle(promise: Promise<Insight>): Promise<Insight> {
  return withTimeout(promise).catch((): Insight => ({ status: 'offline' }));
}

export function useLatestInsights() {
  const [insights, setInsights] = useState<Insights>(LOADING);
  const [forecast, setForecast] = useState<AdvisoryRecordResponse | null | undefined>(undefined);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    setRefreshing(true);

    const advisoryPromise = withTimeout(fetchLatestAdvisory());
    const [plant, latex, smokehouse, moisture] = await Promise.all([
      advisoryPromise.then(plantInsight).catch((): Insight => ({ status: 'offline' })),
      settle(loadLatex()),
      settle(loadSmokehouse()),
      settle(loadMoisture()),
    ]);
    const advisory = await advisoryPromise.catch(() => null);

    if (!mounted.current) {
      return;
    }
    setInsights({ plant, latex, smokehouse, moisture });
    setForecast(advisory);
    setUpdatedAt(new Date());
    setRefreshing(false);
  }, []);

  return { insights, forecast, updatedAt, refreshing, refresh };
}
