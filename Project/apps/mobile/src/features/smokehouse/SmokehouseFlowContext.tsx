import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import type {
  LayerCode,
  SmokehouseBatchResponse,
  SmokehouseReadingPredictionResponse,
} from '../../types/smokehouse';
import type { VerificationResult } from './verifyPersistedReading';

interface SmokehouseFlowContextValue {
  batch: SmokehouseBatchResponse | null;
  setBatch: (batch: SmokehouseBatchResponse) => void;

  selectedLayer: LayerCode | null;
  setSelectedLayer: (layer: LayerCode) => void;

  lastReading: SmokehouseReadingPredictionResponse | null;
  verification: VerificationResult | null;
  setLastReading: (
    reading: SmokehouseReadingPredictionResponse,
    verification: VerificationResult,
  ) => void;
}

const SmokehouseFlowContext = createContext<SmokehouseFlowContextValue | undefined>(
  undefined,
);

export function SmokehouseFlowProvider({ children }: { children: ReactNode }) {
  const [batch, setBatch] = useState<SmokehouseBatchResponse | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<LayerCode | null>(null);
  const [lastReading, setLastReadingState] =
    useState<SmokehouseReadingPredictionResponse | null>(null);
  const [verification, setVerification] = useState<VerificationResult | null>(null);

  const value = useMemo<SmokehouseFlowContextValue>(
    () => ({
      batch,
      setBatch,
      selectedLayer,
      setSelectedLayer,
      lastReading,
      verification,
      setLastReading: (reading, nextVerification) => {
        setLastReadingState(reading);
        setVerification(nextVerification);
      },
    }),
    [batch, selectedLayer, lastReading, verification],
  );

  return (
    <SmokehouseFlowContext.Provider value={value}>
      {children}
    </SmokehouseFlowContext.Provider>
  );
}

export function useSmokehouseFlow(): SmokehouseFlowContextValue {
  const context = useContext(SmokehouseFlowContext);

  if (!context) {
    throw new Error(
      'useSmokehouseFlow must be used within a SmokehouseFlowProvider',
    );
  }

  return context;
}
