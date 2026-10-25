import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, spacing } from '../../constants/theme';
import { BatchSummaryCard, formatAcid } from '../../features/latex/BatchSummaryCard';
import { Banner, LatexScreen } from '../../features/latex/ui';
import { addCoagulationTime, getAcidStock, getRecommendation, toReadableMessage } from '../../services/latexApi';
import type { AcidStockResponse, CoagulationRecommendationResponse } from '../../types/latex';

export default function BatchResult() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const [batch, setBatch] = useState<CoagulationRecommendationResponse | null>(null);
  const [stock, setStock] = useState<AcidStockResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addingTime, setAddingTime] = useState(false);

  useEffect(() => {
    getRecommendation(Number(id))
      .then(setBatch)
      .catch((err) => setError(toReadableMessage(err)))
      .finally(() => setLoading(false));
    getAcidStock()
      .then(setStock)
      .catch(() => setStock(null));
  }, [id]);

  async function handleAddTime() {
    if (!batch) return;
    setAddingTime(true);
    setError(null);
    try {
      setBatch(await addCoagulationTime(batch.id));
    } catch (err) {
      setError(toReadableMessage(err));
    } finally {
      setAddingTime(false);
    }
  }

  const needed = batch ? Math.round(batch.water_acid?.prepared_1pct_formic_acid_solution_mL ?? 0) : 0;
  const shortOfAcid = stock?.stock_mL != null && needed > 0 && stock.stock_mL < needed;

  return (
    <LatexScreen
      eyebrow="Step 2 of 3"
      title="Recommendation"
      subtitle="Saved as a batch. Follow the guide to prepare it and start the timer."
      footer={
        batch && batch.status === 'recommended' ? (
          <PrimaryButton
            label="Start Step-by-Step Guide  ›"
            onPress={() => router.push({ pathname: '/latex/guide', params: { id: String(batch.id) } })}
          />
        ) : null
      }
    >
      {loading ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator color={colors.primary} />
          <Text style={styles.loadingText}>Loading the batch...</Text>
        </View>
      ) : null}

      {batch ? <BatchSummaryCard batch={batch} /> : null}

      {batch?.outside_training_range ? (
        <Banner tone="warn" title="Outside the trained range" text={batch.notes.join(' ')} />
      ) : null}

      {shortOfAcid ? (
        <Banner
          tone="danger"
          title="Not enough acid solution"
          text={`This batch needs ${formatAcid(needed)} but the stock has ${formatAcid(stock.stock_mL ?? 0)}. Prepare more and update the stock.`}
        />
      ) : stock?.stock_mL != null && needed > 0 ? (
        <Banner
          tone="info"
          title="Acid stock is enough"
          text={`${formatAcid(stock.stock_mL)} in stock, ${formatAcid(stock.stock_mL - needed)} left after this batch.`}
        />
      ) : null}

      {error ? <Banner tone="danger" text={error} /> : null}

      {batch && batch.coagulation_time_hours === null ? (
        addingTime ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Predicting the coagulation time...</Text>
          </View>
        ) : (
          <SecondaryButton label="Add Coagulation Time" onPress={handleAddTime} />
        )
      ) : null}

      {batch ? (
        <>
          {shortOfAcid ? <SecondaryButton label="Update Acid Stock" onPress={() => router.push('/latex/stock')} /> : null}
          <SecondaryButton
            label="Try Other Volumes (What-if)"
            onPress={() =>
              router.push({
                pathname: '/latex/whatif',
                params: { drc: String(batch.drc), volume: String(batch.latex_volume_L) },
              })
            }
          />
          <SecondaryButton label="Done" onPress={() => router.dismissTo('/latex')} />
        </>
      ) : null}
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  loadingText: {
    color: colors.textSecondary,
    fontSize: 13,
  },
});
