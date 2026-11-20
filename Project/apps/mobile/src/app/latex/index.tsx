import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import { formatAcid } from '../../features/latex/BatchSummaryCard';
import {
  ActionTile,
  Banner,
  Card,
  LatexScreen,
  ProgressBar,
  SectionTitle,
  formatClock,
  formatCountdown,
  stageFor,
  tints,
  useNow,
} from '../../features/latex/ui';
import { getAcidStock, listRecommendations, toReadableMessage } from '../../services/latexApi';
import type { AcidStockResponse, CoagulationRecommendationResponse } from '../../types/latex';

function RunningCard({ batch, now }: { batch: CoagulationRecommendationResponse; now: number }) {
  const start = new Date(batch.started_at ?? batch.created_at).getTime();
  const end = new Date(batch.expected_completion_at ?? batch.created_at).getTime();
  const share = end > start ? (now - start) / (end - start) : 1;
  const stage = stageFor(share);
  const due = now >= end;

  return (
    <Pressable
      accessibilityRole="button"
      onPress={() => router.push({ pathname: '/latex/timer', params: { id: String(batch.id) } })}
      style={({ pressed }) => [styles.running, due && styles.runningDue, pressed && styles.pressed]}
    >
      <View style={styles.runningTop}>
        <Text style={styles.runningTitle}>Batch #{batch.id} · {batch.latex_volume_L} L</Text>
        <Text style={[styles.runningStage, due && { color: tints.danger.strong }]}>{stage.label}</Text>
      </View>
      <ProgressBar value={share} color={due ? tints.danger.strong : colors.primary} />
      <View style={styles.runningTop}>
        <Text style={styles.runningMeta}>
          {due ? 'Time reached, check the pan' : `${formatCountdown(end - now)} left`}
        </Text>
        <Text style={styles.runningMeta}>Ready {formatClock(end)}</Text>
      </View>
    </Pressable>
  );
}

export default function LatexHome() {
  const now = useNow();
  const [running, setRunning] = useState<CoagulationRecommendationResponse[]>([]);
  const [stock, setStock] = useState<AcidStockResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // reload whenever the page is shown again
  useFocusEffect(
    useCallback(() => {
      setError(null);
      listRecommendations(20, 'in_progress')
        .then((res) => setRunning(res.items))
        .catch((err) => setError(toReadableMessage(err)));
      getAcidStock()
        .then(setStock)
        .catch(() => setStock(null));
    }, []),
  );

  const stockShare =
    stock?.stock_mL != null && stock.average_batch_acid_mL
      ? stock.stock_mL / (stock.average_batch_acid_mL * 5)
      : 0;

  return (
    <LatexScreen
      eyebrow="Component 2 · Coagulation"
      title="Let's Set the Latex"
      subtitle="Water, acid and coagulation time for every batch, with a guide and a live timer."
    >
      <Card style={styles.startCard}>
        <Text style={styles.startTitle}>Start a new batch</Text>
        <Text style={styles.startText}>Pick the field on the map, enter the latex and get the amounts.</Text>
        <PrimaryButton label="＋  New Batch" onPress={() => router.push('/latex/batch')} />
      </Card>

      {error ? <Banner tone="danger" title="Backend not reached" text={error} /> : null}

      <SectionTitle>Running now</SectionTitle>
      {running.length === 0 ? (
        <Card>
          <Text style={styles.emptyText}>No batch is coagulating. Start one from the guide after a recommendation.</Text>
        </Card>
      ) : (
        running.map((batch) => <RunningCard key={batch.id} batch={batch} now={now} />)
      )}

      <SectionTitle>Acid stock</SectionTitle>
      <Pressable accessibilityRole="button" onPress={() => router.push('/latex/stock')}>
        <Card style={stock?.low_stock ? styles.stockLow : undefined}>
          <View style={styles.runningTop}>
            <Text style={styles.stockValue}>
              🧪 {stock?.stock_mL != null ? formatAcid(stock.stock_mL) : 'Not entered'}
            </Text>
            <Text style={styles.link}>Update ›</Text>
          </View>
          {stock?.stock_mL != null ? (
            <ProgressBar value={stockShare} color={stock.low_stock ? tints.danger.strong : tints.acid.strong} />
          ) : null}
          <Text style={[styles.stockMessage, stock?.low_stock && { color: tints.danger.strong }]}>
            {stock?.message ?? 'Prepared 1% formic acid solution on hand.'}
          </Text>
        </Card>
      </Pressable>

      <SectionTitle>Tools</SectionTitle>
      <View style={styles.grid}>
        <ActionTile
          icon="🎚"
          tint="water"
          title="What-if"
          caption="Slide the volume, see the amounts change"
          onPress={() => router.push('/latex/whatif')}
        />
        <ActionTile
          icon="📋"
          tint="time"
          title="History"
          caption="All batches and their results"
          onPress={() => router.push('/latex/history')}
        />
      </View>

      <Text style={styles.notice}>Decision-support prototype. Always follow the approved processing procedure.</Text>
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  startCard: {
    gap: spacing.sm,
  },
  startTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  startText: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: spacing.xs,
  },
  running: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  runningDue: {
    borderColor: tints.danger.strong,
    backgroundColor: tints.danger.soft,
  },
  pressed: {
    opacity: 0.85,
  },
  runningTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: spacing.sm,
  },
  runningTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  runningStage: {
    fontSize: 13,
    fontWeight: '800',
    color: colors.primary,
  },
  runningMeta: {
    fontSize: 12,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  emptyText: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  stockLow: {
    borderColor: tints.danger.strong,
  },
  stockValue: {
    fontSize: 20,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  stockMessage: {
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  link: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primary,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  notice: {
    textAlign: 'center',
    fontSize: 11,
    color: colors.textSecondary,
    marginTop: spacing.sm,
  },
});
