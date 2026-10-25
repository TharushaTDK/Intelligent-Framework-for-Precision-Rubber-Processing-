import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, spacing } from '../../constants/theme';
import {
  BatchSummaryCard,
  coagulationTimeText,
  formatAcid,
  formatDate,
} from '../../features/latex/BatchSummaryCard';
import { Banner, Card, LatexScreen } from '../../features/latex/ui';
import { getRecommendation, toReadableMessage } from '../../services/latexApi';
import type { CoagulationRecommendationResponse } from '../../types/latex';

const OUTPUT_LABELS = {
  water_acid: 'Water & Acid',
  coagulation_time: 'Coagulation Time',
};

export default function BatchDetails() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const [batch, setBatch] = useState<CoagulationRecommendationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRecommendation(Number(id))
      .then(setBatch)
      .catch((err) => setError(toReadableMessage(err)))
      .finally(() => setLoading(false));
  }, [id]);

  const timeline = batch
    ? [
        { label: 'Recommended', at: batch.created_at },
        { label: 'Timer started', at: batch.started_at },
        {
          label: batch.status === 'cancelled' ? 'Cancelled' : 'Completed',
          at: batch.completed_at ?? (batch.status === 'cancelled' ? 'cancelled' : null),
        },
      ]
    : [];

  return (
    <LatexScreen
      eyebrow={batch ? `Batch #${batch.id}` : 'Batch'}
      title="Batch Details"
      footer={
        batch?.status === 'recommended' ? (
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

      {error ? <Banner tone="danger" text={error} /> : null}

      {batch ? (
        <>
          <BatchSummaryCard batch={batch} />

          <Card>
            <Text style={styles.cardTitle}>Timeline</Text>
            {timeline.map((item, index) => {
              const reached = Boolean(item.at);
              return (
                <View key={item.label} style={styles.timelineRow}>
                  <View style={styles.timelineSide}>
                    <View style={[styles.dot, reached && styles.dotOn]} />
                    {index < timeline.length - 1 ? <View style={[styles.line, reached && styles.lineOn]} /> : null}
                  </View>
                  <View style={styles.timelineText}>
                    <Text style={[styles.timelineLabel, !reached && styles.muted]}>{item.label}</Text>
                    <Text style={styles.timelineAt}>
                      {item.at && item.at !== 'cancelled' ? formatDate(item.at) : reached ? '' : 'Not yet'}
                    </Text>
                  </View>
                </View>
              );
            })}
          </Card>

          {batch.actual_coagulation_time_hours !== null || batch.acid_used_mL !== null ? (
            <Card>
              <Text style={styles.cardTitle}>Result</Text>
              {batch.acid_used_mL !== null ? (
                <InfoRow label="Acid taken from stock" value={formatAcid(batch.acid_used_mL)} />
              ) : null}
              {batch.coagulation_time_hours !== null ? (
                <InfoRow label="Predicted time" value={coagulationTimeText(batch.coagulation_time_hours)} />
              ) : null}
              {batch.actual_coagulation_time_hours !== null ? (
                <InfoRow label="Actual time" value={coagulationTimeText(batch.actual_coagulation_time_hours)} />
              ) : null}
            </Card>
          ) : null}

          <Card>
            <InfoRow label="Asked for" value={batch.outputs.map((output) => OUTPUT_LABELS[output]).join(' and ')} />
            <InfoRow
              label="DRC source"
              value={batch.drc_source === 'manual' ? 'Entered manually' : 'From the chosen location'}
            />
            <InfoRow label="Water & acid model" value={batch.water_acid_model_version} />
            {batch.coagulation_time_model_version ? (
              <InfoRow label="Coagulation time model" value={batch.coagulation_time_model_version} />
            ) : null}
          </Card>
        </>
      ) : null}
    </LatexScreen>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  cardTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  timelineRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  timelineSide: {
    alignItems: 'center',
    width: 16,
  },
  dot: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  dotOn: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  line: {
    flex: 1,
    width: 2,
    minHeight: 22,
    backgroundColor: colors.border,
  },
  lineOn: {
    backgroundColor: colors.primary,
  },
  timelineText: {
    flex: 1,
    paddingBottom: spacing.sm,
  },
  timelineLabel: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  muted: {
    color: colors.textSecondary,
  },
  timelineAt: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: spacing.md,
  },
  infoLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  infoValue: {
    flexShrink: 1,
    fontSize: 13,
    fontWeight: '700',
    color: colors.textPrimary,
    textAlign: 'right',
  },
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
