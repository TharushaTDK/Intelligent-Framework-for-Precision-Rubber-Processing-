import { StyleSheet, Text, View } from 'react-native';

import { colors, radii, spacing } from '../../constants/theme';
import type { CoagulationRecommendationResponse } from '../../types/latex';

// same wording as the backend, e.g. "7 hours 48 minutes"
export function hoursAndMinutes(hours: number): string {
  const total = Math.round(hours * 60);
  const h = Math.floor(total / 60);
  const m = total % 60;
  const parts: string[] = [];
  if (h) parts.push(`${h} hour${h !== 1 ? 's' : ''}`);
  if (m || !h) parts.push(`${m} minute${m !== 1 ? 's' : ''}`);
  return parts.join(' ');
}

// a time of zero or less is not a real duration, the saved model value is left as it is
export function coagulationTimeText(hours: number): string {
  return hours > 0 ? hoursAndMinutes(hours) : 'Not reliable for these inputs';
}

export function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export function formatDrc(drc: number): string {
  return `${Number(drc.toFixed(2))} %`;
}

type BatchSummaryCardProps = {
  batch: CoagulationRecommendationResponse;
};

export function BatchSummaryCard({ batch }: BatchSummaryCardProps) {
  return (
    <View style={styles.wrap}>
      <View style={styles.inputCard}>
        <Text style={styles.batchTag}>Batch #{batch.id}</Text>
        <View style={styles.inputRow}>
          <View style={styles.inputItem}>
            <Text style={styles.inputLabel}>Dry rubber content</Text>
            <Text style={styles.inputValue}>{formatDrc(batch.drc)}</Text>
          </View>
          <View style={styles.inputItem}>
            <Text style={styles.inputLabel}>Latex volume</Text>
            <Text style={styles.inputValue}>{batch.latex_volume_L} L</Text>
          </View>
        </View>
      </View>

      {batch.water_acid ? (
        <>
          <View style={styles.resultCard}>
            <Text style={styles.resultLabel}>Water to add</Text>
            <Text style={styles.resultValue}>{batch.water_acid.water_amount_L.toFixed(2)} L</Text>
          </View>
          <View style={styles.resultCard}>
            <Text style={styles.resultLabel}>Prepared 1% formic acid solution</Text>
            <Text style={styles.resultValue}>
              {batch.water_acid.prepared_1pct_formic_acid_solution_mL.toFixed(1)} mL
            </Text>
          </View>
        </>
      ) : null}

      {batch.coagulation_time_hours !== null ? (
        <View style={styles.resultCard}>
          <Text style={styles.resultLabel}>Predicted coagulation time</Text>
          <Text style={styles.resultValue}>{coagulationTimeText(batch.coagulation_time_hours)}</Text>
        </View>
      ) : null}

      {batch.outside_training_range
        ? batch.notes.map((note) => (
            <View key={note} style={styles.warningBox}>
              <Text style={styles.warningText}>{note}</Text>
            </View>
          ))
        : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md,
  },
  inputCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  batchTag: {
    alignSelf: 'flex-start',
    backgroundColor: colors.background,
    color: colors.primary,
    fontSize: 13,
    fontWeight: '700',
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.sm,
    overflow: 'hidden',
  },
  inputRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  inputItem: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  inputValue: {
    marginTop: 2,
    fontSize: 17,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  resultCard: {
    backgroundColor: colors.primary,
    borderRadius: radii.lg,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
  },
  resultLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.primaryLight,
  },
  resultValue: {
    marginTop: spacing.xs,
    fontSize: 26,
    fontWeight: '700',
    color: colors.textOnPrimary,
  },
  warningBox: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.warning,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  warningText: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
});
