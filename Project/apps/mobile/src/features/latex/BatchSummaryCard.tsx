import { StyleSheet, Text, View } from 'react-native';

import { colors, radii, spacing } from '../../constants/theme';
import { Card, Chip, StatTile } from './ui';
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
  return `${drc.toFixed(1)} %`;
}

// measured with a jug and a cylinder, so water to 0.1 L and acid to 1 mL
export function formatWater(litres: number): string {
  return `${litres.toFixed(1)} L`;
}

export function formatAcid(millilitres: number): string {
  return `${millilitres.toFixed(0)} mL`;
}

type BatchSummaryCardProps = {
  batch: CoagulationRecommendationResponse;
};

export const STATUS_LABELS = {
  recommended: { label: 'Not started', tint: 'warn' },
  in_progress: { label: 'Coagulating', tint: 'water' },
  completed: { label: 'Completed', tint: 'time' },
  cancelled: { label: 'Cancelled', tint: 'danger' },
} as const;

export function BatchSummaryCard({ batch }: BatchSummaryCardProps) {
  const status = STATUS_LABELS[batch.status];
  return (
    <View style={styles.wrap}>
      <Card>
        <View style={styles.headRow}>
          <Text style={styles.batchTag}>Batch #{batch.id}</Text>
          <Chip label={status.label} tint={status.tint} />
        </View>
        <Text style={styles.locationText}>📍 Location {batch.estate_id}</Text>
        <View style={styles.inputRow}>
          <View style={styles.inputItem}>
            <Text style={styles.inputLabel}>Dry rubber content</Text>
            <Text style={styles.inputValue}>{formatDrc(batch.drc)}</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.inputItem}>
            <Text style={styles.inputLabel}>Latex volume</Text>
            <Text style={styles.inputValue}>{batch.latex_volume_L} L</Text>
          </View>
        </View>
      </Card>

      {batch.water_acid ? (
        <View style={styles.tiles}>
          <StatTile icon="💧" tint="water" label="Water to add" value={formatWater(batch.water_acid.water_amount_L)} />
          <StatTile
            icon="🧪"
            tint="acid"
            label="1% acid solution"
            value={formatAcid(batch.water_acid.prepared_1pct_formic_acid_solution_mL)}
          />
        </View>
      ) : null}

      {batch.coagulation_time_hours !== null ? (
        <View style={styles.timeCard}>
          <Text style={styles.timeIcon}>⏱</Text>
          <View style={styles.timeText}>
            <Text style={styles.timeLabel}>Predicted coagulation time</Text>
            <Text style={styles.timeValue}>{coagulationTimeText(batch.coagulation_time_hours)}</Text>
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.md,
  },
  headRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  batchTag: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  locationText: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background,
    borderRadius: radii.md,
    padding: spacing.md,
    marginTop: spacing.xs,
  },
  inputItem: {
    flex: 1,
    alignItems: 'center',
  },
  divider: {
    width: 1,
    alignSelf: 'stretch',
    backgroundColor: colors.border,
  },
  inputLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  inputValue: {
    marginTop: 2,
    fontSize: 20,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  tiles: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  timeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.primary,
    borderRadius: radii.lg,
    padding: spacing.md,
  },
  timeIcon: {
    fontSize: 30,
  },
  timeText: {
    flex: 1,
  },
  timeLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.primaryLight,
  },
  timeValue: {
    marginTop: 2,
    fontSize: 24,
    fontWeight: '800',
    color: colors.textOnPrimary,
  },
});
