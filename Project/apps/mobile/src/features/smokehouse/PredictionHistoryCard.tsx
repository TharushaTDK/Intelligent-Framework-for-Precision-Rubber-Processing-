import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, radii, shadows, spacing, typography } from '../../constants/theme';
import type { FireRiskLabel, SmokehouseHistoryItemResponse } from '../../types/smokehouse';

function fireRiskColor(label: FireRiskLabel): string {
  if (label === 'Normal') return colors.success;
  if (label === 'Warning') return colors.warning;
  return colors.critical;
}

function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

type PredictionHistoryCardProps = {
  item: SmokehouseHistoryItemResponse;
};

export function PredictionHistoryCard({ item }: PredictionHistoryCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { reading, prediction } = item;

  return (
    <View style={styles.card}>
      <Text style={styles.timestamp}>{formatTimestamp(reading.recorded_at)}</Text>

      <View style={styles.metricsRow}>
        <View style={styles.metric}>
          <Text style={styles.metricLabel}>Temperature</Text>
          <Text style={styles.metricValue}>{reading.temperature_c} °C</Text>
        </View>
        <View style={styles.metric}>
          <Text style={styles.metricLabel}>Humidity</Text>
          <Text style={styles.metricValue}>{reading.humidity_pct}%</Text>
        </View>
      </View>

      {!prediction ? (
        <Text style={styles.unavailableText}>
          Prediction unavailable for this reading.
        </Text>
      ) : (
        <>
          <View style={styles.metricsRow}>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Remaining</Text>
              <Text style={styles.metricValue}>
                {prediction.remaining_drying_time_hours.toFixed(2)} h
              </Text>
            </View>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Risk</Text>
              <View
                style={[
                  styles.riskBadge,
                  { backgroundColor: fireRiskColor(prediction.fire_risk_label) },
                ]}
              >
                <Text style={styles.riskBadgeText}>
                  {prediction.fire_risk_label}
                </Text>
              </View>
            </View>
          </View>

          <Pressable
            accessibilityRole="button"
            onPress={() => setExpanded((value) => !value)}
            style={styles.detailsToggle}
          >
            <Text style={styles.detailsToggleText}>View Details</Text>
            <Text style={styles.detailsToggleIcon}>{expanded ? '▾' : '▸'}</Text>
          </Pressable>

          {expanded ? (
            <View style={styles.detailsBox}>
              <Text style={styles.detailsSubtitle}>
                {prediction.fire_risk_label} fire-risk condition
              </Text>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Normal Probability</Text>
                <Text style={styles.detailValue}>
                  {formatPercent(prediction.probabilities.normal)}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Warning Probability</Text>
                <Text style={styles.detailValue}>
                  {formatPercent(prediction.probabilities.warning)}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Critical Probability</Text>
                <Text style={styles.detailValue}>
                  {formatPercent(prediction.probabilities.critical)}
                </Text>
              </View>

              <Text style={[styles.detailLabel, styles.guidanceLabel]}>
                Status Guidance
              </Text>
              <Text style={styles.guidanceText}>
                {prediction.guidance ?? 'No guidance returned.'}
              </Text>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Regression Model</Text>
                <Text style={styles.detailValue}>
                  {prediction.regression_model_version}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Fire-Risk Model</Text>
                <Text style={styles.detailValue}>
                  {prediction.classification_model_version}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Prediction Timestamp</Text>
                <Text style={styles.detailValue}>
                  {formatTimestamp(prediction.predicted_at)}
                </Text>
              </View>
            </View>
          ) : null}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
    ...shadows.soft,
  },
  timestamp: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  metricsRow: {
    flexDirection: 'row',
    gap: spacing.lg,
  },
  metric: {
    gap: 2,
  },
  metricLabel: {
    fontSize: 11,
    color: colors.textSecondary,
  },
  metricValue: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  riskBadge: {
    alignSelf: 'flex-start',
    borderRadius: radii.pill,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
  },
  riskBadgeText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textOnPrimary,
  },
  unavailableText: {
    fontSize: 13,
    color: colors.textSecondary,
    fontStyle: 'italic',
  },
  detailsToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  detailsToggleText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.primary,
  },
  detailsToggleIcon: {
    fontSize: 13,
    color: colors.primary,
  },
  detailsBox: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: spacing.sm,
    gap: spacing.xs,
  },
  detailsSubtitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textPrimary,
    marginBottom: spacing.xs,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  detailValue: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  guidanceLabel: {
    marginTop: spacing.xs,
  },
  guidanceText: {
    fontSize: typography.body.fontSize,
    color: colors.textPrimary,
    lineHeight: 20,
  },
});
