import { router } from 'expo-router';
import { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import type { FireRiskLabel } from '../../types/smokehouse';

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
  return `${(value * 100).toFixed(1)}%`;
}

export default function PredictionResult() {
  const { batch, selectedLayer, lastReading, verification } = useSmokehouseFlow();
  const [detailsExpanded, setDetailsExpanded] = useState(false);

  if (!batch || !selectedLayer || !lastReading) {
    return (
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No prediction available</Text>
          <SecondaryButton
            label="Submit a Reading"
            onPress={() => router.replace('/smokehouse/reading')}
          />
        </View>
      </SafeAreaView>
    );
  }

  const { reading, prediction, missing_numeric_features, historical_reading_count } =
    lastReading;

  const riskColor = fireRiskColor(prediction.fire_risk_label);

  const probabilityRows: { label: string; value: number; color: string }[] = [
    { label: 'Normal', value: prediction.probabilities.normal, color: colors.success },
    { label: 'Warning', value: prediction.probabilities.warning, color: colors.warning },
    { label: 'Critical', value: prediction.probabilities.critical, color: colors.critical },
  ];

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable
        accessibilityRole="button"
        onPress={() => router.back()}
        style={styles.backButton}
      >
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.eyebrow}>PREDICTION RESULT</Text>

        <View style={styles.contextCard}>
          <View style={styles.contextRow}>
            <Text style={styles.contextLabel}>Selected Layer</Text>
            <Text style={styles.contextValue}>{selectedLayer}</Text>
          </View>
          <View style={styles.contextRow}>
            <Text style={styles.contextLabel}>Temperature</Text>
            <Text style={styles.contextValue}>{reading.temperature_c} °C</Text>
          </View>
          <View style={styles.contextRow}>
            <Text style={styles.contextLabel}>Humidity</Text>
            <Text style={styles.contextValue}>{reading.humidity_pct}%</Text>
          </View>
          <View style={styles.contextRow}>
            <Text style={styles.contextLabel}>Prediction Timestamp</Text>
            <Text style={styles.contextValue}>
              {formatTimestamp(prediction.predicted_at)}
            </Text>
          </View>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>DRYING PREDICTION</Text>
          <Text style={styles.dryingValue}>
            {prediction.remaining_drying_time_hours.toFixed(2)}{' '}
            <Text style={styles.dryingUnit}>hours</Text>
          </Text>
          <Text style={styles.dryingCaption}>Remaining Drying Time</Text>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>FIRE-RISK CONDITION</Text>
          <View style={[styles.riskBadge, { backgroundColor: riskColor }]}>
            <Text style={styles.riskBadgeText}>
              {prediction.fire_risk_label} fire-risk condition
            </Text>
          </View>
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>PROBABILITIES</Text>
          {probabilityRows.map((row) => (
            <View key={row.label} style={styles.probabilityRow}>
              <View style={styles.probabilityHeader}>
                <Text style={styles.probabilityLabel}>{row.label}</Text>
                <Text style={styles.probabilityValue}>
                  {formatPercent(row.value)}
                </Text>
              </View>
              <View style={styles.probabilityTrack}>
                <View
                  style={[
                    styles.probabilityFill,
                    {
                      width: `${Math.min(Math.max(row.value * 100, 0), 100)}%`,
                      backgroundColor: row.color,
                    },
                  ]}
                />
              </View>
            </View>
          ))}
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>STATUS GUIDANCE</Text>
          <Text style={styles.guidanceText}>
            {prediction.guidance ?? 'No guidance returned.'}
          </Text>
        </View>

        <Pressable
          accessibilityRole="button"
          onPress={() => setDetailsExpanded((value) => !value)}
          style={styles.detailsToggle}
        >
          <Text style={styles.detailsToggleText}>Model Details</Text>
          <Text style={styles.detailsToggleIcon}>
            {detailsExpanded ? '▾' : '▸'}
          </Text>
        </Pressable>

        {detailsExpanded ? (
          <View style={styles.sectionCard}>
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
              <Text style={styles.detailLabel}>Historical Readings Used</Text>
              <Text style={styles.detailValue}>{historical_reading_count}</Text>
            </View>
            <View style={styles.detailColumn}>
              <Text style={styles.detailLabel}>Missing History Features</Text>
              <Text style={styles.detailValue}>
                {missing_numeric_features.length === 0
                  ? 'None'
                  : missing_numeric_features.join(', ')}
              </Text>
            </View>
          </View>
        ) : null}

        <View style={styles.verificationCard}>
          {verification?.status === 'verified' ? (
            <>
              <Text style={styles.verificationTitleOk}>
                Backend result verified
              </Text>
              <Text style={styles.verificationText}>
                The prediction displayed here matches the latest prediction
                persisted by the backend.
              </Text>
            </>
          ) : verification?.status === 'mismatch' ? (
            <>
              <Text style={styles.verificationTitleBad}>
                Persisted result mismatch
              </Text>
              <Text style={styles.verificationText}>
                {verification.mismatchedFields.join(', ')}
              </Text>
            </>
          ) : (
            <Text style={styles.verificationTitleNeutral}>
              Persistence verification unavailable
            </Text>
          )}
        </View>

        <View style={styles.actions}>
          <PrimaryButton
            label="New Reading — Same Layer"
            onPress={() => router.push('/smokehouse/reading')}
          />
          <SecondaryButton
            label="Choose Another Layer"
            onPress={() => router.push('/smokehouse/layers')}
          />
          <SecondaryButton
            label="Smokehouse Home"
            onPress={() => router.push('/smokehouse')}
          />
          <SecondaryButton
            label="View Prediction History"
            onPress={() => router.push('/smokehouse/history')}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  backLabel: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.primary,
  },
  scrollContent: {
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.xxl,
    gap: spacing.md,
  },
  eyebrow: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.5,
    color: colors.textSecondary,
  },
  contextCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  contextRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  contextLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  contextValue: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  sectionCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
    color: colors.textSecondary,
  },
  dryingValue: {
    fontSize: 36,
    fontWeight: '800',
    color: colors.primaryDark,
  },
  dryingUnit: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  dryingCaption: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  riskBadge: {
    borderRadius: radii.pill,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    alignSelf: 'flex-start',
  },
  riskBadgeText: {
    color: colors.textOnPrimary,
    fontSize: 15,
    fontWeight: '700',
  },
  probabilityRow: {
    gap: spacing.xs,
  },
  probabilityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  probabilityLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  probabilityValue: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  probabilityTrack: {
    height: 8,
    borderRadius: radii.pill,
    backgroundColor: colors.background,
    overflow: 'hidden',
  },
  probabilityFill: {
    height: '100%',
    borderRadius: radii.pill,
  },
  guidanceText: {
    fontSize: typography.body.fontSize,
    color: colors.textPrimary,
    lineHeight: 22,
  },
  detailsToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
  },
  detailsToggleText: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primary,
  },
  detailsToggleIcon: {
    fontSize: 14,
    color: colors.primary,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailColumn: {
    gap: spacing.xs,
  },
  detailLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  detailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  verificationCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  verificationTitleOk: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.success,
  },
  verificationTitleBad: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.critical,
  },
  verificationTitleNeutral: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  verificationText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  actions: {
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.md,
  },
  emptyTitle: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
  },
});
