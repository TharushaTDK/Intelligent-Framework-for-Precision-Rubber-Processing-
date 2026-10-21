import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, radii, spacing } from '../../constants/theme';
import type { AcousticMeasurementResponse } from '../../types/moisture';
import { getReadableRejectionReason, getReadableWarning } from './reasonCodes';

type TechnicalDetailsCardProps = {
  measurement: AcousticMeasurementResponse;
};

export function TechnicalDetailsCard({ measurement }: TechnicalDetailsCardProps) {
  const [expanded, setExpanded] = useState(false);

  const formattedDate = (() => {
    try {
      const date = new Date(measurement.created_at);
      return Number.isNaN(date.getTime()) ? measurement.created_at : date.toLocaleString();
    } catch {
      return measurement.created_at;
    }
  })();

  return (
    <View style={styles.container}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={expanded ? 'Hide technical details' : 'Show technical details'}
        onPress={() => setExpanded((prev) => !prev)}
        style={({ pressed }) => [styles.headerPressable, pressed && styles.pressed]}
      >
        <Text style={styles.headerTitle}>Technical Details & Diagnostics</Text>
        <Text style={styles.chevron}>{expanded ? '▲ Hide' : '▼ Expand'}</Text>
      </Pressable>

      {expanded ? (
        <View style={styles.content}>
          {/* Metadata Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Identification</Text>
            <View style={styles.row}>
              <Text style={styles.label}>Measurement ID:</Text>
              <Text selectable style={[styles.value, styles.mono]}>
                {measurement.measurement_id}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Timestamp:</Text>
              <Text style={styles.value}>{formattedDate}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Filename:</Text>
              <Text style={[styles.value, styles.mono]}>{measurement.filename}</Text>
            </View>
            {measurement.sheet_id ? (
              <View style={styles.row}>
                <Text style={styles.label}>Sheet ID:</Text>
                <Text style={styles.value}>{measurement.sheet_id}</Text>
              </View>
            ) : null}
            {measurement.condition_id ? (
              <View style={styles.row}>
                <Text style={styles.label}>Condition ID:</Text>
                <Text style={styles.value}>{measurement.condition_id}</Text>
              </View>
            ) : null}
            <View style={styles.row}>
              <Text style={styles.label}>Source SHA-256:</Text>
              <Text selectable style={[styles.value, styles.monoSmall]}>
                {measurement.source_sha256}
              </Text>
            </View>
          </View>

          {/* Assessment Pipeline Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Pipeline Status</Text>
            <View style={styles.row}>
              <Text style={styles.label}>Technical status:</Text>
              <Text
                style={[
                  styles.value,
                  styles.badge,
                  measurement.technical_status === 'TECHNICAL_PASSED'
                    ? styles.badgeSuccess
                    : styles.badgeCritical,
                ]}
              >
                {measurement.technical_status}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Decision source:</Text>
              <Text style={styles.value}>{measurement.decision_source}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Classifier status:</Text>
              <Text style={styles.value}>{measurement.classifier_status}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Rule version:</Text>
              <Text style={styles.value}>{measurement.rule_version}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Model version:</Text>
              <Text style={styles.value}>{measurement.model_version ?? 'Not evaluated / technical decision'}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Model status:</Text>
              <Text style={styles.value}>{measurement.model_status}</Text>
            </View>
          </View>

          {/* Rejection Reasons */}
          {measurement.rejection_reasons.length > 0 ? (
            <View style={styles.section}>
              <Text style={styles.sectionTitleWarning}>Rejection Reasons</Text>
              {measurement.rejection_reasons.map((reason, index) => (
                <View key={`reason-${index}`} style={styles.reasonBlock}>
                  <Text style={styles.reasonCode}>Code: {reason}</Text>
                  <Text style={styles.reasonText}>{getReadableRejectionReason(reason)}</Text>
                </View>
              ))}
            </View>
          ) : null}

          {/* Warnings */}
          {measurement.warnings.length > 0 ? (
            <View style={styles.section}>
              <Text style={styles.sectionTitleNotice}>Pipeline Warnings</Text>
              {measurement.warnings.map((warning, index) => (
                <View key={`warning-${index}`} style={styles.warningBlock}>
                  <Text style={styles.warningCode}>Code: {warning}</Text>
                  <Text style={styles.warningText}>{getReadableWarning(warning)}</Text>
                </View>
              ))}
            </View>
          ) : null}

          {/* Acoustic Diagnostics */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Acoustic Diagnostics</Text>
            <View style={styles.row}>
              <Text style={styles.label}>Candidate strike count:</Text>
              <Text style={styles.value}>{measurement.candidate_count}</Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Onset time:</Text>
              <Text style={styles.value}>
                {measurement.onset_seconds !== null
                  ? `${measurement.onset_seconds.toFixed(4)} s`
                  : 'Not detected'}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Segment duration:</Text>
              <Text style={styles.value}>
                {measurement.segment_duration_seconds !== null
                  ? `${measurement.segment_duration_seconds.toFixed(4)} s`
                  : 'Not available'}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Peak amplitude:</Text>
              <Text style={styles.value}>
                {measurement.peak_percent_full_scale !== null
                  ? `${measurement.peak_percent_full_scale.toFixed(1)}% full scale`
                  : 'Not available'}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Near-full-scale fraction:</Text>
              <Text style={styles.value}>
                {measurement.near_full_scale_fraction !== null
                  ? `${(measurement.near_full_scale_fraction * 100).toFixed(4)}%`
                  : 'Not available'}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>Uncalibrated invalidity score:</Text>
              <Text style={styles.value}>
                {measurement.invalid_probability_uncalibrated !== null
                  ? measurement.invalid_probability_uncalibrated.toFixed(4)
                  : 'Not evaluated'}
              </Text>
            </View>
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    marginTop: spacing.sm,
  },
  headerPressable: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.md,
  },
  pressed: {
    backgroundColor: colors.background,
  },
  headerTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  chevron: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.primary,
  },
  content: {
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
    gap: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: spacing.md,
  },
  section: {
    gap: spacing.xs,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textSecondary,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  sectionTitleWarning: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.critical,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  sectionTitleNotice: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.warning,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: spacing.sm,
    paddingVertical: 2,
  },
  label: {
    fontSize: 13,
    color: colors.textSecondary,
    flex: 1,
  },
  value: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.textPrimary,
    flexShrink: 1,
    textAlign: 'right',
  },
  mono: {
    fontFamily: 'monospace',
    fontSize: 11,
  },
  monoSmall: {
    fontFamily: 'monospace',
    fontSize: 10,
    maxWidth: '60%',
  },
  badge: {
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: radii.sm,
    overflow: 'hidden',
    fontWeight: '700',
  },
  badgeSuccess: {
    backgroundColor: '#E8F5E9',
    color: colors.success,
  },
  badgeCritical: {
    backgroundColor: '#FFEBEE',
    color: colors.critical,
  },
  reasonBlock: {
    backgroundColor: '#FFF8F8',
    borderLeftWidth: 3,
    borderLeftColor: colors.critical,
    padding: spacing.sm,
    borderRadius: radii.sm,
    marginBottom: spacing.xs,
  },
  reasonCode: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.critical,
    fontFamily: 'monospace',
  },
  reasonText: {
    fontSize: 12,
    color: colors.textPrimary,
    marginTop: 2,
    lineHeight: 16,
  },
  warningBlock: {
    backgroundColor: '#FFFDF0',
    borderLeftWidth: 3,
    borderLeftColor: colors.warning,
    padding: spacing.sm,
    borderRadius: radii.sm,
    marginBottom: spacing.xs,
  },
  warningCode: {
    fontSize: 11,
    fontWeight: '700',
    color: '#8A6D00',
    fontFamily: 'monospace',
  },
  warningText: {
    fontSize: 12,
    color: colors.textPrimary,
    marginTop: 2,
    lineHeight: 16,
  },
});
