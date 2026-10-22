import { StyleSheet, Text, View } from 'react-native';

import { colors, radii, spacing, typography } from '../../constants/theme';
import type { AcousticMeasurementResponse } from '../../types/moisture';
import { TechnicalDetailsCard } from './TechnicalDetailsCard';

type MeasurementResultCardProps = {
  measurement: AcousticMeasurementResponse;
};

export function MeasurementResultCard({ measurement }: MeasurementResultCardProps) {
  const isValid = measurement.quality_status === 'VALID';

  return (
    <View style={styles.container}>
      {/* Restrained Scope Badge */}
      <View style={styles.headerBadgeWrap}>
        <Text style={styles.preliminaryLabel}>Preliminary recording-quality assessment</Text>
      </View>

      {/* Main Assessment Status Card */}
      <View style={[styles.statusCard, isValid ? styles.statusCardValid : styles.statusCardInvalid]}>
        <Text style={[styles.statusTitle, isValid ? styles.statusTextValid : styles.statusTextInvalid]}>
          {isValid ? 'Recording accepted' : 'Please repeat the recording'}
        </Text>
        <Text style={styles.statusDescription}>
          {isValid
            ? 'The preliminary quality pipeline accepted the audio strike characteristics.'
            : 'The audio did not meet standard strike criteria or quality checks.'}
        </Text>
      </View>

      {/* Backend Guidance */}
      <View style={styles.guidanceCard}>
        <Text style={styles.guidanceHeading}>Backend Guidance</Text>
        <Text style={styles.guidanceText}>{measurement.guidance}</Text>
      </View>

      {/* Scientific/Research Transparency Notice: Moisture Unavailable */}
      <View style={styles.noticeCard}>
        <View style={styles.noticeDot} />
        <View style={styles.noticeContent}>
          <Text style={styles.noticeTitle}>Moisture prediction is not available yet.</Text>
          <Text style={styles.noticeText}>
            This module currently assesses recording validity only. Moisture regression models and dryness categorization are active research steps and will be integrated in a future release.
          </Text>
        </View>
      </View>

      {/* Expandable Technical Details */}
      <TechnicalDetailsCard measurement={measurement} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.md,
  },
  headerBadgeWrap: {
    alignSelf: 'flex-start',
    backgroundColor: '#EDF5EF',
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: radii.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
  },
  preliminaryLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.primaryDark,
  },
  statusCard: {
    borderRadius: radii.lg,
    padding: spacing.lg,
    borderWidth: 1,
  },
  statusCardValid: {
    backgroundColor: '#F0F9F3',
    borderColor: '#C2E8D0',
  },
  statusCardInvalid: {
    backgroundColor: '#FFF5F5',
    borderColor: '#F8C8C8',
  },
  statusTitle: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    marginBottom: spacing.xs,
  },
  statusTextValid: {
    color: colors.success,
  },
  statusTextInvalid: {
    color: colors.critical,
  },
  statusDescription: {
    fontSize: 14,
    color: colors.textSecondary,
    lineHeight: 20,
  },
  guidanceCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  guidanceHeading: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  guidanceText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.textPrimary,
    lineHeight: 22,
  },
  noticeCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  noticeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.accent,
    marginTop: 6,
  },
  noticeContent: {
    flex: 1,
  },
  noticeTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
    marginBottom: 2,
  },
  noticeText: {
    fontSize: 12,
    color: colors.textSecondary,
    lineHeight: 18,
  },
});
