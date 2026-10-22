import { router, useLocalSearchParams } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { parseAdvisoryResult } from '../../features/plantAdvisory/advisoryResult';
import { formatObservationHour } from '../../features/plantAdvisory/fieldLocation';
import type { TappingLabel } from '../../types/plantAdvisory';

// colour plus a written label, so the result never depends on colour alone;
// the action is what the farmer should do for that class
const TAPPING_STYLE: Record<TappingLabel, { color: string; action: string; text: string }> = {
  High: { color: colors.success, action: 'Tap', text: 'Good conditions for tapping.' },
  Moderate: { color: colors.warning, action: 'Tap with caution', text: 'Mixed conditions. Tap only if the bark is dry.' },
  Low: { color: colors.critical, action: 'Delay tapping', text: 'Wet or rainy conditions. Wait until the bark is dry.' },
};

export default function AdvisoryResultScreen() {
  const { data } = useLocalSearchParams<{ data?: string }>();
  const result = parseAdvisoryResult(data);

  if (!result) {
    return (
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <View style={styles.emptyState}>
          <Text style={styles.title}>No advisory to show</Text>
          <Text style={styles.subtitle}>Pick your field and generate an advisory first.</Text>
          <PrimaryButton label="Pick Your Field" onPress={() => router.replace('/plant-advisory/location')} />
        </View>
      </SafeAreaView>
    );
  }

  const tappingStyle = result.tapping ? TAPPING_STYLE[result.tapping.label] : null;

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>{result.forecastBased ? 'Forecast Advisory' : "Today's Advisory"}</Text>
        <Text style={styles.subtitle}>
          {formatObservationHour(result.observationTimestamp)} · {result.latitude}, {result.longitude}
        </Text>

        {result.forecastBased ? (
          <View style={styles.notice}>
            <Text style={styles.noticeText}>
              Based on forecast weather for a future time. Treat it as a rough plan.
            </Text>
          </View>
        ) : null}

        {/* DRC */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Dry rubber content (DRC)</Text>
          <Text style={styles.bigValue}>{result.drcPct.toFixed(2)} %</Text>
          <Text style={styles.caption}>
            {result.latexCondition
              ? `Latex condition: ${result.latexCondition}`
              : 'Normal latex was assumed'}
          </Text>
          <Text style={styles.estimateTag}>Model estimate, not a measured DRC</Text>
        </View>

        {/* Tapping */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Tapping suitability</Text>
          {result.tapping && tappingStyle ? (
            <>
              <View style={styles.labelRow}>
                <View style={[styles.labelDot, { backgroundColor: tappingStyle.color }]} />
                <Text style={styles.tappingLabel}>{result.tapping.label}</Text>
              </View>
              <View style={[styles.advice, { borderColor: tappingStyle.color }]}>
                <Text style={styles.adviceCaption}>Recommendation</Text>
                <Text style={[styles.adviceAction, { color: tappingStyle.color }]}>{tappingStyle.action}</Text>
              </View>
              <Text style={styles.tappingText}>{tappingStyle.text}</Text>
              <View style={styles.probabilities}>
                {result.tapping.classOrder.map((label) => {
                  const value = result.tapping!.probabilities[label.toLowerCase() as 'high' | 'low' | 'moderate'];
                  return (
                    <View key={label} style={styles.probabilityRow}>
                      <Text style={styles.probabilityLabel}>{label}</Text>
                      <View style={styles.barTrack}>
                        <View style={[styles.barFill, { width: `${Math.round(value * 100)}%` }]} />
                      </View>
                      <Text style={styles.probabilityValue}>{(value * 100).toFixed(0)}%</Text>
                    </View>
                  );
                })}
              </View>
              <Text style={styles.estimateTag}>Model estimate (research proxy)</Text>
            </>
          ) : (
            <Text style={styles.unavailable}>Not available. {result.tappingError ?? ''}</Text>
          )}
        </View>

        {/* Price */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Latex price</Text>
          {result.priceLkrPerKg !== null ? (
            <>
              <Text style={styles.bigValue}>
                {result.priceLkrPerKg.toFixed(2)}
                <Text style={styles.unit}> LKR/kg</Text>
              </Text>
              <Text style={styles.caption}>Using the estimated DRC of {result.drcPct.toFixed(2)} %</Text>
              <Text style={styles.estimateTag}>Model estimate, not a market price</Text>
            </>
          ) : (
            <Text style={styles.unavailable}>Not available. {result.priceError ?? ''}</Text>
          )}
        </View>

        <Text style={styles.note}>
          Based on the weather at your field. Latex condition, soil and estate details were not
          provided, so the models used their usual values for these.
        </Text>

        <PrimaryButton label="Check Another Field" onPress={() => router.back()} />
        <SecondaryButton label="Done" onPress={() => router.dismissAll()} />
      </ScrollView>
    </SafeAreaView>
  );
}

// same layout language as the Let's Set the Latex screens
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
    paddingBottom: spacing.xl,
    gap: spacing.lg,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.lg,
  },
  title: {
    fontSize: typography.display.fontSize,
    fontWeight: typography.display.fontWeight,
    color: colors.textPrimary,
  },
  subtitle: {
    fontSize: typography.body.fontSize,
    color: colors.textSecondary,
    lineHeight: 22,
  },
  notice: {
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  noticeText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.accent,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  cardLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  bigValue: {
    fontSize: 34,
    fontWeight: '700',
    color: colors.primaryDark,
  },
  unit: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  caption: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  estimateTag: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.3,
    color: colors.accent,
    textTransform: 'uppercase',
  },
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  labelDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
  },
  tappingLabel: {
    fontSize: 28,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  advice: {
    borderWidth: 1,
    borderRadius: radii.md,
    backgroundColor: colors.background,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  adviceCaption: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.3,
    color: colors.textSecondary,
    textTransform: 'uppercase',
  },
  adviceAction: {
    marginTop: 2,
    fontSize: 20,
    fontWeight: '700',
  },
  tappingText: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  probabilities: {
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  probabilityLabel: {
    width: 72,
    fontSize: 13,
    color: colors.textSecondary,
  },
  barTrack: {
    flex: 1,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.background,
    overflow: 'hidden',
  },
  barFill: {
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.primaryLight,
  },
  probabilityValue: {
    width: 40,
    textAlign: 'right',
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  unavailable: {
    fontSize: 14,
    color: colors.critical,
    lineHeight: 20,
  },
  note: {
    fontSize: 12,
    color: colors.textSecondary,
    lineHeight: 18,
  },
});
