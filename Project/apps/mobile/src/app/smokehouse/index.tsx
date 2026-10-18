import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';

const LAYERS = [
  { code: 'L3', label: 'Upper Layer' },
  { code: 'L2', label: 'Middle Layer' },
  { code: 'L1', label: 'Lower Layer' },
] as const;

export default function SmokehouseOverview() {
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
        <Text style={styles.title}>Smart Smokehouse</Text>
        <Text style={styles.subtitle}>
          Predict drying progress and monitor fire-risk conditions across
          each smokehouse layer.
        </Text>

        <View style={styles.notice}>
          <Text style={styles.noticeText}>
            Decision-support prototype — not a certified fire alarm.
          </Text>
        </View>

        <View style={styles.diagramCard}>
          {LAYERS.map((layer) => (
            <View key={layer.code} style={styles.layerRow}>
              <View style={styles.layerSlab}>
                <Text style={styles.layerCode}>{layer.code}</Text>
              </View>
              <Text style={styles.layerLabel}>{layer.label}</Text>
            </View>
          ))}
        </View>

        <PrimaryButton
          label="Start Prediction"
          onPress={() => router.push('/smokehouse/batch')}
        />

        <SecondaryButton
          label="Prediction History"
          onPress={() => router.push('/smokehouse/history')}
        />
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
    paddingBottom: spacing.xl,
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
    color: colors.textSecondary,
  },
  diagramCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.md,
  },
  layerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  layerSlab: {
    width: 56,
    height: 44,
    borderRadius: radii.sm,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  layerCode: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.primaryDark,
  },
  layerLabel: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.textPrimary,
  },
});
