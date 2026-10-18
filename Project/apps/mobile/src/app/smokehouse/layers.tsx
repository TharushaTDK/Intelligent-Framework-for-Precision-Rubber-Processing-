import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, shadows, spacing, typography } from '../../constants/theme';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import type { LayerCode } from '../../types/smokehouse';

const LAYER_META: { code: LayerCode; label: string }[] = [
  { code: 'L1', label: 'Lower Layer' },
  { code: 'L2', label: 'Middle Layer' },
  { code: 'L3', label: 'Upper Layer' },
];

export default function LayerSelection() {
  const { batch, setSelectedLayer } = useSmokehouseFlow();

  if (!batch) {
    return (
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No batch found</Text>
          <Text style={styles.emptyText}>
            Set up a batch before selecting a layer.
          </Text>
          <SecondaryButton
            label="Go to Batch Setup"
            onPress={() => router.replace('/smokehouse/batch')}
          />
        </View>
      </SafeAreaView>
    );
  }

  function handleSelect(code: LayerCode) {
    setSelectedLayer(code);
    router.push('/smokehouse/reading');
  }

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
        <Text style={styles.title}>Layer Selection</Text>

        <View style={styles.summaryCard}>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Batch Code</Text>
            <Text style={styles.summaryValue}>{batch.batch_code}</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Smokehouse ID</Text>
            <Text style={styles.summaryValue}>{batch.smokehouse_id}</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Status</Text>
            <Text style={styles.summaryValue}>{batch.status}</Text>
          </View>
        </View>

        {LAYER_META.map(({ code, label }) => {
          const layer = batch.layers.find((item) => item.layer_code === code);

          return (
            <Pressable
              key={code}
              accessibilityRole="button"
              onPress={() => handleSelect(code)}
              style={({ pressed }) => [
                styles.layerCard,
                pressed && styles.layerCardPressed,
              ]}
            >
              <View style={styles.layerCardHeader}>
                <Text style={styles.layerCode}>{code}</Text>
                <Text style={styles.arrow}>{'›'}</Text>
              </View>
              <Text style={styles.layerLabel}>{label}</Text>

              {layer ? (
                <View style={styles.layerDetails}>
                  <Text style={styles.layerDetailText}>
                    {layer.number_of_sheets} sheets
                  </Text>
                  <Text style={styles.layerDetailText}>
                    {layer.layer_area_m2} m²
                  </Text>
                </View>
              ) : (
                <Text style={styles.layerDetailText}>
                  Configuration unavailable.
                </Text>
              )}
            </Pressable>
          );
        })}
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
  title: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
  },
  summaryCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  summaryLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  summaryValue: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  layerCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.xs,
    ...shadows.soft,
  },
  layerCardPressed: {
    opacity: 0.85,
  },
  layerCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  layerCode: {
    fontSize: 24,
    fontWeight: '800',
    color: colors.primary,
  },
  arrow: {
    fontSize: 22,
    fontWeight: '700',
    color: colors.primary,
  },
  layerLabel: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  layerDetails: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.xs,
  },
  layerDetailText: {
    fontSize: 13,
    color: colors.textSecondary,
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
  emptyText: {
    fontSize: 13,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
