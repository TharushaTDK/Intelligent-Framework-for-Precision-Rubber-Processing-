import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { formatDate, formatDrc, hoursAndMinutes } from '../../features/latex/BatchSummaryCard';
import { listRecommendations, toReadableMessage } from '../../services/latexApi';
import type { CoagulationRecommendationResponse } from '../../types/latex';

function BatchRow({ batch }: { batch: CoagulationRecommendationResponse }) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={() => router.push({ pathname: '/latex/details', params: { id: String(batch.id) } })}
      style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
    >
      <View style={styles.rowHeader}>
        <Text style={styles.rowTitle}>Batch #{batch.id}</Text>
        <Text style={styles.rowDate}>{formatDate(batch.created_at)}</Text>
      </View>
      <Text style={styles.rowInputs}>
        DRC {formatDrc(batch.drc)} · {batch.latex_volume_L} L
      </Text>
      <View style={styles.rowValues}>
        {batch.water_acid ? (
          <>
            <Text style={styles.rowValue}>Water {batch.water_acid.water_amount_L.toFixed(2)} L</Text>
            <Text style={styles.rowValue}>
              Acid {batch.water_acid.prepared_1pct_formic_acid_solution_mL.toFixed(1)} mL
            </Text>
          </>
        ) : null}
        {batch.coagulation_time_hours !== null ? (
          <Text style={styles.rowValue}>Time {hoursAndMinutes(batch.coagulation_time_hours)}</Text>
        ) : null}
      </View>
      <Text style={styles.rowLink}>View details ›</Text>
    </Pressable>
  );
}

export default function BatchHistory() {
  const [items, setItems] = useState<CoagulationRecommendationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listRecommendations();
      setItems(response.items);
    } catch (err) {
      setError(toReadableMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  // reload every time the page is opened so new batches show up
  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <FlatList
        data={items}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => <BatchRow batch={item} />}
        contentContainerStyle={styles.listContent}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        refreshing={loading && items.length > 0}
        onRefresh={load}
        ListHeaderComponent={
          <View style={styles.headerSection}>
            <Text style={styles.title}>Batch History</Text>
            <Text style={styles.subtitle}>Saved batches, newest first.</Text>

            {loading && items.length === 0 ? (
              <View style={styles.loadingRow}>
                <ActivityIndicator color={colors.primary} />
                <Text style={styles.loadingText}>Loading batches...</Text>
              </View>
            ) : null}

            {error ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            ) : null}
          </View>
        }
        ListEmptyComponent={
          !loading && !error ? (
            <View style={styles.emptyBox}>
              <Text style={styles.emptyText}>No batches yet.</Text>
              <PrimaryButton label="New Batch" onPress={() => router.replace('/latex/batch')} />
            </View>
          ) : null
        }
      />
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
  listContent: {
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.xxl,
  },
  headerSection: {
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  title: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
  },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  row: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.xs,
  },
  rowPressed: {
    borderColor: colors.primary,
  },
  rowHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: spacing.sm,
  },
  rowTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  rowDate: {
    flexShrink: 1,
    fontSize: 12,
    color: colors.textSecondary,
    textAlign: 'right',
  },
  rowInputs: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  rowValues: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  rowValue: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.primaryDark,
    backgroundColor: colors.background,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radii.sm,
    overflow: 'hidden',
  },
  rowLink: {
    marginTop: spacing.xs,
    fontSize: 13,
    fontWeight: '600',
    color: colors.primary,
  },
  emptyBox: {
    alignItems: 'stretch',
    gap: spacing.md,
    paddingVertical: spacing.xl,
  },
  emptyText: {
    textAlign: 'center',
    fontSize: 14,
    color: colors.textSecondary,
  },
  errorBox: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.critical,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  errorText: {
    color: colors.critical,
    fontSize: 13,
    fontWeight: '600',
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
