import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import {
  STATUS_LABELS,
  coagulationTimeText,
  formatAcid,
  formatDate,
  formatDrc,
  formatWater,
} from '../../features/latex/BatchSummaryCard';
import { Banner, Chip, tints } from '../../features/latex/ui';
import { listRecommendations, toReadableMessage } from '../../services/latexApi';
import type { CoagulationRecommendationResponse, RecommendationStatus } from '../../types/latex';

const FILTERS: { key: string; label: string; match: (s: RecommendationStatus) => boolean }[] = [
  { key: 'all', label: 'All', match: () => true },
  { key: 'running', label: 'Running', match: (s) => s === 'in_progress' },
  { key: 'waiting', label: 'Not started', match: (s) => s === 'recommended' },
  { key: 'done', label: 'Done', match: (s) => s === 'completed' || s === 'cancelled' },
];

function BatchRow({ batch }: { batch: CoagulationRecommendationResponse }) {
  const status = STATUS_LABELS[batch.status];
  // a running batch opens its timer, the rest open the details
  const open = () =>
    router.push({
      pathname: batch.status === 'in_progress' ? '/latex/timer' : '/latex/details',
      params: { id: String(batch.id) },
    });

  return (
    <Pressable
      accessibilityRole="button"
      onPress={open}
      style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
    >
      <View style={styles.rowHeader}>
        <Text style={styles.rowTitle}>Batch #{batch.id}</Text>
        <Chip label={status.label} tint={status.tint} />
      </View>
      <Text style={styles.rowInputs}>
        DRC {formatDrc(batch.drc)} · {batch.latex_volume_L} L · {formatDate(batch.created_at)}
      </Text>
      <View style={styles.rowValues}>
        {batch.water_acid ? (
          <>
            <Text style={[styles.rowValue, styles.waterValue]}>💧 {formatWater(batch.water_acid.water_amount_L)}</Text>
            <Text style={[styles.rowValue, styles.acidValue]}>
              🧪 {formatAcid(batch.water_acid.prepared_1pct_formic_acid_solution_mL)}
            </Text>
          </>
        ) : null}
        {batch.coagulation_time_hours !== null ? (
          <Text style={styles.rowValue}>⏱ {coagulationTimeText(batch.coagulation_time_hours)}</Text>
        ) : null}
      </View>
    </Pressable>
  );
}

export default function BatchHistory() {
  const [items, setItems] = useState<CoagulationRecommendationResponse[]>([]);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listRecommendations(100);
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

  const active = FILTERS.find((f) => f.key === filter) ?? FILTERS[0];
  const shown = items.filter((item) => active.match(item.status));
  const completed = items.filter((item) => item.status === 'completed').length;
  const running = items.filter((item) => item.status === 'in_progress').length;

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <FlatList
        data={shown}
        keyExtractor={(item) => String(item.id)}
        renderItem={({ item }) => <BatchRow batch={item} />}
        contentContainerStyle={styles.listContent}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        refreshing={loading && items.length > 0}
        onRefresh={load}
        ListHeaderComponent={
          <View style={styles.headerSection}>
            <View style={styles.hero}>
              <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.back}>
                <Text style={styles.backLabel}>{'‹  Back'}</Text>
              </Pressable>
              <Text style={styles.title}>Batch History</Text>
              <View style={styles.statsRow}>
                <View style={styles.stat}>
                  <Text style={styles.statValue}>{items.length}</Text>
                  <Text style={styles.statLabel}>Batches</Text>
                </View>
                <View style={styles.stat}>
                  <Text style={styles.statValue}>{running}</Text>
                  <Text style={styles.statLabel}>Running</Text>
                </View>
                <View style={styles.stat}>
                  <Text style={styles.statValue}>{completed}</Text>
                  <Text style={styles.statLabel}>Completed</Text>
                </View>
              </View>
            </View>

            <View style={styles.filters}>
              {FILTERS.map((f) => (
                <Pressable
                  key={f.key}
                  accessibilityRole="tab"
                  accessibilityState={{ selected: f.key === filter }}
                  onPress={() => setFilter(f.key)}
                  style={[styles.filter, f.key === filter && styles.filterOn]}
                >
                  <Text style={[styles.filterText, f.key === filter && styles.filterTextOn]}>{f.label}</Text>
                </Pressable>
              ))}
            </View>

            {loading && items.length === 0 ? (
              <View style={styles.loadingRow}>
                <ActivityIndicator color={colors.primary} />
                <Text style={styles.loadingText}>Loading batches...</Text>
              </View>
            ) : null}

            {error ? <Banner tone="danger" text={error} /> : null}
          </View>
        }
        ListEmptyComponent={
          !loading && !error ? (
            <View style={styles.emptyBox}>
              <Text style={styles.emptyText}>{items.length ? 'No batches in this filter.' : 'No batches yet.'}</Text>
              {items.length === 0 ? (
                <PrimaryButton label="New Batch" onPress={() => router.replace('/latex/batch')} />
              ) : null}
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
  listContent: {
    paddingBottom: spacing.xxl,
  },
  headerSection: {
    gap: spacing.md,
    marginBottom: spacing.md,
  },
  hero: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.lg,
    borderBottomLeftRadius: radii.xl,
    borderBottomRightRadius: radii.xl,
  },
  back: {
    alignSelf: 'flex-start',
    paddingVertical: spacing.sm,
  },
  backLabel: {
    color: colors.primaryLight,
    fontSize: 15,
    fontWeight: '700',
  },
  title: {
    color: colors.textOnPrimary,
    fontSize: 28,
    fontWeight: '800',
  },
  statsRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  stat: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderRadius: radii.md,
    padding: spacing.sm,
    alignItems: 'center',
  },
  statValue: {
    color: colors.textOnPrimary,
    fontSize: 22,
    fontWeight: '800',
  },
  statLabel: {
    color: colors.primaryLight,
    fontSize: 12,
    fontWeight: '600',
  },
  filters: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  filter: {
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
  },
  filterOn: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  filterText: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  filterTextOn: {
    color: colors.textOnPrimary,
  },
  row: {
    marginHorizontal: spacing.md,
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
    fontWeight: '800',
    color: colors.textPrimary,
  },
  rowInputs: {
    fontSize: 12,
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
    fontWeight: '700',
    color: colors.primaryDark,
    backgroundColor: '#E4F2E8',
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
    borderRadius: radii.sm,
    overflow: 'hidden',
  },
  waterValue: {
    color: tints.water.strong,
    backgroundColor: tints.water.soft,
  },
  acidValue: {
    color: tints.acid.strong,
    backgroundColor: tints.acid.soft,
  },
  emptyBox: {
    alignItems: 'stretch',
    gap: spacing.md,
    padding: spacing.xl,
  },
  emptyText: {
    textAlign: 'center',
    fontSize: 14,
    color: colors.textSecondary,
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
