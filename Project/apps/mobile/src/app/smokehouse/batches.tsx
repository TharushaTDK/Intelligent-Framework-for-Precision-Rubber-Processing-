import { router } from 'expo-router';
import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, shadows, spacing, typography } from '../../constants/theme';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import { SmokehouseIdInput } from '../../features/smokehouse/SmokehouseIdInput';
import {
  getSmokehouseBatch,
  listSmokehouseBatches,
  SmokehouseApiError,
  updateSmokehouseBatchStatus,
} from '../../services/smokehouseApi';
import type { BatchStatus, SmokehouseBatchSummary } from '../../types/smokehouse';

type StatusFilter = 'all' | 'running' | 'completed';

function statusMeta(status: BatchStatus): { label: string; running: boolean } {
  if (status === 'active') {
    return { label: 'RUNNING', running: true };
  }
  if (status === 'completed') {
    return { label: 'COMPLETED', running: false };
  }
  // Cancelled batches share the neutral "completed" visual bucket rather
  // than introducing a third status category in the UI.
  return { label: 'CANCELLED', running: false };
}

function formatStarted(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const datePart = date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
  const timePart = date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  });
  return `${datePart} · ${timePart}`;
}

export default function CurrentBatches() {
  const { setBatch, setSelectedLayer } = useSmokehouseFlow();

  const [smokehouseIdInput, setSmokehouseIdInput] = useState('');
  const [smokehouseError, setSmokehouseError] = useState<string | null>(null);
  const [batchListLoading, setBatchListLoading] = useState(false);
  const [batchListError, setBatchListError] = useState<string | null>(null);
  const [batchResults, setBatchResults] = useState<SmokehouseBatchSummary[] | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const [filter, setFilter] = useState<StatusFilter>('all');
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyBatchCode, setBusyBatchCode] = useState<string | null>(null);

  async function handleFindBatches() {
    const trimmed = smokehouseIdInput.trim();

    if (!trimmed) {
      setSmokehouseError('Smokehouse ID is required.');
      return;
    }

    setSmokehouseError(null);
    setActionError(null);
    setBatchListLoading(true);
    setBatchListError(null);

    try {
      const response = await listSmokehouseBatches(trimmed);
      setBatchResults(response.batches);
      setHasSearched(true);
    } catch (error) {
      setBatchResults(null);
      setHasSearched(true);

      setBatchListError(
        error instanceof SmokehouseApiError
          ? error.message
          : 'Unable to load batches. Please try again.',
      );
    } finally {
      setBatchListLoading(false);
    }
  }

  async function handleContinueMonitoring(batchCode: string) {
    setActionError(null);
    setBusyBatchCode(batchCode);

    try {
      const fullBatch = await getSmokehouseBatch(batchCode);
      setBatch(fullBatch);
      router.push('/smokehouse/layers');
    } catch (error) {
      setActionError(
        error instanceof SmokehouseApiError
          ? error.message
          : 'Unable to continue this batch.',
      );
    } finally {
      setBusyBatchCode(null);
    }
  }

  async function handleViewHistory(batchCode: string) {
    setActionError(null);
    setBusyBatchCode(batchCode);

    try {
      const fullBatch = await getSmokehouseBatch(batchCode);
      setBatch(fullBatch);
      setSelectedLayer('L1');
      router.push('/smokehouse/history');
    } catch (error) {
      setActionError(
        error instanceof SmokehouseApiError
          ? error.message
          : 'Unable to open batch history.',
      );
    } finally {
      setBusyBatchCode(null);
    }
  }

  async function handleMarkCompleted(batchCode: string) {
    setActionError(null);
    setBusyBatchCode(batchCode);

    try {
      await updateSmokehouseBatchStatus(batchCode, { status: 'completed' });
      await handleFindBatches();
    } catch (error) {
      setActionError(
        error instanceof SmokehouseApiError
          ? error.message
          : 'Unable to update batch status.',
      );
    } finally {
      setBusyBatchCode(null);
    }
  }

  function confirmMarkCompleted(batchCode: string) {
    Alert.alert(
      'Mark this drying batch as completed?',
      'Completed batches remain available in prediction history but will no longer accept new monitoring predictions.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Mark Completed',
          style: 'destructive',
          onPress: () => handleMarkCompleted(batchCode),
        },
      ],
    );
  }

  const sortedBatches = useMemo(() => {
    if (!batchResults) {
      return [];
    }
    const rank = (status: BatchStatus) => (status === 'active' ? 0 : 1);
    return [...batchResults].sort((a, b) => rank(a.status) - rank(b.status));
  }, [batchResults]);

  const filteredBatches = useMemo(
    () =>
      sortedBatches.filter((item) => {
        if (filter === 'running') return item.status === 'active';
        if (filter === 'completed') return item.status !== 'active';
        return true;
      }),
    [sortedBatches, filter],
  );

  const emptyMessage =
    filter === 'running'
      ? 'No running batches.'
      : filter === 'completed'
        ? 'No completed batches yet.'
        : 'No smokehouse batches found.';

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable
        accessibilityRole="button"
        onPress={() => router.back()}
        style={styles.backButton}
      >
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <FlatList
        data={filteredBatches}
        keyExtractor={(item) => item.batch_code}
        contentContainerStyle={styles.listContent}
        keyboardShouldPersistTaps="handled"
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        renderItem={({ item }) => {
          const meta = statusMeta(item.status);
          const isBusy = busyBatchCode === item.batch_code;

          return (
            <View style={styles.batchCard}>
              <View style={styles.batchHeaderRow}>
                <Text style={styles.batchCode}>{item.batch_code}</Text>
                <View
                  style={[
                    styles.statusBadge,
                    meta.running ? styles.statusBadgeRunning : styles.statusBadgeCompleted,
                  ]}
                >
                  <Text
                    style={[
                      styles.statusBadgeText,
                      meta.running
                        ? styles.statusBadgeTextRunning
                        : styles.statusBadgeTextCompleted,
                    ]}
                  >
                    {meta.label}
                  </Text>
                </View>
              </View>

              <Text style={styles.batchMeta}>
                Started: {formatStarted(item.started_at)}
              </Text>

              {isBusy ? (
                <View style={styles.loadingRow}>
                  <ActivityIndicator color={colors.primary} />
                </View>
              ) : meta.running ? (
                <View style={styles.actionsColumn}>
                  <Pressable
                    accessibilityRole="button"
                    onPress={() => handleContinueMonitoring(item.batch_code)}
                    style={({ pressed }) => [
                      styles.primaryAction,
                      pressed && styles.primaryActionPressed,
                    ]}
                  >
                    <Text style={styles.primaryActionText}>Continue Monitoring</Text>
                  </Pressable>

                  <Pressable
                    accessibilityRole="button"
                    onPress={() => confirmMarkCompleted(item.batch_code)}
                    style={styles.subtleAction}
                  >
                    <Text style={styles.subtleActionText}>Mark as Completed</Text>
                  </Pressable>
                </View>
              ) : (
                <Pressable
                  accessibilityRole="button"
                  onPress={() => handleViewHistory(item.batch_code)}
                  style={({ pressed }) => [
                    styles.primaryAction,
                    pressed && styles.primaryActionPressed,
                  ]}
                >
                  <Text style={styles.primaryActionText}>View History</Text>
                </Pressable>
              )}
            </View>
          );
        }}
        ListHeaderComponent={
          <View style={styles.headerSection}>
            <Text style={styles.title}>Current / Previous Batches</Text>
            <Text style={styles.subtitle}>
              Find a smokehouse&apos;s drying batches and continue monitoring an
              existing one.
            </Text>

            <View style={styles.card}>
              <SmokehouseIdInput
                value={smokehouseIdInput}
                onChangeText={setSmokehouseIdInput}
                error={smokehouseError ?? undefined}
              />
              <SecondaryButton label="Find Batches" onPress={handleFindBatches} />
            </View>

            {batchListLoading ? (
              <View style={styles.loadingRow}>
                <ActivityIndicator color={colors.primary} />
                <Text style={styles.loadingText}>Looking up batches...</Text>
              </View>
            ) : null}

            {!batchListLoading && batchListError ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{batchListError}</Text>
              </View>
            ) : null}

            {!batchListLoading && actionError ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{actionError}</Text>
              </View>
            ) : null}

            {!batchListLoading && batchResults ? (
              <View style={styles.filterRow}>
                {(
                  [
                    { key: 'all', label: 'ALL' },
                    { key: 'running', label: 'RUNNING' },
                    { key: 'completed', label: 'COMPLETED' },
                  ] as const
                ).map((option) => {
                  const selected = filter === option.key;
                  return (
                    <Pressable
                      key={option.key}
                      accessibilityRole="button"
                      onPress={() => setFilter(option.key)}
                      style={[styles.filterChip, selected && styles.filterChipSelected]}
                    >
                      <Text
                        style={[
                          styles.filterChipText,
                          selected && styles.filterChipTextSelected,
                        ]}
                      >
                        {option.label}
                      </Text>
                    </Pressable>
                  );
                })}
              </View>
            ) : null}
          </View>
        }
        ListEmptyComponent={
          !batchListLoading && !batchListError && hasSearched ? (
            <View style={styles.emptyBox}>
              <Text style={styles.emptyText}>{emptyMessage}</Text>
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
    gap: spacing.md,
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
    lineHeight: 19,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
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
  filterRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  filterChip: {
    flex: 1,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.pill,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    backgroundColor: colors.surface,
  },
  filterChipSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  filterChipText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  filterChipTextSelected: {
    color: colors.textOnPrimary,
  },
  batchCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
    ...shadows.soft,
  },
  batchHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  batchCode: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  batchMeta: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  statusBadge: {
    borderRadius: radii.pill,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
  },
  statusBadgeRunning: {
    backgroundColor: colors.success,
  },
  statusBadgeCompleted: {
    backgroundColor: colors.border,
  },
  statusBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  statusBadgeTextRunning: {
    color: colors.textOnPrimary,
  },
  statusBadgeTextCompleted: {
    color: colors.textSecondary,
  },
  actionsColumn: {
    gap: spacing.xs,
  },
  primaryAction: {
    minHeight: 44,
    borderRadius: radii.pill,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
  },
  primaryActionPressed: {
    backgroundColor: colors.primaryDark,
  },
  primaryActionText: {
    color: colors.textOnPrimary,
    fontSize: 14,
    fontWeight: '700',
  },
  subtleAction: {
    alignSelf: 'center',
    paddingVertical: spacing.xs,
  },
  subtleActionText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textSecondary,
    textDecorationLine: 'underline',
  },
  emptyBox: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 13,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
