import { router } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { PredictionHistoryCard } from '../../features/smokehouse/PredictionHistoryCard';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import { getSmokehouseLayerHistory, SmokehouseApiError } from '../../services/smokehouseApi';
import type {
  LayerCode,
  SmokehouseHistoryItemResponse,
  SmokehouseLayerHistoryResponse,
} from '../../types/smokehouse';

const LAYER_OPTIONS: LayerCode[] = ['L1', 'L2', 'L3'];

export default function PredictionHistory() {
  const { batch, selectedLayer } = useSmokehouseFlow();

  const [batchCodeInput, setBatchCodeInput] = useState(batch?.batch_code ?? '');
  const [layerInput, setLayerInput] = useState<LayerCode | null>(selectedLayer ?? null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const [loading, setLoading] = useState(false);
  const [hasQueried, setHasQueried] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [history, setHistory] = useState<SmokehouseLayerHistoryResponse | null>(null);

  const autoLoadedRef = useRef(false);

  async function loadHistory(batchCode: string, layerCode: LayerCode) {
    setLoading(true);
    setQueryError(null);

    try {
      const response = await getSmokehouseLayerHistory(batchCode, layerCode);
      setHistory(response);
      setHasQueried(true);
    } catch (error) {
      setHistory(null);
      setHasQueried(true);

      if (error instanceof SmokehouseApiError && error.status === 404) {
        setQueryError('No history was found for that batch and layer.');
      } else if (error instanceof SmokehouseApiError) {
        setQueryError(error.message);
      } else {
        setQueryError('Could not reach the backend. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!autoLoadedRef.current && batch && selectedLayer) {
      autoLoadedRef.current = true;
      loadHistory(batch.batch_code, selectedLayer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleLoadHistory() {
    const nextErrors: Record<string, string> = {};

    if (!batchCodeInput.trim()) {
      nextErrors.batchCode = 'Batch code is required.';
    }

    if (!layerInput) {
      nextErrors.layer = 'Select a layer.';
    }

    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    loadHistory(batchCodeInput.trim(), layerInput as LayerCode);
  }

  function handleRefresh() {
    if (!batchCodeInput.trim() || !layerInput) {
      return;
    }

    loadHistory(batchCodeInput.trim(), layerInput);
  }

  const displayItems: SmokehouseHistoryItemResponse[] = history
    ? [...history.items].reverse()
    : [];

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
        data={displayItems}
        keyExtractor={(item) => String(item.reading.id)}
        renderItem={({ item }) => <PredictionHistoryCard item={item} />}
        contentContainerStyle={styles.listContent}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        ListHeaderComponent={
          <View style={styles.headerSection}>
            <Text style={styles.title}>Prediction History</Text>
            <Text style={styles.subtitle}>
              Review previous readings and model predictions for a
              smokehouse layer.
            </Text>

            <View style={styles.card}>
              <FormField
                label="Batch Code"
                value={batchCodeInput}
                onChangeText={setBatchCodeInput}
                placeholder="e.g. SH01_B01"
                error={errors.batchCode}
              />

              <Text style={styles.fieldLabel}>Layer</Text>
              <View style={styles.layerRow}>
                {LAYER_OPTIONS.map((code) => {
                  const selected = layerInput === code;
                  return (
                    <Pressable
                      key={code}
                      accessibilityRole="button"
                      onPress={() => setLayerInput(code)}
                      style={[
                        styles.layerOption,
                        selected && styles.layerOptionSelected,
                      ]}
                    >
                      <Text
                        style={[
                          styles.layerOptionText,
                          selected && styles.layerOptionTextSelected,
                        ]}
                      >
                        {code}
                      </Text>
                    </Pressable>
                  );
                })}
              </View>
              {errors.layer ? (
                <Text style={styles.fieldError}>{errors.layer}</Text>
              ) : null}

              <PrimaryButton label="Load History" onPress={handleLoadHistory} />
            </View>

            {loading ? (
              <View style={styles.loadingRow}>
                <ActivityIndicator color={colors.primary} />
                <Text style={styles.loadingText}>Loading history...</Text>
              </View>
            ) : null}

            {!loading && queryError ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{queryError}</Text>
              </View>
            ) : null}

            {!loading && history ? (
              <View style={styles.summaryCard}>
                <View style={styles.summaryRow}>
                  <Text style={styles.summaryLabel}>Batch</Text>
                  <Text style={styles.summaryValue}>{history.batch_code}</Text>
                </View>
                <View style={styles.summaryRow}>
                  <Text style={styles.summaryLabel}>Smokehouse</Text>
                  <Text style={styles.summaryValue}>{history.smokehouse_id}</Text>
                </View>
                <View style={styles.summaryRow}>
                  <Text style={styles.summaryLabel}>Layer</Text>
                  <Text style={styles.summaryValue}>{history.layer_code}</Text>
                </View>
                <View style={styles.summaryRow}>
                  <Text style={styles.summaryLabel}>Total Readings</Text>
                  <Text style={styles.summaryValue}>{history.items.length}</Text>
                </View>

                <Pressable
                  accessibilityRole="button"
                  onPress={handleRefresh}
                  style={styles.refreshButton}
                >
                  <Text style={styles.refreshButtonText}>Refresh</Text>
                </Pressable>
              </View>
            ) : null}

            {!loading &&
            !queryError &&
            hasQueried &&
            history &&
            history.items.length === 0 ? (
              <View style={styles.emptyBox}>
                <Text style={styles.emptyText}>
                  No readings found for this batch and layer.
                </Text>
              </View>
            ) : null}
          </View>
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
  fieldLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  layerRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  layerOption: {
    flex: 1,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    backgroundColor: colors.background,
  },
  layerOptionSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  layerOptionText: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  layerOptionTextSelected: {
    color: colors.textOnPrimary,
  },
  fieldError: {
    fontSize: 12,
    color: colors.critical,
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
  refreshButton: {
    marginTop: spacing.sm,
    alignSelf: 'flex-start',
  },
  refreshButtonText: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primary,
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
