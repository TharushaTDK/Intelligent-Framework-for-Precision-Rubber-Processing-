import { router } from 'expo-router';
import { useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import { createSmokehouseBatch, SmokehouseApiError } from '../../services/smokehouseApi';
import type { LayerCode, SmokehouseLayerCreate } from '../../types/smokehouse';

const LAYER_CODES: LayerCode[] = ['L1', 'L2', 'L3'];

type LayerInput = {
  sheets: string;
  area: string;
};

type LayerInputs = Record<LayerCode, LayerInput>;

const EMPTY_LAYER_INPUTS: LayerInputs = {
  L1: { sheets: '', area: '' },
  L2: { sheets: '', area: '' },
  L3: { sheets: '', area: '' },
};

export default function BatchSetup() {
  const { setBatch } = useSmokehouseFlow();

  const [batchCode, setBatchCode] = useState('');
  const [smokehouseId, setSmokehouseId] = useState('');
  const [layerInputs, setLayerInputs] = useState<LayerInputs>(EMPTY_LAYER_INPUTS);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateLayerField(code: LayerCode, field: keyof LayerInput, value: string) {
    setLayerInputs((previous) => ({
      ...previous,
      [code]: { ...previous[code], [field]: value },
    }));
  }

  function validate(): { errors: Record<string, string>; layers: SmokehouseLayerCreate[] } | null {
    const nextErrors: Record<string, string> = {};
    const layers: SmokehouseLayerCreate[] = [];

    if (!batchCode.trim()) {
      nextErrors.batchCode = 'Batch code is required.';
    }

    if (!smokehouseId.trim()) {
      nextErrors.smokehouseId = 'Smokehouse ID is required.';
    }

    for (const code of LAYER_CODES) {
      const input = layerInputs[code];

      const sheets = Number(input.sheets);
      if (!input.sheets.trim() || !Number.isInteger(sheets) || sheets <= 0) {
        nextErrors[`${code}.sheets`] = 'Enter a positive whole number.';
      }

      const area = Number(input.area);
      if (!input.area.trim() || !Number.isFinite(area) || area <= 0) {
        nextErrors[`${code}.area`] = 'Enter a positive number.';
      }

      if (!nextErrors[`${code}.sheets`] && !nextErrors[`${code}.area`]) {
        layers.push({
          layer_code: code,
          number_of_sheets: sheets,
          layer_area_m2: area,
        });
      }
    }

    if (Object.keys(nextErrors).length > 0) {
      return { errors: nextErrors, layers: [] };
    }

    return { errors: {}, layers };
  }

  async function handleSubmit() {
    const result = validate();

    if (!result || result.layers.length !== 3) {
      setErrors(result?.errors ?? {});
      return;
    }

    setErrors({});
    setSubmitError(null);
    setLoading(true);

    try {
      const response = await createSmokehouseBatch({
        batch_code: batchCode.trim(),
        smokehouse_id: smokehouseId.trim(),
        layers: result.layers,
      });

      setBatch(response);
      router.push('/smokehouse/layers');
    } catch (error) {
      setSubmitError(
        error instanceof SmokehouseApiError
          ? error.message
          : 'Could not reach the backend. Please try again.',
      );
    } finally {
      setLoading(false);
    }
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
        <Text style={styles.title}>Batch Setup</Text>
        <Text style={styles.subtitle}>
          Configure the drying batch and its three smokehouse layers.
        </Text>

        <View style={styles.card}>
          <FormField
            label="Batch Code"
            value={batchCode}
            onChangeText={setBatchCode}
            placeholder="e.g. SH01_B01"
            error={errors.batchCode}
          />
          <FormField
            label="Smokehouse ID"
            value={smokehouseId}
            onChangeText={setSmokehouseId}
            placeholder="e.g. SH01"
            error={errors.smokehouseId}
          />
        </View>

        {LAYER_CODES.map((code) => (
          <View key={code} style={styles.card}>
            <Text style={styles.cardTitle}>{code} Configuration</Text>
            <FormField
              label="Number of Sheets"
              value={layerInputs[code].sheets}
              onChangeText={(value) => updateLayerField(code, 'sheets', value)}
              placeholder="e.g. 40"
              keyboardType="number-pad"
              error={errors[`${code}.sheets`]}
            />
            <FormField
              label="Layer Area (m²)"
              value={layerInputs[code].area}
              onChangeText={(value) => updateLayerField(code, 'area', value)}
              placeholder="e.g. 8.5"
              keyboardType="decimal-pad"
              error={errors[`${code}.area`]}
            />
          </View>
        ))}

        {submitError ? (
          <View style={styles.submitErrorBox}>
            <Text style={styles.submitErrorText}>{submitError}</Text>
          </View>
        ) : null}

        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Creating batch...</Text>
          </View>
        ) : (
          <PrimaryButton label="Save & Continue" onPress={handleSubmit} />
        )}
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
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  submitErrorBox: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.critical,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  submitErrorText: {
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
