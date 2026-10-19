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
import { SmokehouseIdInput } from '../../features/smokehouse/SmokehouseIdInput';
import { createSmokehouseBatch, SmokehouseApiError } from '../../services/smokehouseApi';
import type { LayerCode, SmokehouseLayerCreate } from '../../types/smokehouse';

const LAYER_CODES: LayerCode[] = ['L1', 'L2', 'L3'];

type LayerInput = {
  sheets: string;
};

type LayerInputs = Record<LayerCode, LayerInput>;

const EMPTY_LAYER_INPUTS: LayerInputs = {
  L1: { sheets: '' },
  L2: { sheets: '' },
  L3: { sheets: '' },
};

export default function BatchSetup() {
  const { setBatch } = useSmokehouseFlow();

  const [batchCode, setBatchCode] = useState('');
  const [smokehouseId, setSmokehouseId] = useState('');
  const [widthInput, setWidthInput] = useState('');
  const [lengthInput, setLengthInput] = useState('');
  const [layerInputs, setLayerInputs] = useState<LayerInputs>(EMPTY_LAYER_INPUTS);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateSheets(code: LayerCode, value: string) {
    setLayerInputs((previous) => ({
      ...previous,
      [code]: { ...previous[code], sheets: value },
    }));
  }

  const widthValue = Number(widthInput);
  const lengthValue = Number(lengthInput);
  const hasValidDimensions =
    widthInput.trim() !== '' &&
    lengthInput.trim() !== '' &&
    Number.isFinite(widthValue) &&
    Number.isFinite(lengthValue) &&
    widthValue > 0 &&
    lengthValue > 0;
  const calculatedLayerArea = hasValidDimensions ? widthValue * lengthValue : null;

  function validate(): { errors: Record<string, string>; layers: SmokehouseLayerCreate[] } | null {
    const nextErrors: Record<string, string> = {};
    const layers: SmokehouseLayerCreate[] = [];

    if (!batchCode.trim()) {
      nextErrors.batchCode = 'Batch code is required.';
    }

    if (!smokehouseId.trim()) {
      nextErrors.smokehouseId = 'Smokehouse ID is required.';
    }

    if (!widthInput.trim() || !Number.isFinite(widthValue) || widthValue <= 0) {
      nextErrors.width = 'Enter a positive number.';
    }

    if (!lengthInput.trim() || !Number.isFinite(lengthValue) || lengthValue <= 0) {
      nextErrors.length = 'Enter a positive number.';
    }

    const area =
      !nextErrors.width && !nextErrors.length ? widthValue * lengthValue : null;

    for (const code of LAYER_CODES) {
      const input = layerInputs[code];

      const sheets = Number(input.sheets);
      if (!input.sheets.trim() || !Number.isInteger(sheets) || sheets <= 0) {
        nextErrors[`${code}.sheets`] = 'Enter a positive whole number.';
      }

      if (!nextErrors[`${code}.sheets`] && area !== null) {
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
        keyboardShouldPersistTaps="handled"
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
          <SmokehouseIdInput
            value={smokehouseId}
            onChangeText={setSmokehouseId}
            error={errors.smokehouseId}
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Smokehouse Dimensions</Text>
          <FormField
            label="Width (m)"
            value={widthInput}
            onChangeText={setWidthInput}
            placeholder="e.g. 4"
            keyboardType="decimal-pad"
            error={errors.width}
          />
          <FormField
            label="Length (m)"
            value={lengthInput}
            onChangeText={setLengthInput}
            placeholder="e.g. 6"
            keyboardType="decimal-pad"
            error={errors.length}
          />
          <View style={styles.readOnlyField}>
            <Text style={styles.readOnlyLabel}>Calculated Layer Area</Text>
            <View style={styles.readOnlyBox}>
              <Text style={styles.readOnlyValue}>
                {calculatedLayerArea !== null
                  ? `${calculatedLayerArea.toFixed(2)} m²`
                  : '— m²'}
              </Text>
            </View>
          </View>
        </View>

        {LAYER_CODES.map((code) => (
          <View key={code} style={styles.card}>
            <Text style={styles.cardTitle}>{code} Configuration</Text>
            <FormField
              label="Number of Sheets"
              value={layerInputs[code].sheets}
              onChangeText={(value) => updateSheets(code, value)}
              placeholder="e.g. 40"
              keyboardType="number-pad"
              error={errors[`${code}.sheets`]}
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
  readOnlyField: {
    gap: spacing.xs,
  },
  readOnlyLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  readOnlyBox: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background,
  },
  readOnlyValue: {
    fontSize: typography.body.fontSize,
    fontWeight: '700',
    color: colors.textPrimary,
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
