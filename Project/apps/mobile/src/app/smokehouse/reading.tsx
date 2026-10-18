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
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { useSmokehouseFlow } from '../../features/smokehouse/SmokehouseFlowContext';
import { verifyPersistedReading } from '../../features/smokehouse/verifyPersistedReading';
import {
  createSmokehouseReading,
  getSmokehouseLatestStatus,
  SmokehouseApiError,
} from '../../services/smokehouseApi';

export default function ReadingEntry() {
  const { batch, selectedLayer, setLastReading } = useSmokehouseFlow();

  const [temperature, setTemperature] = useState('');
  const [humidity, setHumidity] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!batch) {
    return (
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No batch found</Text>
          <SecondaryButton
            label="Go to Batch Setup"
            onPress={() => router.replace('/smokehouse/batch')}
          />
        </View>
      </SafeAreaView>
    );
  }

  if (!selectedLayer) {
    return (
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No layer selected</Text>
          <SecondaryButton
            label="Choose a Layer"
            onPress={() => router.replace('/smokehouse/layers')}
          />
        </View>
      </SafeAreaView>
    );
  }

  function validate(): { temperature_c: number; humidity_pct: number } | null {
    const nextErrors: Record<string, string> = {};

    const temperatureValue = Number(temperature);
    if (!temperature.trim() || !Number.isFinite(temperatureValue)) {
      nextErrors.temperature = 'Enter a valid temperature.';
    }

    const humidityValue = Number(humidity);
    if (
      !humidity.trim() ||
      !Number.isFinite(humidityValue) ||
      humidityValue < 0 ||
      humidityValue > 100
    ) {
      nextErrors.humidity = 'Enter a humidity value between 0 and 100.';
    }

    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      return null;
    }

    return { temperature_c: temperatureValue, humidity_pct: humidityValue };
  }

  async function handleSubmit() {
    const values = validate();
    if (!values || !batch || !selectedLayer) {
      return;
    }

    setSubmitError(null);
    setLoading(true);

    try {
      const posted = await createSmokehouseReading({
        batch_code: batch.batch_code,
        layer_code: selectedLayer,
        temperature_c: values.temperature_c,
        humidity_pct: values.humidity_pct,
        source: 'manual',
      });

      let verification;
      try {
        const latest = await getSmokehouseLatestStatus(
          batch.batch_code,
          selectedLayer,
        );
        verification = verifyPersistedReading(posted, latest);
      } catch {
        verification = { status: 'unavailable' as const, mismatchedFields: [] };
      }

      setLastReading(posted, verification);
      router.push('/smokehouse/result');
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
        <Text style={styles.title}>Reading Entry</Text>

        <View style={styles.summaryCard}>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Batch</Text>
            <Text style={styles.summaryValue}>{batch.batch_code}</Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Layer</Text>
            <Text style={styles.summaryValue}>{selectedLayer}</Text>
          </View>
        </View>

        <View style={styles.card}>
          <FormField
            label="Temperature (°C)"
            value={temperature}
            onChangeText={setTemperature}
            placeholder="e.g. 52.5"
            keyboardType="default"
            error={errors.temperature}
          />
          <FormField
            label="Relative Humidity (%)"
            value={humidity}
            onChangeText={setHumidity}
            placeholder="e.g. 68"
            keyboardType="decimal-pad"
            error={errors.humidity}
          />
        </View>

        {submitError ? (
          <View style={styles.submitErrorBox}>
            <Text style={styles.submitErrorText}>{submitError}</Text>
          </View>
        ) : null}

        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Running prediction...</Text>
          </View>
        ) : (
          <PrimaryButton label="Submit Reading" onPress={handleSubmit} />
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
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
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
});
