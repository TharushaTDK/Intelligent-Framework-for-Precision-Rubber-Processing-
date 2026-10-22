import { useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { getMeasurementById, toReadableMoistureError } from '../../services/moistureApi';
import type { AcousticMeasurementResponse } from '../../types/moisture';
import { MeasurementResultCard } from './MeasurementResultCard';

export function SavedResultLookup() {
  const [measurementIdInput, setMeasurementIdInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AcousticMeasurementResponse | null>(null);

  async function handleLookup() {
    const trimmed = measurementIdInput.trim();
    if (!trimmed) {
      setError('Please enter a measurement UUID to search.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const measurement = await getMeasurementById(trimmed);
      setResult(measurement);
    } catch (err: unknown) {
      setError(toReadableMoistureError(err));
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setMeasurementIdInput('');
    setError(null);
    setResult(null);
  }

  return (
    <View style={styles.container}>
      <View style={styles.headerBox}>
        <Text style={styles.heading}>Find Saved Measurement</Text>
        <Text style={styles.subheading}>
          Lookup an existing acoustic measurement result directly by its UUID.
        </Text>
      </View>

      {!result ? (
        <View style={styles.formCard}>
          <Text style={styles.inputLabel}>Measurement UUID</Text>
          <TextInput
            value={measurementIdInput}
            onChangeText={(text) => {
              setMeasurementIdInput(text);
              if (error) setError(null);
            }}
            placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000"
            placeholderTextColor={colors.textSecondary}
            autoCapitalize="none"
            autoCorrect={false}
            editable={!loading}
            style={[styles.input, error ? styles.inputError : null]}
          />

          {error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {loading ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color={colors.primary} />
              <Text style={styles.loadingText}>Searching backend storage...</Text>
            </View>
          ) : (
            <PrimaryButton label="Find Result" onPress={handleLookup} />
          )}
        </View>
      ) : (
        <View style={styles.resultWrap}>
          <MeasurementResultCard measurement={result} />
          <View style={styles.actionRow}>
            <SecondaryButton label="Find Another Result" onPress={handleClear} />
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.md,
  },
  headerBox: {
    gap: spacing.xs,
  },
  heading: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  subheading: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 18,
  },
  formCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.md,
  },
  inputLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 14,
    color: colors.textPrimary,
    backgroundColor: colors.background,
    fontFamily: 'monospace',
  },
  inputError: {
    borderColor: colors.critical,
  },
  errorBox: {
    backgroundColor: '#FFF5F5',
    borderLeftWidth: 3,
    borderLeftColor: colors.critical,
    padding: spacing.sm,
    borderRadius: radii.sm,
  },
  errorText: {
    fontSize: 12,
    color: colors.critical,
    lineHeight: 16,
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
  resultWrap: {
    gap: spacing.md,
  },
  actionRow: {
    marginTop: spacing.sm,
  },
});
