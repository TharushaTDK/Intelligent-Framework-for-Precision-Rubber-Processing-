import { router } from 'expo-router';
import { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { createRecommendation, MANUAL_ESTATE_ID, toReadableMessage } from '../../services/latexApi';
import type { CoagulationOutput } from '../../types/latex';

const OUTPUT_OPTIONS: { key: string; label: string; caption: string; outputs: CoagulationOutput[] }[] = [
  { key: 'water_acid', label: 'Water & Acid', caption: 'Water and 1% formic acid solution', outputs: ['water_acid'] },
  { key: 'coagulation_time', label: 'Coagulation Time', caption: 'How long the batch takes', outputs: ['coagulation_time'] },
  { key: 'both', label: 'Both', caption: 'All three values', outputs: ['water_acid', 'coagulation_time'] },
];

export default function NewBatch() {
  const [drc, setDrc] = useState('');
  const [volume, setVolume] = useState('');
  const [option, setOption] = useState('both');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function validate(): Record<string, string> {
    const nextErrors: Record<string, string> = {};
    const drcValue = Number(drc);
    const volumeValue = Number(volume);

    // same limits as the backend schema
    if (!drc.trim() || !Number.isFinite(drcValue) || drcValue <= 0 || drcValue > 100) {
      nextErrors.drc = 'Enter a DRC above 0 and up to 100 %.';
    }
    if (!volume.trim() || !Number.isFinite(volumeValue) || volumeValue <= 0) {
      nextErrors.volume = 'Enter a latex volume above 0 L.';
    }
    return nextErrors;
  }

  async function handleSubmit() {
    const nextErrors = validate();
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    const selected = OUTPUT_OPTIONS.find((item) => item.key === option) ?? OUTPUT_OPTIONS[2];

    setSubmitError(null);
    setLoading(true);

    try {
      const batch = await createRecommendation({
        estate_id: MANUAL_ESTATE_ID,
        drc: Number(drc),
        latex_volume_L: Number(volume),
        outputs: selected.outputs,
      });
      router.push({ pathname: '/latex/result', params: { id: String(batch.id) } });
    } catch (error) {
      setSubmitError(toReadableMessage(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.title}>New Batch</Text>
        <Text style={styles.subtitle}>Enter the latex batch details.</Text>

        <View style={styles.card}>
          <FormField
            label="Dry rubber content (%)"
            value={drc}
            onChangeText={setDrc}
            placeholder="e.g. 35.2"
            keyboardType="decimal-pad"
            error={errors.drc}
          />
          <FormField
            label="Latex volume (L)"
            value={volume}
            onChangeText={setVolume}
            placeholder="e.g. 100"
            keyboardType="decimal-pad"
            error={errors.volume}
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>What do you need?</Text>
          {OUTPUT_OPTIONS.map((item) => {
            const selected = item.key === option;
            return (
              <Pressable
                key={item.key}
                accessibilityRole="radio"
                accessibilityState={{ selected }}
                onPress={() => setOption(item.key)}
                style={[styles.option, selected && styles.optionSelected]}
              >
                <View style={[styles.radio, selected && styles.radioSelected]}>
                  {selected ? <View style={styles.radioDot} /> : null}
                </View>
                <View style={styles.optionText}>
                  <Text style={styles.optionLabel}>{item.label}</Text>
                  <Text style={styles.optionCaption}>{item.caption}</Text>
                </View>
              </Pressable>
            );
          })}
        </View>

        {submitError ? (
          <View style={styles.submitErrorBox}>
            <Text style={styles.submitErrorText}>{submitError}</Text>
          </View>
        ) : null}

        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Getting the recommendation...</Text>
          </View>
        ) : (
          <PrimaryButton label="Get Recommendation" onPress={handleSubmit} />
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
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    minHeight: 56,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  optionSelected: {
    borderColor: colors.primary,
    backgroundColor: colors.background,
  },
  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioSelected: {
    borderColor: colors.primary,
  },
  radioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
  optionText: {
    flex: 1,
  },
  optionLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  optionCaption: {
    marginTop: 2,
    fontSize: 12,
    color: colors.textSecondary,
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
