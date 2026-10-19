import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { BatchSummaryCard } from '../../features/latex/BatchSummaryCard';
import { addCoagulationTime, getRecommendation, toReadableMessage } from '../../services/latexApi';
import type { CoagulationRecommendationResponse } from '../../types/latex';

export default function BatchResult() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const [batch, setBatch] = useState<CoagulationRecommendationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addingTime, setAddingTime] = useState(false);

  useEffect(() => {
    getRecommendation(Number(id))
      .then(setBatch)
      .catch((err) => setError(toReadableMessage(err)))
      .finally(() => setLoading(false));
  }, [id]);

  async function handleAddTime() {
    if (!batch) return;
    setAddingTime(true);
    setError(null);
    try {
      setBatch(await addCoagulationTime(batch.id));
    } catch (err) {
      setError(toReadableMessage(err));
    } finally {
      setAddingTime(false);
    }
  }

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Recommendation</Text>
        <Text style={styles.subtitle}>The batch is saved and can be found in the batch history.</Text>

        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Loading the batch...</Text>
          </View>
        ) : null}

        {batch ? <BatchSummaryCard batch={batch} /> : null}

        {error ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}

        {batch && batch.coagulation_time_hours === null ? (
          addingTime ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color={colors.primary} />
              <Text style={styles.loadingText}>Predicting the coagulation time...</Text>
            </View>
          ) : (
            <PrimaryButton label="Add Coagulation Time" onPress={handleAddTime} />
          )
        ) : null}

        {!loading ? (
          <>
            <SecondaryButton label="Batch History" onPress={() => router.push('/latex/history')} />
            <SecondaryButton label="Done" onPress={() => router.dismissTo('/latex')} />
          </>
        ) : null}
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
