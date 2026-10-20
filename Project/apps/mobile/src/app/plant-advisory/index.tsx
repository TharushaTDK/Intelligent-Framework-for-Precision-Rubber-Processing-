import { router } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { getModelCatalog, toReadableMessage } from '../../services/plantAdvisoryApi';

const STEPS = [
  { number: '1', title: 'Pick your field', text: 'Drop a pin on the map or use your current location.' },
  { number: '2', title: 'Get the weather', text: 'Temperature, humidity, wind and rainfall for that place and time.' },
  { number: '3', title: 'Get the advisory', text: 'Estimated DRC, tapping suitability and latex price.' },
];

type ServiceState =
  | { kind: 'loading' }
  | { kind: 'ready'; modelCount: number }
  | { kind: 'error'; message: string };

export default function PlantAdvisoryOverview() {
  const [service, setService] = useState<ServiceState>({ kind: 'loading' });

  // one real request so we know the backend and the three models are there
  const checkService = useCallback(async () => {
    setService({ kind: 'loading' });
    try {
      const catalog = await getModelCatalog();
      setService({ kind: 'ready', modelCount: catalog.models.length });
    } catch (error) {
      setService({ kind: 'error', message: toReadableMessage(error) });
    }
  }, []);

  useEffect(() => {
    checkService();
  }, [checkService]);

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>How's Today Your Rubber Plant?</Text>
        <Text style={styles.subtitle}>
          Check today's weather at your rubber field and get a tapping and collection advisory.
        </Text>

        <View style={styles.notice}>
          <Text style={styles.noticeText}>
            Decision-support prototype. Results are model estimates, not field measurements.
          </Text>
        </View>

        <View style={styles.stepsCard}>
          {STEPS.map((step) => (
            <View key={step.number} style={styles.stepRow}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>{step.number}</Text>
              </View>
              <View style={styles.stepText}>
                <Text style={styles.stepTitle}>{step.title}</Text>
                <Text style={styles.stepCaption}>{step.text}</Text>
              </View>
            </View>
          ))}
        </View>

        <PrimaryButton label="Start Advisory" onPress={() => router.push('/plant-advisory/location')} />

        <View style={styles.statusCard}>
          {service.kind === 'loading' && (
            <View style={styles.statusRow}>
              <ActivityIndicator color={colors.primary} />
              <Text style={styles.statusText}>Checking the advisory service...</Text>
            </View>
          )}

          {service.kind === 'ready' && (
            <View style={styles.statusRow}>
              <View style={[styles.statusDot, { backgroundColor: colors.success }]} />
              <Text style={styles.statusText}>
                Advisory service is online ({service.modelCount} models ready)
              </Text>
            </View>
          )}

          {service.kind === 'error' && (
            <View style={styles.errorBlock}>
              <View style={styles.statusRow}>
                <View style={[styles.statusDot, { backgroundColor: colors.critical }]} />
                <Text style={[styles.statusText, styles.errorTitle]}>Service unavailable</Text>
              </View>
              <Text style={styles.errorMessage}>{service.message}</Text>
              <SecondaryButton label="Try Again" onPress={checkService} />
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// layout and spacing follow the Let's Set the Latex overview so the
// two components feel like one app
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
    paddingBottom: spacing.xl,
    gap: spacing.lg,
  },
  title: {
    fontSize: typography.display.fontSize,
    fontWeight: typography.display.fontWeight,
    color: colors.textPrimary,
  },
  subtitle: {
    fontSize: typography.body.fontSize,
    color: colors.textSecondary,
    lineHeight: 22,
  },
  notice: {
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  noticeText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  stepsCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.md,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  stepNumber: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumberText: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.primaryDark,
  },
  stepText: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  stepCaption: {
    marginTop: 2,
    fontSize: 13,
    color: colors.textSecondary,
  },
  statusCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: {
    flex: 1,
    fontSize: 14,
    color: colors.textPrimary,
  },
  errorBlock: {
    gap: spacing.sm,
  },
  errorTitle: {
    fontWeight: '700',
    color: colors.critical,
  },
  errorMessage: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 19,
  },
});
