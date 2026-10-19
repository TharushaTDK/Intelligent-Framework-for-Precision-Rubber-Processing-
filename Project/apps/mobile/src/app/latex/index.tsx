import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';

const STEPS = [
  { number: '1', title: 'Enter the batch', text: 'Dry rubber content and latex volume.' },
  { number: '2', title: 'Choose what you need', text: 'Water and acid amount, coagulation time or both.' },
  { number: '3', title: 'Get the recommendation', text: 'Predicted by the trained models and saved as a batch.' },
];

export default function LatexOverview() {
  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>Let's Set the Latex</Text>
        <Text style={styles.subtitle}>
          Get the water and 1% formic acid solution to add to a latex batch and the expected
          coagulation time.
        </Text>

        <View style={styles.notice}>
          <Text style={styles.noticeText}>
            Decision-support prototype. Always follow the approved processing procedure.
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

        <PrimaryButton label="New Batch" onPress={() => router.push('/latex/batch')} />
        <SecondaryButton label="Batch History" onPress={() => router.push('/latex/history')} />
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
});
