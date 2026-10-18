import { router } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BrandMark } from '../components/BrandMark';
import { PrimaryButton } from '../components/PrimaryButton';
import { colors, radii, spacing, typography } from '../constants/theme';

export default function Welcome() {
  return (
    <View style={styles.root}>
      <View style={[styles.blob, styles.blobTopLeft]} />
      <View style={[styles.blob, styles.blobTopRight]} />

      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        <View style={styles.content}>
          <View style={styles.hero}>
            <BrandMark size={96} />

            <Text style={styles.title}>RubberSense</Text>

            <Text style={styles.tagline}>SENSE. PREDICT. IMPROVE.</Text>

            <Text style={styles.description}>
              Smarter insights for precision rubber processing.
            </Text>

            <View style={styles.divider} />
          </View>

          <View style={styles.footer}>
            <PrimaryButton
              label="Get Started"
              onPress={() => router.push('/home')}
            />
            <Text style={styles.footerCaption}>
              Precision Rubber Intelligence
            </Text>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
    overflow: 'hidden',
  },
  safeArea: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.xl,
  },
  blob: {
    position: 'absolute',
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: colors.primaryLight,
    opacity: 0.25,
  },
  blobTopLeft: {
    top: -90,
    left: -70,
  },
  blobTopRight: {
    top: -40,
    right: -90,
    width: 160,
    height: 160,
    borderRadius: 80,
  },
  hero: {
    alignItems: 'center',
    gap: spacing.md,
  },
  divider: {
    width: 48,
    height: 3,
    borderRadius: radii.pill,
    backgroundColor: colors.primaryLight,
    marginTop: spacing.sm,
  },
  title: {
    fontSize: typography.display.fontSize,
    fontWeight: typography.display.fontWeight,
    letterSpacing: typography.display.letterSpacing,
    color: colors.textPrimary,
    marginTop: spacing.md,
  },
  tagline: {
    fontSize: typography.tagline.fontSize,
    fontWeight: typography.tagline.fontWeight,
    letterSpacing: typography.tagline.letterSpacing,
    color: colors.primary,
  },
  description: {
    fontSize: typography.body.fontSize,
    color: colors.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
    marginTop: spacing.xs,
  },
  footer: {
    alignItems: 'center',
    gap: spacing.md,
  },
  footerCaption: {
    fontSize: typography.caption.fontSize,
    fontWeight: typography.caption.fontWeight,
    letterSpacing: typography.caption.letterSpacing,
    color: colors.textSecondary,
    textTransform: 'uppercase',
  },
});
