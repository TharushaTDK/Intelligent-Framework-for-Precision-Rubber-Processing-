import { router } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BrandMark } from '../components/BrandMark';
import {
  DropletMark,
  LeafMark,
  ThermometerMark,
  WaveMark,
} from '../components/ComponentMarks';
import { FeatureCard } from '../components/FeatureCard';
import { colors, spacing, typography } from '../constants/theme';

export default function Home() {
  return (
    <View style={styles.root}>
      <View style={styles.blob} />

      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <BrandMark size={44} />
            <View style={styles.headerText}>
              <Text style={styles.appName}>RubberSense</Text>
              <Text style={styles.headerSubtitle}>
                Choose a smart tool for your rubber process
              </Text>
            </View>
          </View>

          <View style={styles.sectionIntro}>
            <Text style={styles.sectionTitle}>Smart Tools</Text>
            <Text style={styles.sectionCaption}>
              Precision support from field to processing.
            </Text>
          </View>

          <View style={styles.grid}>
            <FeatureCard
              label="Component 01"
              title="Let's Set the Latex"
              description="Coagulation guidance for better latex processing."
              icon={<DropletMark />}
              onPress={() => router.push('/latex')}
            />

            <FeatureCard
              label="Component 02"
              title="How's Today Your Rubber Plant?"
              description="Weather-aware support for tapping decisions."
              icon={<LeafMark />}
              onPress={() => router.push('/plant-advisory')}
            />

            <FeatureCard
              label="Component 03"
              title="Smart Smokehouse"
              description="Monitor drying progress and fire-risk conditions."
              icon={<ThermometerMark tone={colors.primaryDark} />}
              onPress={() => router.push('/smokehouse')}
              emphasized
              badge="Our Module"
            />

            <FeatureCard
              label="Component 04"
              title="How's the Moisture?"
              description="Estimate sheet moisture using acoustic analysis."
              icon={<WaveMark />}
              onPress={() => router.push('/moisture')}
            />
          </View>
        </ScrollView>
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
  blob: {
    position: 'absolute',
    top: -100,
    right: -90,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: colors.primaryLight,
    opacity: 0.2,
  },
  safeArea: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.xl,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  headerText: {
    flex: 1,
  },
  appName: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
  },
  headerSubtitle: {
    marginTop: 2,
    fontSize: 13,
    color: colors.textSecondary,
  },
  sectionIntro: {
    marginTop: spacing.xl,
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  sectionCaption: {
    marginTop: 2,
    fontSize: 13,
    color: colors.textSecondary,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
});
