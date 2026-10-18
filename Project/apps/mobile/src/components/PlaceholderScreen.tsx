import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors, spacing, typography } from '../constants/theme';

type PlaceholderScreenProps = {
  title: string;
  status: string;
  note: string;
};

export function PlaceholderScreen({ title, status, note }: PlaceholderScreenProps) {
  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable
        accessibilityRole="button"
        onPress={() => router.back()}
        style={styles.backButton}
      >
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <View style={styles.content}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.status}>{status}</Text>
        <Text style={styles.note}>{note}</Text>
      </View>
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
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.sm,
  },
  title: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
    textAlign: 'center',
  },
  status: {
    fontSize: typography.body.fontSize,
    fontWeight: '700',
    color: colors.primary,
    textAlign: 'center',
  },
  note: {
    marginTop: spacing.sm,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
  },
});
