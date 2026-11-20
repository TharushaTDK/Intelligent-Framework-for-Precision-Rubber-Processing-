import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { ChevronIcon } from '../../components/UiIcons';
import { colors, radii, shadows, spacing } from '../../constants/theme';
import { COMPONENTS } from './componentCatalog';

export function ToolGrid() {
  return (
    <View style={styles.grid}>
      {COMPONENTS.map((info) => (
        <Pressable
          key={info.key}
          accessibilityRole="button"
          accessibilityLabel={info.title}
          onPress={() => router.push(info.route)}
          style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}
        >
          <View style={styles.cardTop}>
            <View style={styles.iconBox}>{info.icon(22)}</View>
            <ChevronIcon size={20} tone={colors.primaryLight} />
          </View>
          <Text style={styles.number}>COMPONENT {info.number}</Text>
          <Text style={styles.title} numberOfLines={2}>
            {info.title}
          </Text>
          <Text style={styles.subtitle} numberOfLines={2}>
            {info.subtitle}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    rowGap: spacing.sm + 4,
  },
  card: {
    width: '48%',
    minHeight: 158,
    backgroundColor: colors.surface,
    borderRadius: radii.md + 2,
    padding: spacing.sm + 6,
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.soft,
  },
  cardPressed: {
    backgroundColor: '#EEF6F0',
  },
  cardTop: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  iconBox: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: '#EAF6EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  number: {
    marginTop: spacing.sm + 4,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.1,
    color: colors.success,
  },
  title: {
    marginTop: 2,
    fontSize: 14.5,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  subtitle: {
    marginTop: 4,
    fontSize: 12,
    lineHeight: 16,
    color: colors.textSecondary,
  },
});
