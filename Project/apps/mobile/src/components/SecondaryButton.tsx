import { Pressable, StyleSheet, Text, type GestureResponderEvent } from 'react-native';

import { colors, radii, spacing, typography } from '../constants/theme';

type SecondaryButtonProps = {
  label: string;
  onPress: (event: GestureResponderEvent) => void;
};

export function SecondaryButton({ label, onPress }: SecondaryButtonProps) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        pressed && styles.buttonPressed,
      ]}
    >
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 50,
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    backgroundColor: 'transparent',
  },
  buttonPressed: {
    backgroundColor: colors.background,
  },
  label: {
    color: colors.primary,
    fontSize: typography.button.fontSize,
    fontWeight: typography.button.fontWeight,
    letterSpacing: typography.button.letterSpacing,
  },
});
