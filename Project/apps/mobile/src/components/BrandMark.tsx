import { StyleSheet, View } from 'react-native';

import { colors, shadows } from '../constants/theme';

type BrandMarkProps = {
  size?: number;
};

export function BrandMark({ size = 96 }: BrandMarkProps) {
  const leafSize = size * 0.46;

  return (
    <View
      style={[
        styles.badge,
        { width: size, height: size, borderRadius: size / 2 },
      ]}
    >
      <View
        style={[
          styles.leaf,
          {
            width: leafSize,
            height: leafSize,
            borderTopLeftRadius: leafSize / 2,
            borderBottomRightRadius: leafSize / 2,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.soft,
  },
  leaf: {
    backgroundColor: colors.background,
    borderTopRightRadius: 0,
    borderBottomLeftRadius: 0,
    transform: [{ rotate: '45deg' }],
  },
});
