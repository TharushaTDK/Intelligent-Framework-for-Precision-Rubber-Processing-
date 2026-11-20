import { Image, StyleSheet, View } from 'react-native';

import { colors, shadows } from '../constants/theme';

const logo = require('../../assets/brand-logo.png');

type BrandMarkProps = {
  size?: number;
};

export function BrandMark({ size = 96 }: BrandMarkProps) {
  return (
    <View
      style={[
        styles.badge,
        { width: size, height: size, borderRadius: size / 2 },
      ]}
    >
      <Image
        source={logo}
        style={{ width: size, height: size, borderRadius: size / 2 }}
        resizeMode="cover"
        accessibilityIgnoresInvertColors
      />
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.soft,
  },
});
