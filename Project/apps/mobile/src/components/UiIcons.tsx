import { StyleSheet, Text, View } from 'react-native';

import { colors } from '../constants/theme';

type IconProps = {
  size?: number;
  tone?: string;
};

/** Settings cog drawn from plain views (no icon font needed). */
export function GearIcon({ size = 22, tone = colors.textOnPrimary }: IconProps) {
  const tooth = size * 0.2;
  const ring = size * 0.66;

  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      {[0, 45, 90, 135].map((deg) => (
        <View
          key={deg}
          style={{
            position: 'absolute',
            width: tooth,
            height: size,
            borderRadius: tooth / 3,
            backgroundColor: tone,
            transform: [{ rotate: `${deg}deg` }],
          }}
        />
      ))}
      <View
        style={{
          width: ring,
          height: ring,
          borderRadius: ring / 2,
          backgroundColor: tone,
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <View
          style={{
            width: ring * 0.42,
            height: ring * 0.42,
            borderRadius: ring,
            backgroundColor: colors.primary,
          }}
        />
      </View>
    </View>
  );
}

/** Map pin. */
export function PinIcon({ size = 12, tone = colors.textOnPrimary }: IconProps) {
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <View
        style={{
          width: size * 0.8,
          height: size * 0.8,
          borderTopLeftRadius: size,
          borderTopRightRadius: size,
          borderBottomLeftRadius: size,
          borderBottomRightRadius: 0,
          backgroundColor: tone,
          transform: [{ rotate: '45deg' }],
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <View
          style={{
            width: size * 0.28,
            height: size * 0.28,
            borderRadius: size,
            backgroundColor: colors.primary,
          }}
        />
      </View>
    </View>
  );
}

export function ChevronIcon({ size = 18, tone = colors.primary }: IconProps) {
  return <Text style={[styles.chevron, { fontSize: size, lineHeight: size + 2, color: tone }]}>›</Text>;
}

/** Sun partly behind a cloud — used for the forecast slide. */
export function SunCloudIcon({ size = 56 }: { size?: number }) {
  const sun = size * 0.55;
  return (
    <View style={{ width: size, height: size }}>
      <View
        style={{
          position: 'absolute',
          top: size * 0.06,
          right: size * 0.06,
          width: sun,
          height: sun,
          borderRadius: sun / 2,
          backgroundColor: '#F4C24F',
        }}
      />
      <View
        style={{
          position: 'absolute',
          bottom: size * 0.14,
          left: size * 0.04,
          width: size * 0.8,
          height: size * 0.34,
          borderRadius: size,
          backgroundColor: colors.surface,
          borderWidth: 1.5,
          borderColor: colors.primaryLight,
        }}
      />
      <View
        style={{
          position: 'absolute',
          bottom: size * 0.3,
          left: size * 0.2,
          width: size * 0.38,
          height: size * 0.38,
          borderRadius: size,
          backgroundColor: colors.surface,
          borderWidth: 1.5,
          borderColor: colors.primaryLight,
          borderBottomColor: 'transparent',
        }}
      />
    </View>
  );
}

/** Small round status dot. */
export function StatusDot({ tone, size = 8 }: { tone: string; size?: number }) {
  return <View style={{ width: size, height: size, borderRadius: size / 2, backgroundColor: tone }} />;
}

const styles = StyleSheet.create({
  chevron: {
    fontWeight: '700',
  },
});
