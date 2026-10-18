import { StyleSheet, View } from 'react-native';

import { colors } from '../constants/theme';

type MarkProps = {
  size?: number;
  tone?: string;
};

export function DropletMark({ size = 28, tone = colors.primary }: MarkProps) {
  return (
    <View
      style={[
        styles.droplet,
        {
          width: size,
          height: size,
          borderTopLeftRadius: size / 2,
          borderTopRightRadius: size / 2,
          borderBottomRightRadius: size / 2,
          backgroundColor: tone,
        },
      ]}
    />
  );
}

export function LeafMark({ size = 26, tone = colors.primary }: MarkProps) {
  return (
    <View
      style={[
        styles.leaf,
        {
          width: size,
          height: size,
          borderTopLeftRadius: size / 2,
          borderBottomRightRadius: size / 2,
          backgroundColor: tone,
        },
      ]}
    />
  );
}

export function ThermometerMark({ size = 28, tone = colors.primary }: MarkProps) {
  const stemWidth = size * 0.28;
  const bulbSize = size * 0.6;

  return (
    <View style={{ width: size, height: size, alignItems: 'center' }}>
      <View
        style={[
          styles.stem,
          {
            width: stemWidth,
            height: size - bulbSize * 0.55,
            borderRadius: stemWidth / 2,
            backgroundColor: tone,
          },
        ]}
      />
      <View
        style={[
          styles.bulb,
          {
            width: bulbSize,
            height: bulbSize,
            borderRadius: bulbSize / 2,
            backgroundColor: tone,
            marginTop: -(bulbSize * 0.45),
          },
        ]}
      />
    </View>
  );
}

export function WaveMark({ size = 28, tone = colors.primary }: MarkProps) {
  const bars = [0.45, 0.8, 1, 0.6];

  return (
    <View style={[styles.waveRow, { height: size, width: size }]}>
      {bars.map((ratio, index) => (
        <View
          key={index}
          style={[
            styles.waveBar,
            {
              height: size * ratio,
              backgroundColor: tone,
            },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  droplet: {
    borderBottomLeftRadius: 0,
    transform: [{ rotate: '180deg' }],
  },
  leaf: {
    borderTopRightRadius: 0,
    borderBottomLeftRadius: 0,
    transform: [{ rotate: '45deg' }],
  },
  stem: {},
  bulb: {},
  waveRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
  },
  waveBar: {
    width: 4,
    borderRadius: 2,
  },
});
