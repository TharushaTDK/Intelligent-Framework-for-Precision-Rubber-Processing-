import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { StatusDot } from '../../components/UiIcons';
import { colors, radii, spacing } from '../../constants/theme';
import { COMPONENTS, type ComponentInfo } from './componentCatalog';
import type { Insight, Insights } from './useLatestInsights';

type Props = {
  insights: Insights;
};

export function LatestPredictions({ insights }: Props) {
  return (
    <View style={styles.grid}>
      {COMPONENTS.map((info) => (
        <InsightTile key={info.key} info={info} insight={insights[info.key]} />
      ))}
    </View>
  );
}

// read-only summary card: it reminds the user of their last result and is not tappable
function InsightTile({ info, insight }: { info: ComponentInfo; insight: Insight }) {
  return (
    <View accessibilityLabel={`${info.shortTitle} latest result`} style={styles.tile}>
      <View style={styles.tileHeader}>
        <View style={styles.iconBox}>{info.icon(16)}</View>
        <Text style={styles.tileName} numberOfLines={1}>
          {info.shortTitle}
        </Text>
      </View>

      <InsightBody info={info} insight={insight} />
    </View>
  );
}

function InsightBody({ info, insight }: { info: ComponentInfo; insight: Insight }) {
  switch (insight.status) {
    case 'loading':
      return (
        <View style={styles.loading}>
          <ActivityIndicator size="small" color={colors.primaryLight} />
        </View>
      );

    case 'offline':
      return (
        <View style={styles.body}>
          <View style={styles.valueRow}>
            <StatusDot tone={colors.textSecondary} />
            <Text style={[styles.value, styles.valueMuted]}>Offline</Text>
          </View>
          <Text style={styles.detail} numberOfLines={2}>
            Can’t reach the service right now. Pull down to refresh.
          </Text>
        </View>
      );

    case 'empty':
      return (
        <View style={styles.body}>
          <Text style={[styles.value, styles.valueReady]}>Get ready</Text>
          <Text style={styles.detail} numberOfLines={3}>
            No results yet. Your latest {info.shortTitle.toLowerCase()} result will be shown here.
          </Text>
        </View>
      );

    case 'ready':
      return (
        <View style={styles.body}>
          <View style={styles.valueRow}>
            <StatusDot tone={insight.tone} />
            <Text style={styles.value} numberOfLines={1}>
              {insight.value}
            </Text>
          </View>
          <Text style={styles.caption} numberOfLines={1}>
            {insight.caption}
          </Text>
          <Text style={styles.detail} numberOfLines={2}>
            {insight.detail}
          </Text>
          <View style={styles.footer}>
            <Text style={styles.when} numberOfLines={1}>
              {insight.when ?? 'Live status'}
            </Text>
          </View>
        </View>
      );
  }
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    rowGap: spacing.sm + 4,
  },
  tile: {
    width: '48%',
    minHeight: 150,
    backgroundColor: colors.surface,
    borderRadius: radii.md + 2,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.sm + 4,
  },
  tileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  iconBox: {
    width: 28,
    height: 28,
    borderRadius: 9,
    backgroundColor: '#EAF6EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  tileName: {
    flex: 1,
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  body: {
    flex: 1,
    marginTop: spacing.sm + 2,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  value: {
    flexShrink: 1,
    fontSize: 18,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  valueMuted: {
    color: colors.textSecondary,
  },
  valueReady: {
    color: colors.primary,
  },
  caption: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textPrimary,
    marginTop: 1,
  },
  detail: {
    marginTop: 4,
    fontSize: 11.5,
    lineHeight: 16,
    color: colors.textSecondary,
  },
  footer: {
    marginTop: 'auto',
    paddingTop: spacing.xs,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  when: {
    flex: 1,
    fontSize: 11,
    color: colors.success,
    fontWeight: '600',
  },
});
