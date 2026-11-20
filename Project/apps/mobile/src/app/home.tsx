import { router, useFocusEffect } from 'expo-router';
import { setStatusBarStyle } from 'expo-status-bar';
import { useCallback, useRef } from 'react';
import {
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { BrandMark } from '../components/BrandMark';
import { GearIcon, PinIcon } from '../components/UiIcons';
import { colors, radii, spacing } from '../constants/theme';
import { ForecastSlideshow } from '../features/home/ForecastSlideshow';
import { LatestPredictions } from '../features/home/LatestPredictions';
import { ToolGrid } from '../features/home/ToolGrid';
import { useLatestInsights } from '../features/home/useLatestInsights';
import { useProfile } from '../features/profile/profileStorage';

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

export default function Home() {
  const insets = useSafeAreaInsets();
  const { profile, preferences, reload } = useProfile();
  const { insights, forecast, updatedAt, refreshing, refresh } = useLatestInsights();

  // white clock while the green header is behind the status bar,
  // dark clock once the page scrolls onto the light background
  const statusBarStyle = useRef<'light' | 'dark'>('light');
  const applyStatusBar = useCallback((style: 'light' | 'dark') => {
    statusBarStyle.current = style;
    setStatusBarStyle(style);
  }, []);
  const onScroll = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const next = event.nativeEvent.contentOffset.y > 180 ? 'dark' : 'light';
    if (next !== statusBarStyle.current) {
      applyStatusBar(next);
    }
  };

  // refresh whenever the user comes back to Home (e.g. after a new prediction)
  useFocusEffect(
    useCallback(() => {
      setStatusBarStyle(statusBarStyle.current);
      void reload();
      void refresh();
      return () => setStatusBarStyle('dark');
    }, [reload, refresh]),
  );

  const firstName = profile?.fullName.trim().split(/\s+/)[0];

  return (
    <View style={styles.root}>
      <ScrollView
        contentContainerStyle={{ paddingBottom: insets.bottom + spacing.xl }}
        showsVerticalScrollIndicator={false}
        onScroll={onScroll}
        scrollEventThrottle={64}
        refreshControl={
          <RefreshControl
            refreshing={refreshing && updatedAt !== null}
            onRefresh={() => void refresh()}
            colors={[colors.primary]}
            tintColor={colors.textOnPrimary}
            progressViewOffset={insets.top}
          />
        }
      >
        <View style={[styles.header, { paddingTop: insets.top + spacing.md }]}>
          <View style={[styles.headerBlob, styles.headerBlobA]} />
          <View style={[styles.headerBlob, styles.headerBlobB]} />

          <View style={styles.headerRow}>
            <View style={styles.logoTile}>
              <BrandMark size={46} />
            </View>

            <View style={styles.headerText}>
              <Text style={styles.greeting} numberOfLines={1}>
                {greeting()}
                {firstName ? `, ${firstName}` : ''} 👋
              </Text>
              <Text style={styles.estate} numberOfLines={1}>
                {profile?.estateName ?? 'RubberSense'}
              </Text>
              <View style={styles.locationRow}>
                <PinIcon size={11} tone="#CDE8D6" />
                <Text style={styles.location} numberOfLines={1}>
                  {profile ? `${profile.district}, Sri Lanka` : 'Precision rubber intelligence'}
                </Text>
              </View>
            </View>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Open settings"
              hitSlop={8}
              onPress={() => router.push('/settings')}
              style={({ pressed }) => [styles.settingsButton, pressed && styles.settingsPressed]}
            >
              <GearIcon size={20} tone={colors.textOnPrimary} />
            </Pressable>
          </View>
        </View>

        <View style={styles.content}>
          <View style={styles.slideshowWrap}>
            <ForecastSlideshow forecast={forecast} autoPlay={preferences.autoPlaySlides} />
          </View>

          {preferences.showLatestPredictions ? (
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <View>
                  <Text style={styles.sectionTitle}>Latest Predictions</Text>
                  <Text style={styles.sectionCaption}>Your most recent result from each tool</Text>
                </View>
                {updatedAt ? (
                  <Pressable hitSlop={8} onPress={() => void refresh()}>
                    <Text style={styles.updated}>
                      {refreshing
                        ? 'Updating…'
                        : `Updated ${updatedAt.toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}`}
                    </Text>
                  </Pressable>
                ) : null}
              </View>
              <LatestPredictions insights={insights} />
            </View>
          ) : null}

          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <View>
                <Text style={styles.sectionTitle}>Explore Tools</Text>
                <Text style={styles.sectionCaption}>Precision support from field to processing</Text>
              </View>
            </View>
            <ToolGrid />
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg,
    paddingBottom: 96,
    borderBottomLeftRadius: radii.xl,
    borderBottomRightRadius: radii.xl,
    overflow: 'hidden',
  },
  headerBlob: {
    position: 'absolute',
    borderRadius: 999,
    backgroundColor: '#2E8B57',
  },
  headerBlobA: {
    width: 240,
    height: 240,
    top: -120,
    right: -70,
    opacity: 0.55,
  },
  headerBlobB: {
    width: 160,
    height: 160,
    bottom: -90,
    left: -50,
    opacity: 0.35,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm + 4,
  },
  logoTile: {
    width: 54,
    height: 54,
    borderRadius: 16,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerText: {
    flex: 1,
  },
  greeting: {
    fontSize: 13,
    color: '#CDE8D6',
    fontWeight: '600',
  },
  estate: {
    marginTop: 1,
    fontSize: 21,
    fontWeight: '800',
    color: colors.textOnPrimary,
  },
  locationRow: {
    marginTop: 3,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  location: {
    flexShrink: 1,
    fontSize: 12,
    color: '#CDE8D6',
  },
  settingsButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: 'rgba(255,255,255,0.16)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingsPressed: {
    backgroundColor: 'rgba(255,255,255,0.28)',
  },
  content: {
    paddingHorizontal: spacing.lg - 4,
  },
  slideshowWrap: {
    marginTop: -80,
  },
  section: {
    marginTop: spacing.lg,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginBottom: spacing.sm + 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  sectionCaption: {
    marginTop: 1,
    fontSize: 12,
    color: colors.textSecondary,
  },
  updated: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.success,
  },
});
