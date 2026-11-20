import { router } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
} from 'react-native';

import { SunCloudIcon } from '../../components/UiIcons';
import { colors, radii, shadows, spacing } from '../../constants/theme';
import type { AdvisoryRecordResponse } from '../../types/plantAdvisory';
import { COMPONENTS, type ComponentInfo } from './componentCatalog';
import { formatRelativeTime } from './useLatestInsights';

const SLIDE_HEIGHT = 268;
const AUTO_PLAY_MS = 6000;
const RESUME_AFTER_TOUCH_MS = 10000;

type Props = {
  forecast: AdvisoryRecordResponse | null | undefined;
  autoPlay: boolean;
};

export function ForecastSlideshow({ forecast, autoPlay }: Props) {
  const scrollRef = useRef<ScrollView>(null);
  const [width, setWidth] = useState(0);
  const [index, setIndex] = useState(0);
  const pausedUntil = useRef(0);
  const slideCount = COMPONENTS.length + 1;

  const onLayout = (event: LayoutChangeEvent) => setWidth(event.nativeEvent.layout.width);

  const goTo = useCallback(
    (next: number) => {
      if (!width) return;
      scrollRef.current?.scrollTo({ x: next * width, animated: true });
      setIndex(next);
    },
    [width],
  );

  useEffect(() => {
    if (!autoPlay || !width) return undefined;
    const timer = setInterval(() => {
      if (Date.now() < pausedUntil.current) return;
      setIndex((current) => {
        const next = (current + 1) % slideCount;
        scrollRef.current?.scrollTo({ x: next * width, animated: true });
        return next;
      });
    }, AUTO_PLAY_MS);
    return () => clearInterval(timer);
  }, [autoPlay, width, slideCount]);

  const onMomentumEnd = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
    if (!width) return;
    setIndex(Math.round(event.nativeEvent.contentOffset.x / width));
  };

  return (
    <View style={styles.card} onLayout={onLayout}>
      {width > 0 ? (
        <ScrollView
          ref={scrollRef}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          onScrollBeginDrag={() => {
            pausedUntil.current = Date.now() + RESUME_AFTER_TOUCH_MS;
          }}
          onMomentumScrollEnd={onMomentumEnd}
        >
          <View style={{ width, height: SLIDE_HEIGHT }}>
            <ForecastSlide forecast={forecast} />
          </View>
          {COMPONENTS.map((info) => (
            <View key={info.key} style={{ width, height: SLIDE_HEIGHT }}>
              <ComponentSlide info={info} />
            </View>
          ))}
        </ScrollView>
      ) : (
        <View style={{ height: SLIDE_HEIGHT }} />
      )}

      <View style={styles.dots}>
        {Array.from({ length: slideCount }).map((_, dot) => (
          <Pressable
            key={dot}
            accessibilityRole="button"
            accessibilityLabel={`Show slide ${dot + 1}`}
            hitSlop={8}
            onPress={() => {
              pausedUntil.current = Date.now() + RESUME_AFTER_TOUCH_MS;
              goTo(dot);
            }}
          >
            <View style={[styles.dot, dot === index && styles.dotActive]} />
          </Pressable>
        ))}
      </View>
    </View>
  );
}

// ------------------------------------------------------------
// Forecast slide (from the most recent saved plant advisory)
// ------------------------------------------------------------

function describeConditions(record: AdvisoryRecordResponse): string {
  let sky = 'Partly cloudy';
  if (record.forecast_rain_next_6h_mm >= 5) sky = 'Rain expected';
  else if (record.forecast_rain_next_6h_mm >= 0.5) sky = 'Light showers likely';
  else if (record.relative_humidity_pct >= 85) sky = 'Humid & overcast';
  else if (record.temperature_c >= 31) sky = 'Hot & sunny';

  const tapping =
    record.tapping_label === 'High'
      ? 'good for tapping'
      : record.tapping_label === 'Moderate'
        ? 'fair for tapping'
        : record.tapping_label === 'Low'
          ? 'not ideal for tapping'
          : null;

  return tapping ? `${sky} · ${tapping}` : sky;
}

function isToday(iso: string): boolean {
  const date = new Date(iso);
  const now = new Date();
  return (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  );
}

function ForecastSlide({ forecast }: { forecast: AdvisoryRecordResponse | null | undefined }) {
  if (forecast === undefined) {
    return (
      <View style={[styles.slide, styles.center]}>
        <ActivityIndicator color={colors.primary} />
        <Text style={styles.muted}>Loading your field weather…</Text>
      </View>
    );
  }

  if (forecast === null) {
    return (
      <View style={styles.slide}>
        <Text style={styles.eyebrow}>TODAY&apos;S FORECAST</Text>
        <View style={styles.forecastTop}>
          <View style={styles.flex}>
            <Text style={styles.emptyTitle}>Get ready for today</Text>
            <Text style={styles.emptyBody}>
              Run a plant advisory for your field and your local weather, tapping score and DRC
              estimate will appear right here.
            </Text>
          </View>
          <View style={styles.weatherBadge}>
            <SunCloudIcon size={52} />
          </View>
        </View>
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push('/plant-advisory')}
          style={({ pressed }) => [styles.cta, pressed && styles.ctaPressed]}
        >
          <Text style={styles.ctaLabel}>Get today’s advisory</Text>
        </Pressable>
      </View>
    );
  }

  const fresh = isToday(forecast.observation_timestamp);
  const when = formatRelativeTime(forecast.created_at);

  return (
    <Pressable style={styles.slide} onPress={() => router.push('/plant-advisory')}>
      <Text style={styles.eyebrow}>{fresh ? "TODAY'S FORECAST" : 'LATEST FIELD WEATHER'}</Text>
      <View style={styles.forecastTop}>
        <View style={styles.flex}>
          <Text style={styles.temperature}>{Math.round(forecast.temperature_c)}°C</Text>
          <Text style={styles.condition} numberOfLines={2}>
            {describeConditions(forecast)}
          </Text>
        </View>
        <View style={styles.weatherBadge}>
          <SunCloudIcon size={52} />
        </View>
      </View>

      <View style={styles.divider} />

      <View style={styles.statsRow}>
        <Stat value={`${Math.round(forecast.relative_humidity_pct)}%`} label="Humidity" />
        <View style={styles.statDivider} />
        <Stat value={`${forecast.forecast_rain_next_6h_mm.toFixed(1)} mm`} label="Rain · next 6h" />
        <View style={styles.statDivider} />
        <Stat value={`${Math.round(forecast.wind_speed_kmh)} km/h`} label="Wind" />
      </View>

      <Text style={styles.footnote} numberOfLines={1}>
        {fresh ? 'From your latest advisory' : 'From your last advisory'}
        {when ? ` · ${when}` : ''} · tap for a new advisory
      </Text>
    </Pressable>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

// ------------------------------------------------------------
// Component introduction slides (illustration + description)
// ------------------------------------------------------------

function SlideArt({ info }: { info: ComponentInfo }) {
  return (
    <View style={[styles.art, { backgroundColor: info.palette.to }]}>
      <View style={[styles.artBlob, styles.artBlobA, { backgroundColor: info.palette.from }]} />
      <View style={[styles.artBlob, styles.artBlobB, { backgroundColor: info.palette.from }]} />
      <View style={[styles.artRing, { borderColor: info.palette.ink }]} />
      <View style={styles.artHill} />
      <View style={styles.artIconWrap}>{info.icon(34, info.palette.to)}</View>
      <View style={styles.artText}>
        <Text style={[styles.artNumber, { color: info.palette.ink }]}>COMPONENT {info.number}</Text>
        <Text style={styles.artTitle} numberOfLines={2}>
          {info.title}
        </Text>
      </View>
    </View>
  );
}

function ComponentSlide({ info }: { info: ComponentInfo }) {
  return (
    <View style={styles.componentSlide}>
      <SlideArt info={info} />
      <View style={styles.componentBody}>
        <Text style={styles.componentDescription} numberOfLines={3}>
          {info.description}
        </Text>
        <View style={styles.chipsRow}>
          {info.highlights.map((item) => (
            <View key={item} style={styles.chip}>
              <Text style={styles.chipLabel}>{item}</Text>
            </View>
          ))}
        </View>
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push(info.route)}
          style={({ pressed }) => [styles.openLink, pressed && styles.openLinkPressed]}
        >
          <Text style={styles.openLabel}>Open {info.shortTitle} ›</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.soft,
  },
  slide: {
    flex: 1,
    padding: spacing.md + 2,
  },
  center: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  flex: {
    flex: 1,
  },
  muted: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  eyebrow: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.2,
    color: colors.success,
  },
  forecastTop: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginTop: spacing.xs,
  },
  temperature: {
    fontSize: 40,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  condition: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: 2,
  },
  weatherBadge: {
    width: 76,
    height: 76,
    borderRadius: radii.md + 4,
    backgroundColor: '#EAF6EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  divider: {
    height: 1,
    backgroundColor: colors.border,
    marginVertical: spacing.md,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  stat: {
    flex: 1,
    alignItems: 'center',
    gap: 2,
  },
  statDivider: {
    width: 1,
    height: 32,
    backgroundColor: colors.border,
  },
  statValue: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  statLabel: {
    fontSize: 11,
    color: colors.textSecondary,
  },
  footnote: {
    marginTop: spacing.md,
    fontSize: 11,
    color: colors.textSecondary,
    textAlign: 'center',
  },
  emptyTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  emptyBody: {
    marginTop: spacing.xs,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  cta: {
    marginTop: 'auto',
    minHeight: 46,
    borderRadius: radii.pill,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ctaPressed: {
    backgroundColor: colors.primaryDark,
  },
  ctaLabel: {
    color: colors.textOnPrimary,
    fontSize: 15,
    fontWeight: '700',
  },
  componentSlide: {
    flex: 1,
  },
  art: {
    height: 128,
    overflow: 'hidden',
    justifyContent: 'flex-end',
    padding: spacing.md,
  },
  artBlob: {
    position: 'absolute',
    borderRadius: 999,
    opacity: 0.55,
  },
  artBlobA: {
    width: 180,
    height: 180,
    top: -90,
    right: -40,
  },
  artBlobB: {
    width: 120,
    height: 120,
    bottom: -70,
    left: -30,
  },
  artRing: {
    position: 'absolute',
    width: 92,
    height: 92,
    borderRadius: 46,
    borderWidth: 1.5,
    top: 16,
    right: 16,
    opacity: 0.35,
  },
  artHill: {
    position: 'absolute',
    left: -40,
    right: -40,
    bottom: -150,
    height: 190,
    borderRadius: 400,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  artIconWrap: {
    position: 'absolute',
    top: 30,
    right: 34,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  artText: {
    paddingRight: 96,
  },
  artNumber: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.4,
  },
  artTitle: {
    marginTop: 2,
    fontSize: 18,
    fontWeight: '800',
    color: colors.textOnPrimary,
  },
  componentBody: {
    flex: 1,
    padding: spacing.md,
    paddingBottom: spacing.sm,
    gap: spacing.sm,
  },
  componentDescription: {
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  chipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  chip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radii.pill,
    backgroundColor: '#EAF6EE',
  },
  chipLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.primary,
  },
  openLink: {
    marginTop: 'auto',
    alignSelf: 'flex-start',
    paddingVertical: 4,
  },
  openLinkPressed: {
    opacity: 0.6,
  },
  openLabel: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.primary,
  },
  dots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
    paddingBottom: spacing.sm + 2,
    paddingTop: 2,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.border,
  },
  dotActive: {
    width: 18,
    backgroundColor: colors.primary,
  },
});
