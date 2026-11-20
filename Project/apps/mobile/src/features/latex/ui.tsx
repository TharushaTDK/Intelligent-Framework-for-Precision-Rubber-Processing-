import { router } from 'expo-router';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  PanResponder,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors, radii, shadows, spacing } from '../../constants/theme';
import type { StageKey } from '../../types/latex';

// one colour per output so water, acid and time are easy to tell apart
export const tints = {
  water: { strong: '#2F7FBF', soft: '#E3F0FA' },
  acid: { strong: '#B9772F', soft: '#FBF0E3' },
  time: { strong: colors.primary, soft: '#E4F2E8' },
  warn: { strong: '#9A6A00', soft: '#FFF4DA' },
  danger: { strong: colors.critical, soft: '#FBE7E4' },
} as const;

// same limits as the backend STAGES, so the live stage matches the saved one
export const STAGES: { key: StageKey; label: string; upTo: number; description: string }[] = [
  { key: 'liquid', label: 'Liquid', upTo: 0.25, description: 'The acid is mixing in, the latex is still liquid. Keep the pan covered and still.' },
  { key: 'thickening', label: 'Thickening', upTo: 0.6, description: 'The latex is starting to thicken. Do not stir or move the pan.' },
  { key: 'setting', label: 'Setting', upTo: 1, description: 'The coagulum is forming and getting firmer. Leave it undisturbed.' },
  { key: 'firm', label: 'Ready to check', upTo: Infinity, description: 'The expected time is reached. Check that the coagulum is firm before sheeting.' },
];

export function stageFor(share: number) {
  return STAGES.find((stage) => share < stage.upTo) ?? STAGES[STAGES.length - 1];
}

// ticks so timers and stages move without reloading
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}

export function formatClock(iso: string | number): string {
  return new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

// 1:05:09 style countdown
export function formatCountdown(ms: number): string {
  const total = Math.max(Math.round(ms / 1000), 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

type ScreenProps = {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  children: ReactNode;
  footer?: ReactNode;
  showBack?: boolean;
};

export function LatexScreen({ title, subtitle, eyebrow, children, footer, showBack = true }: ScreenProps) {
  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.hero}>
          {showBack ? (
            <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.back}>
              <Text style={styles.backLabel}>{'‹  Back'}</Text>
            </Pressable>
          ) : null}
          {eyebrow ? <Text style={styles.eyebrow}>{eyebrow}</Text> : null}
          <Text style={styles.heroTitle}>{title}</Text>
          {subtitle ? <Text style={styles.heroSubtitle}>{subtitle}</Text> : null}
          <View style={styles.heroBubbleBig} />
          <View style={styles.heroBubbleSmall} />
        </View>
        <View style={styles.body}>{children}</View>
      </ScrollView>
      {footer ? <View style={styles.footer}>{footer}</View> : null}
    </SafeAreaView>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function SectionTitle({ children, action }: { children: string; action?: ReactNode }) {
  return (
    <View style={styles.sectionRow}>
      <Text style={styles.sectionTitle}>{children}</Text>
      {action}
    </View>
  );
}

type Tint = keyof typeof tints;

export function StatTile({ label, value, tint, icon }: { label: string; value: string; tint: Tint; icon: string }) {
  const t = tints[tint];
  return (
    <View style={[styles.tile, { backgroundColor: t.soft }]}>
      <View style={[styles.tileIcon, { backgroundColor: t.strong }]}>
        <Text style={styles.tileIconText}>{icon}</Text>
      </View>
      <Text style={[styles.tileValue, { color: t.strong }]}>{value}</Text>
      <Text style={styles.tileLabel}>{label}</Text>
    </View>
  );
}

export function Banner({ tone, title, text }: { tone: 'info' | 'warn' | 'danger'; title?: string; text: string }) {
  const t = tone === 'info' ? tints.time : tone === 'warn' ? tints.warn : tints.danger;
  return (
    <View style={[styles.banner, { backgroundColor: t.soft, borderLeftColor: t.strong }]}>
      {title ? <Text style={[styles.bannerTitle, { color: t.strong }]}>{title}</Text> : null}
      <Text style={styles.bannerText}>{text}</Text>
    </View>
  );
}

export function ProgressBar({ value, color = colors.primary, height = 10 }: { value: number; color?: string; height?: number }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  return (
    <View style={[styles.barTrack, { height, borderRadius: height / 2 }]}>
      <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: color, borderRadius: height / 2 }]} />
    </View>
  );
}

export function Chip({ label, tint }: { label: string; tint: Tint }) {
  const t = tints[tint];
  return (
    <View style={[styles.chip, { backgroundColor: t.soft }]}>
      <Text style={[styles.chipText, { color: t.strong }]}>{label}</Text>
    </View>
  );
}

export function ActionTile({ icon, title, caption, onPress, tint }: {
  icon: string;
  title: string;
  caption: string;
  onPress: () => void;
  tint: Tint;
}) {
  const t = tints[tint];
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.action, pressed && styles.pressed]}
    >
      <View style={[styles.actionIcon, { backgroundColor: t.soft }]}>
        <Text style={styles.actionIconText}>{icon}</Text>
      </View>
      <Text style={styles.actionTitle}>{title}</Text>
      <Text style={styles.actionCaption}>{caption}</Text>
    </Pressable>
  );
}

// drawing of the pan, the latex turns from milky liquid to a firm white slab
export function StagePan({ share }: { share: number }) {
  const s = Math.min(Math.max(share, 0), 1);
  const firm = share >= 1;
  const surface = firm ? '#F4EBD8' : s > 0.6 ? '#F7F1E4' : s > 0.25 ? '#FBF8F0' : '#FFFFFF';
  return (
    <View style={styles.panWrap}>
      <View style={styles.panRim} />
      <View style={styles.pan}>
        <View style={[styles.panSerum, { height: `${12 + s * 22}%`, opacity: 0.25 + s * 0.6 }]} />
        <View
          style={[
            styles.panLatex,
            {
              backgroundColor: surface,
              height: `${72 - s * 18}%`,
              borderTopLeftRadius: 8 + s * 22,
              borderTopRightRadius: 8 + s * 22,
            },
          ]}
        >
          {!firm && s < 0.25 ? (
            <View style={styles.ripples}>
              <View style={styles.ripple} />
              <View style={[styles.ripple, { width: 26 }]} />
            </View>
          ) : null}
        </View>
      </View>
    </View>
  );
}

export function StageTrack({ share }: { share: number }) {
  const current = stageFor(share);
  const currentIndex = STAGES.indexOf(current);
  return (
    <View style={styles.track}>
      {STAGES.map((stage, index) => {
        const done = index < currentIndex;
        const active = index === currentIndex;
        return (
          <View key={stage.key} style={styles.trackItem}>
            <View style={styles.trackDotRow}>
              {index > 0 ? <View style={[styles.trackLine, (done || active) && styles.trackLineOn]} /> : <View style={styles.trackLineGap} />}
              <View style={[styles.trackDot, done && styles.trackDotDone, active && styles.trackDotActive]}>
                <Text style={[styles.trackDotText, (done || active) && styles.trackDotTextOn]}>{done ? '✓' : index + 1}</Text>
              </View>
              {index < STAGES.length - 1 ? <View style={[styles.trackLine, done && styles.trackLineOn]} /> : <View style={styles.trackLineGap} />}
            </View>
            <Text style={[styles.trackLabel, active && styles.trackLabelActive]} numberOfLines={2}>
              {stage.label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

type SliderProps = {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format: (value: number) => string;
  color?: string;
};

// plain view slider, no extra package needed
export function ValueSlider({ value, min, max, step, onChange, format, color = colors.primary }: SliderProps) {
  const width = useRef(1);
  const startX = useRef(0);
  const latest = useRef(onChange);
  latest.current = onChange;

  const snap = (x: number) => {
    const share = Math.min(Math.max(x / width.current, 0), 1);
    const raw = min + share * (max - min);
    return Math.round(Math.round(raw / step) * step * 100) / 100;
  };

  const responder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (event) => {
        startX.current = event.nativeEvent.locationX;
        latest.current(snap(startX.current));
      },
      onPanResponderMove: (_, gesture) => latest.current(snap(startX.current + gesture.dx)),
    }),
  ).current;

  const share = (value - min) / (max - min);
  const nudge = (dir: number) => onChange(Math.min(Math.max(Math.round((value + dir * step) * 100) / 100, min), max));

  return (
    <View style={styles.slider}>
      <Pressable accessibilityLabel="Less" onPress={() => nudge(-1)} style={styles.sliderButton}>
        <Text style={[styles.sliderButtonText, { color }]}>−</Text>
      </Pressable>
      <View style={styles.sliderMiddle}>
        <View
          style={styles.sliderHit}
          onLayout={(event) => {
            width.current = event.nativeEvent.layout.width || 1;
          }}
          {...responder.panHandlers}
        >
          <View pointerEvents="none" style={styles.sliderRail}>
            <View style={[styles.sliderFill, { width: `${share * 100}%`, backgroundColor: color }]} />
          </View>
          <View pointerEvents="none" style={[styles.sliderThumb, { left: `${share * 100}%`, borderColor: color }]} />
        </View>
        <View style={styles.sliderEnds}>
          <Text style={styles.sliderEnd}>{format(min)}</Text>
          <Text style={styles.sliderEnd}>{format(max)}</Text>
        </View>
      </View>
      <Pressable accessibilityLabel="More" onPress={() => nudge(1)} style={styles.sliderButton}>
        <Text style={[styles.sliderButtonText, { color }]}>+</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  scroll: { paddingBottom: spacing.xxl },
  hero: {
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xl + spacing.md,
    borderBottomLeftRadius: radii.xl,
    borderBottomRightRadius: radii.xl,
    overflow: 'hidden',
  },
  back: { alignSelf: 'flex-start', paddingVertical: spacing.sm, marginBottom: spacing.xs },
  backLabel: { color: colors.primaryLight, fontSize: 15, fontWeight: '700' },
  eyebrow: { color: colors.primaryLight, fontSize: 12, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase' },
  heroTitle: { marginTop: 4, color: colors.textOnPrimary, fontSize: 28, fontWeight: '800' },
  heroSubtitle: { marginTop: spacing.xs, color: '#D7EADD', fontSize: 14, lineHeight: 20, maxWidth: '92%' },
  heroBubbleBig: {
    position: 'absolute', right: -50, top: -40, width: 170, height: 170, borderRadius: 85,
    backgroundColor: 'rgba(255,255,255,0.07)',
  },
  heroBubbleSmall: {
    position: 'absolute', right: 60, bottom: -30, width: 80, height: 80, borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  body: { marginTop: -spacing.lg, paddingHorizontal: spacing.md, gap: spacing.md },
  footer: {
    paddingHorizontal: spacing.md, paddingTop: spacing.sm, paddingBottom: spacing.md, gap: spacing.sm,
    backgroundColor: colors.background, borderTopWidth: 1, borderTopColor: colors.border,
  },
  card: {
    backgroundColor: colors.surface, borderRadius: radii.lg, padding: spacing.md, gap: spacing.sm,
    borderWidth: 1, borderColor: colors.border, ...shadows.soft,
  },
  sectionRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: spacing.xs },
  sectionTitle: { fontSize: 17, fontWeight: '800', color: colors.textPrimary },
  tile: { flex: 1, borderRadius: radii.md, padding: spacing.md, gap: 4, minWidth: 96 },
  tileIcon: { width: 34, height: 34, borderRadius: 17, alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
  tileIconText: { fontSize: 16 },
  tileValue: { fontSize: 22, fontWeight: '800' },
  tileLabel: { fontSize: 12, color: colors.textSecondary, fontWeight: '600' },
  banner: { borderRadius: radii.md, borderLeftWidth: 5, padding: spacing.md, gap: 2 },
  bannerTitle: { fontSize: 14, fontWeight: '800' },
  bannerText: { fontSize: 13, color: colors.textPrimary, lineHeight: 19 },
  barTrack: { backgroundColor: '#E8EFEA', overflow: 'hidden' },
  barFill: { height: '100%' },
  chip: { alignSelf: 'flex-start', borderRadius: radii.pill, paddingHorizontal: 10, paddingVertical: 3 },
  chipText: { fontSize: 12, fontWeight: '800' },
  action: {
    flexBasis: '47%', flexGrow: 1, backgroundColor: colors.surface, borderRadius: radii.lg, padding: spacing.md,
    borderWidth: 1, borderColor: colors.border, gap: 4, ...shadows.soft,
  },
  pressed: { opacity: 0.85, transform: [{ scale: 0.98 }] },
  actionIcon: { width: 44, height: 44, borderRadius: 14, alignItems: 'center', justifyContent: 'center', marginBottom: 6 },
  actionIconText: { fontSize: 22 },
  actionTitle: { fontSize: 15, fontWeight: '800', color: colors.textPrimary },
  actionCaption: { fontSize: 12, color: colors.textSecondary, lineHeight: 16 },
  panWrap: { alignItems: 'center', paddingVertical: spacing.sm },
  panRim: { width: 236, height: 14, borderRadius: 7, backgroundColor: '#9AA7A0' },
  pan: {
    width: 220, height: 120, marginTop: -4, backgroundColor: '#C9D2CD', borderBottomLeftRadius: 28,
    borderBottomRightRadius: 28, overflow: 'hidden', justifyContent: 'flex-end',
  },
  panSerum: { position: 'absolute', left: 0, right: 0, bottom: 0, backgroundColor: '#E9E2B8' },
  panLatex: { marginHorizontal: 10, marginBottom: 8, alignItems: 'center', paddingTop: 14 },
  ripples: { alignItems: 'center', gap: 6 },
  ripple: { width: 60, height: 3, borderRadius: 2, backgroundColor: '#E9EEF0' },
  track: { flexDirection: 'row' },
  trackItem: { flex: 1, alignItems: 'center', gap: 6 },
  trackDotRow: { flexDirection: 'row', alignItems: 'center', width: '100%' },
  trackLine: { flex: 1, height: 3, backgroundColor: colors.border },
  trackLineOn: { backgroundColor: colors.primary },
  trackLineGap: { flex: 1 },
  trackDot: {
    width: 30, height: 30, borderRadius: 15, borderWidth: 2, borderColor: colors.border,
    backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center',
  },
  trackDotDone: { backgroundColor: colors.primary, borderColor: colors.primary },
  trackDotActive: { borderColor: colors.primary, backgroundColor: '#E4F2E8', transform: [{ scale: 1.12 }] },
  trackDotText: { fontSize: 12, fontWeight: '800', color: colors.textSecondary },
  trackDotTextOn: { color: colors.primaryDark },
  trackLabel: { fontSize: 11, color: colors.textSecondary, textAlign: 'center', fontWeight: '600' },
  trackLabelActive: { color: colors.primary, fontWeight: '800' },
  slider: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  sliderButton: {
    width: 40, height: 40, borderRadius: 20, backgroundColor: colors.background, alignItems: 'center',
    justifyContent: 'center', borderWidth: 1, borderColor: colors.border,
  },
  sliderButtonText: { fontSize: 22, fontWeight: '800', marginTop: -2 },
  sliderMiddle: { flex: 1 },
  sliderHit: { height: 36, justifyContent: 'center' },
  sliderRail: { height: 8, borderRadius: 4, backgroundColor: '#E8EFEA', overflow: 'hidden' },
  sliderFill: { height: '100%' },
  sliderThumb: {
    position: 'absolute', width: 26, height: 26, borderRadius: 13, marginLeft: -13, backgroundColor: colors.surface,
    borderWidth: 4, ...shadows.soft,
  },
  sliderEnds: { flexDirection: 'row', justifyContent: 'space-between' },
  sliderEnd: { fontSize: 11, color: colors.textSecondary },
});
