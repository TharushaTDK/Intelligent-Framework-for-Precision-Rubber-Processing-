import { useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { colors, spacing } from '../../constants/theme';
import { coagulationTimeText, formatAcid, formatDrc, formatWater } from '../../features/latex/BatchSummaryCard';
import { Banner, Card, LatexScreen, StatTile, ValueSlider, tints } from '../../features/latex/ui';
import { listRecommendations, previewRecommendation, toReadableMessage } from '../../services/latexApi';
import type { CoagulationPreviewResponse } from '../../types/latex';

// latex volume range of the Component 2 training data
const VOLUME = { min: 1.0, max: 2.0, step: 0.1 };

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

function change(now: number, base: number, unit: (v: number) => string): string {
  const diff = now - base;
  if (Math.abs(diff) < 0.0001) return 'same as your batch';
  return `${diff > 0 ? '▲' : '▼'} ${unit(Math.abs(diff))} vs your batch`;
}

export default function WhatIf() {
  const params = useLocalSearchParams<{ drc?: string; volume?: string }>();

  // DRC stays fixed at the location's value, only the volume moves
  const [drc, setDrc] = useState<number | null>(params.drc ? Number(params.drc) : null);
  const [volume, setVolume] = useState(params.volume ? clamp(Number(params.volume), VOLUME.min, VOLUME.max) : 1.5);
  const [result, setResult] = useState<CoagulationPreviewResponse | null>(null);
  const [base, setBase] = useState<CoagulationPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const version = useRef(0);

  // opened from the home page, start from the latest batch's DRC
  useEffect(() => {
    if (drc !== null) return;
    listRecommendations(1)
      .then((res) => setDrc(res.items[0]?.drc ?? 43.9))
      .catch(() => setDrc(43.9));
  }, [drc]);

  // wait until the slider stops before asking the models
  useEffect(() => {
    if (drc === null) return;
    const mine = ++version.current;
    setLoading(true);
    const timer = setTimeout(() => {
      previewRecommendation({ drc, latex_volume_L: volume, outputs: ['water_acid', 'coagulation_time'] })
        .then((res) => {
          if (mine !== version.current) return;
          setResult(res);
          setBase((prev) => prev ?? res);
          setError(null);
        })
        .catch((err) => mine === version.current && setError(toReadableMessage(err)))
        .finally(() => mine === version.current && setLoading(false));
    }, 350);
    return () => clearTimeout(timer);
  }, [drc, volume]);

  const water = result?.water_acid?.water_amount_L;
  const acid = result?.water_acid?.prepared_1pct_formic_acid_solution_mL;
  const time = result?.coagulation_time_hours;

  return (
    <LatexScreen
      eyebrow="What-if"
      title="Try Different Batches"
      subtitle="Move the latex volume and see how the water, acid and time change for this location. Nothing is saved."
    >
      <Card>
        <View style={styles.sliderHead}>
          <Text style={styles.sliderTitle}>Latex volume</Text>
          <Text style={[styles.sliderValue, { color: tints.water.strong }]}>{volume.toFixed(1)} L</Text>
        </View>
        <ValueSlider
          value={volume}
          {...VOLUME}
          onChange={setVolume}
          format={(v) => `${v.toFixed(1)} L`}
          color={tints.water.strong}
        />

        <View style={[styles.sliderHead, styles.drcRow]}>
          <Text style={styles.sliderTitle}>Dry rubber content</Text>
          <Text style={[styles.sliderValue, { color: tints.acid.strong }]}>{drc !== null ? formatDrc(drc) : '—'}</Text>
        </View>
        <Text style={styles.drcCaption}>From the chosen location, it stays the same.</Text>
      </Card>

      <View style={styles.resultHead}>
        <Text style={styles.resultTitle}>Model result</Text>
        {loading ? <ActivityIndicator size="small" color={colors.primary} /> : null}
      </View>

      {result ? (
        <>
          <View style={styles.tiles}>
            <StatTile icon="💧" tint="water" label="Water" value={water != null ? formatWater(water) : '—'} />
            <StatTile icon="🧪" tint="acid" label="1% acid" value={acid != null ? formatAcid(acid) : '—'} />
          </View>
          <View style={styles.timeCard}>
            <Text style={styles.timeIcon}>⏱</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.timeLabel}>Coagulation time</Text>
              <Text style={styles.timeValue}>{time != null ? coagulationTimeText(time) : '—'}</Text>
            </View>
          </View>

          {base && water != null && acid != null && time != null ? (
            <Card>
              <Text style={styles.compareTitle}>Compared with where you started</Text>
              <Text style={styles.compareRow}>
                💧 Water {change(water, base.water_acid?.water_amount_L ?? water, formatWater)}
              </Text>
              <Text style={styles.compareRow}>
                🧪 Acid {change(acid, base.water_acid?.prepared_1pct_formic_acid_solution_mL ?? acid, formatAcid)}
              </Text>
              <Text style={styles.compareRow}>
                ⏱ Time {change(time, base.coagulation_time_hours ?? time, (v) => coagulationTimeText(v))}
              </Text>
            </Card>
          ) : null}
        </>
      ) : null}

      <Banner
        tone="info"
        title="Why it changes"
        text="More latex needs more water and acid, and takes longer to coagulate."
      />

      {error ? <Banner tone="danger" text={error} /> : null}
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  sliderHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
  },
  sliderTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  sliderValue: {
    fontSize: 22,
    fontWeight: '800',
  },
  drcRow: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  drcCaption: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  resultHead: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  resultTitle: {
    fontSize: 17,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  tiles: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  timeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.primary,
    borderRadius: 24,
    padding: spacing.md,
  },
  timeIcon: {
    fontSize: 30,
  },
  timeLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.primaryLight,
  },
  timeValue: {
    marginTop: 2,
    fontSize: 24,
    fontWeight: '800',
    color: colors.textOnPrimary,
  },
  compareTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  compareRow: {
    fontSize: 13,
    color: colors.textSecondary,
  },
});
