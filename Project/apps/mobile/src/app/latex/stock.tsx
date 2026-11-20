import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import { formatAcid } from '../../features/latex/BatchSummaryCard';
import { Banner, Card, LatexScreen, tints } from '../../features/latex/ui';
import { addAcidStock, getAcidStock, setAcidStock, toReadableMessage } from '../../services/latexApi';
import type { AcidStockResponse } from '../../types/latex';

const STOCK_MAX = 100000;

export default function AcidStock() {
  const [stock, setStock] = useState<AcidStockResponse | null>(null);
  const [added, setAdded] = useState('');
  const [total, setTotal] = useState('');
  const [showCorrect, setShowCorrect] = useState(false);
  const [fieldError, setFieldError] = useState<{ added?: string; total?: string }>({});
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useFocusEffect(
    useCallback(() => {
      getAcidStock()
        .then(setStock)
        .catch((err) => setError(toReadableMessage(err)));
    }, []),
  );

  async function save(kind: 'added' | 'total') {
    const text = kind === 'added' ? added : total;
    const value = Number(text);
    const min = kind === 'added' ? 1 : 0;
    if (!text.trim() || !Number.isFinite(value) || value < min || value > STOCK_MAX) {
      setFieldError({ [kind]: 'Enter the amount in mL, for example 1000.' });
      return;
    }
    setFieldError({});
    setError(null);
    setSaving(true);
    try {
      setStock(kind === 'added' ? await addAcidStock(value) : await setAcidStock(value));
      setAdded('');
      setTotal('');
      setSaved(kind === 'added' ? `✓ ${formatAcid(value)} added to the stock` : '✓ Total saved');
    } catch (err) {
      setError(toReadableMessage(err));
    } finally {
      setSaving(false);
    }
  }

  const addedValue = Number(added);
  const current = stock?.stock_mL ?? 0;

  // gauge is full at five average batches
  const full = stock?.average_batch_acid_mL ? stock.average_batch_acid_mL * 5 : 1000;
  const level = stock?.stock_mL != null ? Math.min(stock.stock_mL / full, 1) : 0;
  const low = Boolean(stock?.low_stock);
  const liquid = low ? tints.danger.strong : tints.acid.strong;

  return (
    <LatexScreen
      eyebrow="Acid stock"
      title="1% Acid Solution"
      subtitle="Add each new jar of prepared 1% formic acid solution. Each started batch takes its own amount."
    >
      {!stock && !error ? <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} /> : null}

      {stock ? (
        <Card style={styles.gaugeCard}>
          <View style={styles.jar}>
            <View style={styles.jarLid} />
            <View style={styles.jarBody}>
              <View style={[styles.jarLiquid, { height: `${level * 100}%`, backgroundColor: liquid }]} />
              <Text style={styles.jarText}>{stock.stock_mL != null ? formatAcid(stock.stock_mL) : '—'}</Text>
            </View>
          </View>
          <View style={styles.gaugeInfo}>
            <Text style={styles.infoLabel}>In stock</Text>
            <Text style={[styles.infoValue, low && { color: tints.danger.strong }]}>
              {stock.stock_mL != null ? formatAcid(stock.stock_mL) : 'Not entered'}
            </Text>
            <Text style={styles.infoLabel}>Batches left</Text>
            <Text style={styles.infoValue}>{stock.batches_left ?? '—'}</Text>
            <Text style={styles.infoLabel}>Average per batch</Text>
            <Text style={styles.infoValueSmall}>
              {stock.average_batch_acid_mL ? formatAcid(stock.average_batch_acid_mL) : '—'}
            </Text>
          </View>
        </Card>
      ) : null}

      {stock ? (
        <Banner tone={low ? 'danger' : stock.stock_mL == null ? 'warn' : 'info'} title={low ? 'Low stock' : undefined} text={stock.message} />
      ) : null}

      <Card>
        <Text style={styles.cardTitle}>➕ Add new solution</Text>
        <Text style={styles.cardText}>Prepared more 1% acid solution? Enter only the new amount.</Text>
        <FormField
          label="How much did you just prepare? (mL)"
          value={added}
          onChangeText={(text) => {
            setAdded(text);
            setSaved(null);
          }}
          placeholder="e.g. 1000"
          keyboardType="decimal-pad"
          error={fieldError.added}
        />
        {addedValue > 0 ? (
          <Text style={styles.sumText}>
            {formatAcid(current)} + {formatAcid(addedValue)} = {formatAcid(current + addedValue)}
          </Text>
        ) : null}
        {saving ? <ActivityIndicator color={colors.primary} /> : <PrimaryButton label="Add to Stock" onPress={() => save('added')} />}
        {saved ? <Text style={styles.savedText}>{saved}</Text> : null}
      </Card>

      <Card>
        <Pressable accessibilityRole="button" onPress={() => setShowCorrect((v) => !v)} style={styles.correctHead}>
          <Text style={styles.cardTitle}>✏️ Correct the total</Text>
          <Text style={styles.toggle}>{showCorrect ? 'Hide' : 'Show'}</Text>
        </Pressable>
        <Text style={styles.cardText}>Only if the jar has a different amount, e.g. after a spill or a wrong count.</Text>
        {showCorrect ? (
          <>
            <FormField
              label="Total in the jar now (mL)"
              value={total}
              onChangeText={(text) => {
                setTotal(text);
                setSaved(null);
              }}
              placeholder="e.g. 2150"
              keyboardType="decimal-pad"
              error={fieldError.total}
            />
            {saving ? null : <SecondaryButton label="Set Total" onPress={() => save('total')} />}
          </>
        ) : null}
      </Card>

      <Card style={styles.howCard}>
        <Text style={styles.cardTitle}>How it works</Text>
        <Text style={styles.cardText}>• Adding new solution puts it on top of what is left.</Text>
        <Text style={styles.cardText}>• Starting a batch timer takes that batch's acid from the stock.</Text>
        <Text style={styles.cardText}>• A batch cannot start if the stock is less than it needs.</Text>
        <Text style={styles.cardText}>• You are warned when only one batch is left.</Text>
      </Card>

      {error ? <Banner tone="danger" text={error} /> : null}
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  gaugeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.lg,
  },
  jar: {
    alignItems: 'center',
  },
  jarLid: {
    width: 64,
    height: 14,
    borderRadius: 4,
    backgroundColor: '#8C7A63',
  },
  jarBody: {
    width: 104,
    height: 150,
    marginTop: 2,
    borderRadius: radii.md,
    borderWidth: 3,
    borderColor: '#D9CFC1',
    backgroundColor: '#FBF8F3',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  jarLiquid: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    opacity: 0.85,
  },
  jarText: {
    fontSize: 15,
    fontWeight: '800',
    color: colors.textPrimary,
    backgroundColor: 'rgba(255,255,255,0.8)',
    paddingHorizontal: 6,
    borderRadius: 6,
    overflow: 'hidden',
  },
  gaugeInfo: {
    flex: 1,
    gap: 2,
  },
  infoLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: spacing.xs,
  },
  infoValue: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  infoValueSmall: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  cardText: {
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  sumText: {
    fontSize: 15,
    fontWeight: '800',
    color: tints.acid.strong,
    textAlign: 'center',
  },
  correctHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  toggle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primary,
  },
  savedText: {
    textAlign: 'center',
    fontSize: 13,
    fontWeight: '700',
    color: colors.success,
  },
  howCard: {
    backgroundColor: tints.acid.soft,
    borderColor: tints.acid.soft,
  },
});
