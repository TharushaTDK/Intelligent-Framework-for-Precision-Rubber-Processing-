import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import { Banner, Card, LatexScreen, ProgressBar, tints } from '../../features/latex/ui';
import { getRecommendation, toReadableMessage, updateBatchStatus } from '../../services/latexApi';
import type { CoagulationRecommendationResponse, CoagulationStep } from '../../types/latex';

export default function BatchGuide() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const [batch, setBatch] = useState<CoagulationRecommendationResponse | null>(null);
  const [done, setDone] = useState<Record<number, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [stockProblem, setStockProblem] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    getRecommendation(Number(id))
      .then(setBatch)
      .catch((err) => setError(toReadableMessage(err)));
  }, [id]);

  // the last two steps are starting the timer and checking at the end
  const steps = batch?.steps ?? [];
  const prepSteps = steps.slice(0, -2);
  const afterSteps = steps.slice(-2);
  const doneCount = prepSteps.filter((step) => done[step.number]).length;
  const ready = prepSteps.length > 0 && doneCount === prepSteps.length;

  async function startTimer() {
    if (!batch || !ready) return;
    setStarting(true);
    setError(null);
    setStockProblem(false);
    try {
      await updateBatchStatus(batch.id, 'in_progress');
      router.replace({ pathname: '/latex/timer', params: { id: String(batch.id) } });
    } catch (err) {
      const message = toReadableMessage(err);
      setError(message);
      setStockProblem(message.includes('acid solution'));
    } finally {
      setStarting(false);
    }
  }

  return (
    <LatexScreen
      eyebrow={batch ? `Batch #${batch.id} · ${batch.latex_volume_L} L` : 'Batch guide'}
      title="Prepare the Batch"
      subtitle="Tick each step as you do it. The timer starts when the pan is ready."
      footer={
        batch ? (
          starting ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color={colors.primary} />
              <Text style={styles.loadingText}>Starting the timer...</Text>
            </View>
          ) : (
            <View style={!ready ? styles.disabled : undefined}>
              <PrimaryButton
                label={ready ? '⏱  Start Coagulation Timer' : `Finish the steps (${doneCount}/${prepSteps.length})`}
                onPress={startTimer}
              />
            </View>
          )
        ) : null
      }
    >
      {!batch && !error ? <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} /> : null}

      {batch ? (
        <Card>
          <View style={styles.progressHead}>
            <Text style={styles.progressTitle}>Preparation</Text>
            <Text style={styles.progressCount}>
              {doneCount} of {prepSteps.length} done
            </Text>
          </View>
          <ProgressBar value={prepSteps.length ? doneCount / prepSteps.length : 0} />
        </Card>
      ) : null}

      {prepSteps.map((step) => (
        <StepRow
          key={step.number}
          step={step}
          checked={Boolean(done[step.number])}
          onToggle={() => setDone((prev) => ({ ...prev, [step.number]: !prev[step.number] }))}
        />
      ))}

      {afterSteps.length ? (
        <Card style={styles.afterCard}>
          <Text style={styles.afterTitle}>After you start</Text>
          {afterSteps.map((step) => (
            <Text key={step.number} style={styles.afterText}>
              {step.number}. {step.detail}
            </Text>
          ))}
        </Card>
      ) : null}

      <Banner tone="warn" title="Safety" text="Wear gloves. Never mix concentrated acid straight into the latex." />

      {error ? <Banner tone="danger" text={error} /> : null}
      {stockProblem ? <SecondaryButton label="Update Acid Stock" onPress={() => router.push('/latex/stock')} /> : null}
    </LatexScreen>
  );
}

function StepRow({ step, checked, onToggle }: { step: CoagulationStep; checked: boolean; onToggle: () => void }) {
  return (
    <Pressable
      accessibilityRole="checkbox"
      accessibilityState={{ checked }}
      onPress={onToggle}
      style={({ pressed }) => [styles.step, checked && styles.stepDone, pressed && { opacity: 0.85 }]}
    >
      <View style={[styles.check, checked && styles.checkOn]}>
        <Text style={styles.checkText}>{checked ? '✓' : step.number}</Text>
      </View>
      <View style={styles.stepText}>
        <Text style={[styles.stepTitle, checked && styles.stepTitleDone]}>{step.title}</Text>
        <Text style={styles.stepDetail}>{step.detail}</Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  progressHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  progressTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  progressCount: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.primary,
  },
  step: {
    flexDirection: 'row',
    gap: spacing.md,
    alignItems: 'flex-start',
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1.5,
    borderColor: colors.border,
    padding: spacing.md,
  },
  stepDone: {
    borderColor: colors.primary,
    backgroundColor: tints.time.soft,
  },
  check: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 2,
    borderColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkOn: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  checkText: {
    fontSize: 14,
    fontWeight: '800',
    color: colors.primaryDark,
  },
  stepText: {
    flex: 1,
    gap: 2,
  },
  stepTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  stepTitleDone: {
    color: colors.primary,
  },
  stepDetail: {
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  afterCard: {
    backgroundColor: tints.water.soft,
    borderColor: tints.water.soft,
  },
  afterTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: tints.water.strong,
  },
  afterText: {
    fontSize: 13,
    lineHeight: 19,
    color: colors.textPrimary,
  },
  disabled: {
    opacity: 0.55,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  loadingText: {
    color: colors.textSecondary,
    fontSize: 13,
  },
});
