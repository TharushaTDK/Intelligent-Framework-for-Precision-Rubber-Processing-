import { router, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import { coagulationTimeText } from '../../features/latex/BatchSummaryCard';
import {
  Banner,
  Card,
  LatexScreen,
  ProgressBar,
  StagePan,
  StageTrack,
  formatClock,
  formatCountdown,
  stageFor,
  tints,
  useNow,
} from '../../features/latex/ui';
import { getRecommendation, toReadableMessage, updateBatchStatus } from '../../services/latexApi';
import type { CoagulationRecommendationResponse } from '../../types/latex';

export default function BatchTimer() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const now = useNow();

  const [batch, setBatch] = useState<CoagulationRecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // pop up once when the time is reached while the screen is open
  const dueShown = useRef(false);

  useFocusEffect(
    useCallback(() => {
      getRecommendation(Number(id))
        .then(setBatch)
        .catch((err) => setError(toReadableMessage(err)));
    }, [id]),
  );

  const running = batch?.status === 'in_progress';
  const start = batch?.started_at ? new Date(batch.started_at).getTime() : 0;
  const end = batch?.expected_completion_at ? new Date(batch.expected_completion_at).getTime() : 0;
  const share = batch?.status === 'completed' ? 1 : end > start ? (now - start) / (end - start) : 0;
  const due = running && now >= end;
  const stage = stageFor(share);

  useEffect(() => {
    if (due && !dueShown.current) {
      dueShown.current = true;
      Alert.alert('Coagulation time reached', `Batch #${batch?.id}: check that the coagulum is firm before sheeting.`);
    }
  }, [due, batch?.id]);

  const upcoming = (batch?.alerts ?? []).filter((alert) => new Date(alert.alert_at).getTime() > now).slice(0, 3);

  async function changeStatus(status: 'completed' | 'cancelled') {
    if (!batch) return;
    setSaving(true);
    setError(null);
    try {
      setBatch(await updateBatchStatus(batch.id, status));
    } catch (err) {
      setError(toReadableMessage(err));
    } finally {
      setSaving(false);
    }
  }

  function confirm(status: 'completed' | 'cancelled') {
    const text = status === 'completed' ? 'Is the coagulum firm and ready for sheeting?' : 'Stop this batch? It cannot be restarted.';
    Alert.alert(status === 'completed' ? 'Mark as completed' : 'Cancel batch', text, [
      { text: 'No', style: 'cancel' },
      { text: 'Yes', onPress: () => changeStatus(status) },
    ]);
  }

  return (
    <LatexScreen
      eyebrow={batch ? `Batch #${batch.id} · ${batch.latex_volume_L} L` : 'Coagulation timer'}
      title={batch?.status === 'completed' ? 'Batch Completed' : 'Coagulation Timer'}
      subtitle="Keep the pan covered and still. The stage moves with the predicted time."
      footer={
        running ? (
          saving ? (
            <ActivityIndicator color={colors.primary} />
          ) : (
            <>
              <PrimaryButton label="✓  Mark as Completed" onPress={() => confirm('completed')} />
              <SecondaryButton label="Cancel Batch" onPress={() => confirm('cancelled')} />
            </>
          )
        ) : batch ? (
          <SecondaryButton label="Back to Coagulation Home" onPress={() => router.dismissTo('/latex')} />
        ) : null
      }
    >
      {!batch && !error ? <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} /> : null}

      {batch && batch.status === 'recommended' ? (
        <Banner tone="warn" title="Timer not started" text="Finish the preparation guide to start this batch." />
      ) : null}

      {batch && batch.started_at ? (
        <>
          <Card style={styles.clockCard}>
            <Text style={styles.stageName}>{stage.label}</Text>
            <StagePan share={share} />
            <Text style={[styles.countdown, due && { color: tints.danger.strong }]}>
              {batch.status === 'completed' ? 'Done' : due ? `+${formatCountdown(now - end)}` : formatCountdown(end - now)}
            </Text>
            <Text style={styles.countdownLabel}>
              {batch.status === 'completed' ? 'Coagulation finished' : due ? 'over the predicted time' : 'time left'}
            </Text>
            <ProgressBar value={share} color={due ? tints.danger.strong : colors.primary} height={12} />
            <View style={styles.timesRow}>
              <View>
                <Text style={styles.timeLabel}>Started</Text>
                <Text style={styles.timeValue}>{formatClock(start)}</Text>
              </View>
              <View style={styles.timeRight}>
                <Text style={styles.timeLabel}>Ready at</Text>
                <Text style={styles.timeValue}>{formatClock(end)}</Text>
              </View>
            </View>
          </Card>

          <Card>
            <StageTrack share={share} />
            <Text style={styles.stageText}>{stage.description}</Text>
          </Card>

          {due ? (
            <Banner
              tone="danger"
              title="⏰ Time reached"
              text="Check the pan now. If it is firm, mark the batch completed. If not, wait and check again."
            />
          ) : null}

          {running && upcoming.length ? (
            <Card>
              <Text style={styles.cardTitle}>🔔 Next reminders</Text>
              {upcoming.map((alert) => (
                <View key={alert.alert_at + alert.type} style={styles.alertRow}>
                  <Text style={styles.alertTime}>{formatClock(alert.alert_at)}</Text>
                  <Text style={styles.alertText}>{alert.type === 'due' ? 'Time to check the pan' : alert.message.split(': ').pop()}</Text>
                </View>
              ))}
            </Card>
          ) : null}

          {batch.status === 'completed' && batch.actual_coagulation_time_hours !== null ? (
            <Card>
              <Text style={styles.cardTitle}>Result</Text>
              <View style={styles.timesRow}>
                <View>
                  <Text style={styles.timeLabel}>Predicted</Text>
                  <Text style={styles.timeValue}>{coagulationTimeText(batch.coagulation_time_hours ?? 0)}</Text>
                </View>
                <View style={styles.timeRight}>
                  <Text style={styles.timeLabel}>Actual</Text>
                  <Text style={styles.timeValue}>{coagulationTimeText(batch.actual_coagulation_time_hours)}</Text>
                </View>
              </View>
            </Card>
          ) : null}
        </>
      ) : null}

      {batch?.status === 'cancelled' ? <Banner tone="danger" title="Cancelled" text="This batch was stopped." /> : null}
      {error ? <Banner tone="danger" text={error} /> : null}
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  clockCard: {
    alignItems: 'stretch',
  },
  stageName: {
    alignSelf: 'center',
    fontSize: 14,
    fontWeight: '800',
    color: colors.primary,
    backgroundColor: tints.time.soft,
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
    borderRadius: radii.pill,
    overflow: 'hidden',
  },
  countdown: {
    textAlign: 'center',
    fontSize: 46,
    fontWeight: '800',
    color: colors.textPrimary,
    fontVariant: ['tabular-nums'],
  },
  countdownLabel: {
    textAlign: 'center',
    marginTop: -spacing.xs,
    marginBottom: spacing.xs,
    fontSize: 13,
    color: colors.textSecondary,
  },
  timesRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
  },
  timeRight: {
    alignItems: 'flex-end',
  },
  timeLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  timeValue: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  stageText: {
    marginTop: spacing.xs,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textPrimary,
    textAlign: 'center',
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  alertRow: {
    flexDirection: 'row',
    gap: spacing.md,
    alignItems: 'center',
  },
  alertTime: {
    width: 70,
    fontSize: 13,
    fontWeight: '800',
    color: colors.primary,
  },
  alertText: {
    flex: 1,
    fontSize: 13,
    color: colors.textSecondary,
  },
});
