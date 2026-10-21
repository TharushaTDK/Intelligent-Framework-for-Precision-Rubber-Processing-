import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, radii, spacing, typography } from '../../constants/theme';
import type { AcousticDeviceStatusResponse, AcousticHealthResponse } from '../../types/moisture';

type SystemStatusCardProps = {
  health: AcousticHealthResponse | null;
  deviceStatus: AcousticDeviceStatusResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
};

export function SystemStatusCard({
  health,
  deviceStatus,
  loading,
  error,
  onRefresh,
}: SystemStatusCardProps) {
  const isModelAvailable = Boolean(health?.model_available);
  const isDeviceConfigured = Boolean(deviceStatus?.configured);

  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>System Readiness</Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Refresh system status"
          disabled={loading}
          onPress={onRefresh}
          style={({ pressed }) => [styles.refreshButton, pressed && styles.refreshButtonPressed]}
        >
          {loading ? (
            <ActivityIndicator size="small" color={colors.primary} />
          ) : (
            <Text style={styles.refreshLabel}>Refresh</Text>
          )}
        </Pressable>
      </View>

      <Text style={styles.disclaimer}>
        Checked against the FastAPI backend. Recorder address configuration does not verify live hardware power.
      </Text>

      <View style={styles.statusList}>
        {/* Quality Model Status */}
        <View style={styles.statusRow}>
          <View style={styles.statusIndicator}>
            <View
              style={[
                styles.dot,
                isModelAvailable ? styles.dotSuccess : styles.dotWarning,
              ]}
            />
          </View>
          <View style={styles.statusTextWrap}>
            <Text style={styles.statusName}>Quality Model</Text>
            <Text style={styles.statusDetail}>
              {health
                ? isModelAvailable
                  ? `Available (version: ${health.model_version || 'preliminary'})`
                  : 'Unavailable on backend'
                : 'Not checked yet'}
            </Text>
          </View>
        </View>

        {/* Device Address Configuration */}
        <View style={styles.statusRow}>
          <View style={styles.statusIndicator}>
            <View
              style={[
                styles.dot,
                isDeviceConfigured ? styles.dotSuccess : styles.dotWarning,
              ]}
            />
          </View>
          <View style={styles.statusTextWrap}>
            <Text style={styles.statusName}>Recorder Device Address</Text>
            <Text style={styles.statusDetail}>
              {deviceStatus
                ? isDeviceConfigured
                  ? 'Configured on backend (FastAPI configured)'
                  : 'Not configured on backend'
                : 'Not checked yet'}
            </Text>
          </View>
        </View>
      </View>

      {error ? (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  refreshButton: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radii.sm,
    backgroundColor: colors.background,
    minHeight: 32,
    minWidth: 64,
    alignItems: 'center',
    justifyContent: 'center',
  },
  refreshButtonPressed: {
    opacity: 0.7,
  },
  refreshLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.primary,
  },
  disclaimer: {
    fontSize: 11,
    color: colors.textSecondary,
    lineHeight: 16,
  },
  statusList: {
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  statusIndicator: {
    width: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  dotSuccess: {
    backgroundColor: colors.success,
  },
  dotWarning: {
    backgroundColor: colors.warning,
  },
  statusTextWrap: {
    flex: 1,
  },
  statusName: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  statusDetail: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  errorBox: {
    backgroundColor: colors.background,
    borderLeftWidth: 3,
    borderLeftColor: colors.critical,
    padding: spacing.sm,
    borderRadius: radii.sm,
    marginTop: spacing.xs,
  },
  errorText: {
    fontSize: 12,
    color: colors.critical,
    lineHeight: 16,
  },
});
