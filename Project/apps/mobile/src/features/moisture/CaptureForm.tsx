import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { FormField } from '../../components/FormField';
import { colors, radii, shadows, spacing, typography } from '../../constants/theme';
import type {
  AcousticDeviceStatusResponse,
  AcousticHealthResponse,
} from '../../types/moisture';
import { SystemStatusCard } from './SystemStatusCard';

type CaptureFormProps = {
  sheetId: string;
  setSheetId: (val: string) => void;
  conditionId: string;
  setConditionId: (val: string) => void;
  health: AcousticHealthResponse | null;
  deviceStatus: AcousticDeviceStatusResponse | null;
  checkingStatus: boolean;
  statusError: string | null;
  onRefreshStatus: () => void;
  isCapturing: boolean;
  onStartCapture: () => void;
  captureError: string | null;
};

const PREPARATION_STEPS = [
  'Power the ESP32 recording device and ensure Wi-Fi connectivity.',
  'Keep the microphone at the standard distance and orientation used during collection.',
  'Use the standard wooden striker mechanism.',
  'Be ready to deliver exactly one clean strike immediately after tapping "Start recording".',
];

export function CaptureForm({
  sheetId,
  setSheetId,
  conditionId,
  setConditionId,
  health,
  deviceStatus,
  checkingStatus,
  statusError,
  onRefreshStatus,
  isCapturing,
  onStartCapture,
  captureError,
}: CaptureFormProps) {
  return (
    <View style={styles.container}>
      {/* System Status Check */}
      <SystemStatusCard
        health={health}
        deviceStatus={deviceStatus}
        loading={checkingStatus}
        error={statusError}
        onRefresh={onRefreshStatus}
      />

      {/* Preparation Instructions Card */}
      <View style={styles.prepCard}>
        <Text style={styles.prepCardTitle}>Preparation & Standard Setup</Text>
        <Text style={styles.prepCardSubtitle}>
          Follow these experimental procedures to produce valid acoustic data:
        </Text>
        <View style={styles.stepList}>
          {PREPARATION_STEPS.map((step, index) => (
            <View key={index} style={styles.stepRow}>
              <View style={styles.stepNumberBadge}>
                <Text style={styles.stepNumberText}>{index + 1}</Text>
              </View>
              <Text style={styles.stepText}>{step}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Optional Metadata Inputs */}
      <View style={styles.inputsCard}>
        <Text style={styles.inputsHeading}>Optional Identifiers</Text>
        <Text style={styles.inputsDescription}>
          Attach research identifiers to this measurement (up to 64 characters each).
        </Text>

        <FormField
          label="Sheet ID (optional)"
          value={sheetId}
          onChangeText={setSheetId}
          placeholder="e.g. RSS-2026-04-A"
        />
        <FormField
          label="Condition ID (optional)"
          value={conditionId}
          onChangeText={setConditionId}
          placeholder="e.g. DRY-DAY-03"
        />
      </View>

      {/* Capture Error Box */}
      {captureError ? (
        <View style={styles.captureErrorBox}>
          <Text style={styles.captureErrorTitle}>Capture or Analysis Notice</Text>
          <Text style={styles.captureErrorText}>{captureError}</Text>
        </View>
      ) : null}

      {/* In-Flight Status or Action Button */}
      {isCapturing ? (
        <View style={styles.capturingCard}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.capturingTitle}>Recording / processing…</Text>
          <Text style={styles.capturingDescription}>
            Capture requested. Make one standard strike immediately after tapping Start recording; the device does not send a recording-start confirmation.
          </Text>
          <Text style={styles.capturingNote}>
            Do not start another capture while this request is pending. The app waits up to 60 seconds for a response.
          </Text>
        </View>
      ) : (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Start recording"
          onPress={onStartCapture}
          style={({ pressed }) => [
            styles.captureButton,
            pressed && styles.captureButtonPressed,
          ]}
        >
          <Text style={styles.captureButtonLabel}>Start recording</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.lg,
  },
  prepCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  prepCardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  prepCardSubtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 18,
  },
  stepList: {
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
  },
  stepNumberBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
  },
  stepNumberText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.primaryDark,
  },
  stepText: {
    flex: 1,
    fontSize: 13,
    color: colors.textPrimary,
    lineHeight: 18,
  },
  inputsCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.md,
  },
  inputsHeading: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  inputsDescription: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: -spacing.xs,
  },
  captureErrorBox: {
    backgroundColor: '#FFF5F5',
    borderLeftWidth: 3,
    borderLeftColor: colors.critical,
    borderRadius: radii.sm,
    padding: spacing.md,
    gap: spacing.xs,
  },
  captureErrorTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.critical,
  },
  captureErrorText: {
    fontSize: 13,
    color: colors.textPrimary,
    lineHeight: 18,
  },
  captureButton: {
    minHeight: 56,
    borderRadius: radii.pill,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    ...shadows.soft,
  },
  captureButtonPressed: {
    backgroundColor: colors.primaryDark,
  },
  captureButtonLabel: {
    color: colors.textOnPrimary,
    fontSize: typography.button.fontSize,
    fontWeight: typography.button.fontWeight,
    letterSpacing: typography.button.letterSpacing,
  },
  capturingCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.primaryLight,
    padding: spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  capturingTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: colors.primary,
    marginTop: spacing.xs,
  },
  capturingDescription: {
    fontSize: 13,
    color: colors.textPrimary,
    textAlign: 'center',
    lineHeight: 18,
  },
  capturingNote: {
    fontSize: 11,
    color: colors.textSecondary,
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
