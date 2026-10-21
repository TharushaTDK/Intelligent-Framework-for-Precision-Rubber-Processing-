import { router } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import {
  captureMeasurement,
  getAcousticDeviceStatus,
  getAcousticHealth,
  toReadableMoistureError,
} from '../../services/moistureApi';
import type {
  AcousticDeviceStatusResponse,
  AcousticHealthResponse,
  AcousticMeasurementResponse,
} from '../../types/moisture';
import { CaptureForm } from './CaptureForm';
import { MeasurementResultCard } from './MeasurementResultCard';
import { SavedResultLookup } from './SavedResultLookup';

type TabMode = 'capture' | 'lookup';

export function MoistureScreen() {
  const [activeTab, setActiveTab] = useState<TabMode>('capture');

  // Input states (preserved across errors)
  const [sheetId, setSheetId] = useState('');
  const [conditionId, setConditionId] = useState('');

  // System status
  const [health, setHealth] = useState<AcousticHealthResponse | null>(null);
  const [deviceStatus, setDeviceStatus] = useState<AcousticDeviceStatusResponse | null>(null);
  const [checkingStatus, setCheckingStatus] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  // Capture & Result states
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [measurementResult, setMeasurementResult] = useState<AcousticMeasurementResponse | null>(null);

  // Guard against duplicate taps or concurrent submissions
  const isCapturingRef = useRef(false);

  const fetchSystemStatus = useCallback(async () => {
    setCheckingStatus(true);
    setStatusError(null);
    setHealth(null);
    setDeviceStatus(null);
    try {
      const [healthRes, deviceRes] = await Promise.all([
        getAcousticHealth().catch((err) => {
          throw err;
        }),
        getAcousticDeviceStatus().catch((err) => {
          throw err;
        }),
      ]);
      setHealth(healthRes);
      setDeviceStatus(deviceRes);
    } catch (err: unknown) {
      setStatusError(toReadableMoistureError(err));
    } finally {
      setCheckingStatus(false);
    }
  }, []);

  useEffect(() => {
    fetchSystemStatus();
  }, [fetchSystemStatus]);

  async function handleStartCapture() {
    if (isCapturingRef.current) return;
    isCapturingRef.current = true;
    setIsCapturing(true);
    setCaptureError(null);
    setMeasurementResult(null); // Clear previous result immediately

    try {
      const response = await captureMeasurement({
        sheet_id: sheetId,
        condition_id: conditionId,
      });
      setMeasurementResult(response);
    } catch (err: unknown) {
      setCaptureError(toReadableMoistureError(err));
    } finally {
      setIsCapturing(false);
      isCapturingRef.current = false;
    }
  }

  function handleRecordAgain() {
    // Clear result to return to form, preserve IDs for rapid sequential testing
    setMeasurementResult(null);
    setCaptureError(null);
  }

  function handleBackToSetup() {
    setMeasurementResult(null);
    setCaptureError(null);
  }

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      {/* Top Navigation */}
      <View style={styles.topBar}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Back to Home"
          onPress={() => router.back()}
          style={styles.backButton}
        >
          <Text style={styles.backLabel}>{'‹ Back'}</Text>
        </Pressable>
      </View>

      <KeyboardAvoidingView
        style={styles.keyboardAvoid}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Screen Title & Header */}
          <View style={styles.header}>
            <Text style={styles.title}>How’s the Moisture?</Text>
            <Text style={styles.subtitle}>
              Record a standardized rubber-sheet strike and check recording quality.
            </Text>
          </View>

          {/* Mode Switcher Tabs */}
          <View style={styles.tabBar}>
            <Pressable
              accessibilityRole="tab"
              accessibilityState={{ selected: activeTab === 'capture' }}
              onPress={() => setActiveTab('capture')}
              style={[styles.tab, activeTab === 'capture' && styles.tabActive]}
            >
              <Text style={[styles.tabLabel, activeTab === 'capture' && styles.tabLabelActive]}>
                Capture & Quality
              </Text>
            </Pressable>
            <Pressable
              accessibilityRole="tab"
              accessibilityState={{ selected: activeTab === 'lookup' }}
              onPress={() => setActiveTab('lookup')}
              style={[styles.tab, activeTab === 'lookup' && styles.tabActive]}
            >
              <Text style={[styles.tabLabel, activeTab === 'lookup' && styles.tabLabelActive]}>
                Find Saved Result
              </Text>
            </Pressable>
          </View>

          {/* Active Tab Content */}
          {activeTab === 'capture' ? (
            measurementResult ? (
              <View style={styles.resultContainer}>
                <MeasurementResultCard measurement={measurementResult} />
                <View style={styles.resultActions}>
                  <PrimaryButton label="Record again" onPress={handleRecordAgain} />
                  <SecondaryButton label="Back to setup" onPress={handleBackToSetup} />
                </View>
              </View>
            ) : (
              <CaptureForm
                sheetId={sheetId}
                setSheetId={setSheetId}
                conditionId={conditionId}
                setConditionId={setConditionId}
                health={health}
                deviceStatus={deviceStatus}
                checkingStatus={checkingStatus}
                statusError={statusError}
                onRefreshStatus={fetchSystemStatus}
                isCapturing={isCapturing}
                onStartCapture={handleStartCapture}
                captureError={captureError}
              />
            )
          ) : (
            <SavedResultLookup />
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  topBar: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xs,
    paddingBottom: spacing.xs,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingVertical: spacing.xs,
    paddingRight: spacing.md,
  },
  backLabel: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.primary,
  },
  keyboardAvoid: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl,
    gap: spacing.lg,
  },
  header: {
    gap: spacing.xs,
  },
  title: {
    fontSize: typography.display.fontSize,
    fontWeight: typography.display.fontWeight,
    color: colors.textPrimary,
  },
  subtitle: {
    fontSize: typography.body.fontSize,
    color: colors.textSecondary,
    lineHeight: 22,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: colors.surface,
    borderRadius: radii.pill,
    padding: 4,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tab: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderRadius: radii.pill,
  },
  tabActive: {
    backgroundColor: colors.primary,
  },
  tabLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  tabLabelActive: {
    color: colors.textOnPrimary,
  },
  resultContainer: {
    gap: spacing.lg,
  },
  resultActions: {
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
});
