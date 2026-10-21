import * as Location from 'expo-location';
import { router } from 'expo-router';
import { useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { formatDrc } from '../../features/latex/BatchSummaryCard';
import { FieldMap } from '../../features/plantAdvisory/FieldMap';
import { fromGps, fromMap, type FieldLocation } from '../../features/plantAdvisory/fieldLocation';
import { createRecommendation, getLocationDrc, toReadableMessage } from '../../services/latexApi';
import type { CoagulationOutput, LocationDrcResponse } from '../../types/latex';

// latex volume range of the Component 2 training data
const VOLUME_MIN = 1.0;
const VOLUME_MAX = 2.0;

// GPS can hang indoors or on a cold start, so give up after a while
const GPS_TIMEOUT_MS = 20000;

const OUTPUT_OPTIONS: { key: string; label: string; caption: string; outputs: CoagulationOutput[] }[] = [
  { key: 'water_acid', label: 'Water & Acid', caption: 'Water and 1% formic acid solution', outputs: ['water_acid'] },
  { key: 'coagulation_time', label: 'Coagulation Time', caption: 'How long the batch takes', outputs: ['coagulation_time'] },
  { key: 'both', label: 'Both', caption: 'All three values', outputs: ['water_acid', 'coagulation_time'] },
];

export default function NewBatch() {
  const [location, setLocation] = useState<FieldLocation | null>(null);
  const [locationDrc, setLocationDrc] = useState<LocationDrcResponse | null>(null);
  const [drcLoading, setDrcLoading] = useState(false);
  const [drcError, setDrcError] = useState<string | null>(null);
  const [gpsLoading, setGpsLoading] = useState(false);
  const [gpsMessage, setGpsMessage] = useState<string | null>(null);
  // only the latest location may set the DRC, a slow answer for an old pin is dropped
  const pickVersion = useRef(0);
  const [volume, setVolume] = useState('');
  const [option, setOption] = useState('both');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // a new location means a new DRC, worked out from the weather there now
  async function chooseLocation(picked: FieldLocation) {
    const version = ++pickVersion.current;
    setLocation(picked);
    setLocationDrc(null);
    setDrcError(null);
    setSubmitError(null);
    setDrcLoading(true);

    try {
      const result = await getLocationDrc(picked.latitude, picked.longitude);
      if (version === pickVersion.current) {
        setLocationDrc(result);
      }
    } catch (error) {
      if (version === pickVersion.current) {
        setDrcError(toReadableMessage(error));
      }
    } finally {
      if (version === pickVersion.current) {
        setDrcLoading(false);
      }
    }
  }

  function onMapPick(lat: number, lng: number) {
    setGpsMessage(null);
    chooseLocation(fromMap(lat, lng));
  }

  // only runs when the farmer taps the button, Android asks for permission the first time
  async function useMyLocation() {
    setGpsLoading(true);
    setGpsMessage(null);
    const versionAtStart = pickVersion.current;

    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (permission.status !== 'granted') {
        setGpsMessage('Location permission was not given. Tap the map instead.');
        return;
      }

      const servicesOn = await Location.hasServicesEnabledAsync();
      if (!servicesOn) {
        setGpsMessage('Location is turned off on this phone. Turn it on, or tap the map.');
        return;
      }

      const position = await Promise.race([
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), GPS_TIMEOUT_MS)),
      ]);

      // the farmer tapped the map while GPS was searching, keep their pin
      if (pickVersion.current !== versionAtStart) {
        return;
      }
      chooseLocation(fromGps(position.coords.latitude, position.coords.longitude, position.coords.accuracy));
    } catch {
      setGpsMessage('Your location could not be found. Try again, or tap the map.');
    } finally {
      setGpsLoading(false);
    }
  }

  function validate(): Record<string, string> {
    const nextErrors: Record<string, string> = {};
    const volumeValue = Number(volume);

    if (!locationDrc) {
      nextErrors.location = 'Choose the location on the map first.';
    }
    if (!volume.trim() || !Number.isFinite(volumeValue) || volumeValue < VOLUME_MIN || volumeValue > VOLUME_MAX) {
      nextErrors.volume = `Enter a latex volume between ${VOLUME_MIN.toFixed(1)} and ${VOLUME_MAX.toFixed(1)} L.`;
    }
    return nextErrors;
  }

  async function handleSubmit() {
    const nextErrors = validate();
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0 || !locationDrc) {
      return;
    }

    const selected = OUTPUT_OPTIONS.find((item) => item.key === option) ?? OUTPUT_OPTIONS[2];

    setSubmitError(null);
    setLoading(true);

    try {
      const batch = await createRecommendation({
        latitude: locationDrc.latitude,
        longitude: locationDrc.longitude,
        observation_timestamp: locationDrc.observation_timestamp,
        latex_volume_L: Number(volume),
        outputs: selected.outputs,
      });
      router.push({ pathname: '/latex/result', params: { id: String(batch.id) } });
    } catch (error) {
      setSubmitError(toReadableMessage(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backButton}>
        <Text style={styles.backLabel}>{'‹ Back'}</Text>
      </Pressable>

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.title}>New Batch</Text>
        <Text style={styles.subtitle}>Choose where the latex was collected, then enter the batch.</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Location</Text>
          <Text style={styles.rangeText}>Tap the map where the latex was collected.</Text>
          <FieldMap
            pin={location}
            onPick={onMapPick}
            onUnavailable={() => setGpsMessage('The map could not be loaded. Use your current location instead.')}
          />
          <SecondaryButton
            label={gpsLoading ? 'Finding your location...' : 'Use My Current Location'}
            onPress={() => {
              if (!gpsLoading) {
                useMyLocation();
              }
            }}
          />
          {gpsMessage ? <Text style={styles.errorText}>{gpsMessage}</Text> : null}

          {drcLoading ? (
            <View style={styles.drcRow}>
              <ActivityIndicator color={colors.primary} />
              <Text style={styles.loadingText}>Getting the DRC for this location...</Text>
            </View>
          ) : null}
          {locationDrc && !drcLoading ? (
            <View style={styles.drcBox}>
              <Text style={styles.drcLabel}>Dry rubber content</Text>
              <Text style={styles.drcValue}>{formatDrc(locationDrc.drc)}</Text>
              <Text style={styles.rangeText}>Location {locationDrc.location}</Text>
            </View>
          ) : null}
          {drcError ? <Text style={styles.errorText}>{drcError}</Text> : null}
          {errors.location && !locationDrc && !drcLoading ? (
            <Text style={styles.errorText}>{errors.location}</Text>
          ) : null}
        </View>

        <View style={styles.card}>
          <FormField
            label="Latex volume (L)"
            value={volume}
            onChangeText={setVolume}
            placeholder="e.g. 1.5"
            keyboardType="decimal-pad"
            error={errors.volume}
          />
          {!errors.volume ? (
            <Text style={styles.rangeText}>Range: {VOLUME_MIN.toFixed(1)} – {VOLUME_MAX.toFixed(1)} L</Text>
          ) : null}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>What do you need?</Text>
          {OUTPUT_OPTIONS.map((item) => {
            const selected = item.key === option;
            return (
              <Pressable
                key={item.key}
                accessibilityRole="radio"
                accessibilityState={{ selected }}
                onPress={() => setOption(item.key)}
                style={[styles.option, selected && styles.optionSelected]}
              >
                <View style={[styles.radio, selected && styles.radioSelected]}>
                  {selected ? <View style={styles.radioDot} /> : null}
                </View>
                <View style={styles.optionText}>
                  <Text style={styles.optionLabel}>{item.label}</Text>
                  <Text style={styles.optionCaption}>{item.caption}</Text>
                </View>
              </Pressable>
            );
          })}
        </View>

        {submitError ? (
          <View style={styles.submitErrorBox}>
            <Text style={styles.submitErrorText}>{submitError}</Text>
          </View>
        ) : null}

        {loading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Getting the recommendation...</Text>
          </View>
        ) : (
          <PrimaryButton
            label="Get Recommendation"
            onPress={() => {
              if (!drcLoading) {
                handleSubmit();
              }
            }}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  backButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  backLabel: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.primary,
  },
  scrollContent: {
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.xxl,
    gap: spacing.md,
  },
  title: {
    fontSize: typography.title.fontSize,
    fontWeight: typography.title.fontWeight,
    color: colors.textPrimary,
  },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    gap: spacing.sm,
  },
  rangeText: {
    marginTop: -spacing.xs,
    fontSize: 12,
    color: colors.textSecondary,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  drcRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  drcBox: {
    backgroundColor: colors.background,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  drcLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  drcValue: {
    marginTop: 2,
    fontSize: 22,
    fontWeight: '700',
    color: colors.primary,
  },
  errorText: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.critical,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    minHeight: 56,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  optionSelected: {
    borderColor: colors.primary,
    backgroundColor: colors.background,
  },
  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioSelected: {
    borderColor: colors.primary,
  },
  radioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
  optionText: {
    flex: 1,
  },
  optionLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  optionCaption: {
    marginTop: 2,
    fontSize: 12,
    color: colors.textSecondary,
  },
  submitErrorBox: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.critical,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  submitErrorText: {
    color: colors.critical,
    fontSize: 13,
    fontWeight: '600',
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
