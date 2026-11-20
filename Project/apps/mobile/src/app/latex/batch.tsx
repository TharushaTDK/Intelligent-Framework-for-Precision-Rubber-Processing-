import * as Location from 'expo-location';
import { router } from 'expo-router';
import { useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing } from '../../constants/theme';
import { formatDrc } from '../../features/latex/BatchSummaryCard';
import { Banner, Card, LatexScreen } from '../../features/latex/ui';
import { FieldMap } from '../../features/plantAdvisory/FieldMap';
import { fromGps, fromMap, type FieldLocation } from '../../features/plantAdvisory/fieldLocation';
import { createRecommendation, getLocationDrc, toReadableMessage } from '../../services/latexApi';
import type { CoagulationOutput, LocationDrcResponse } from '../../types/latex';

// latex volume range of the Component 2 training data
const VOLUME_MIN = 1.0;
const VOLUME_MAX = 2.0;

// Sri Lanka with a small margin, points outside it are not used
const SRI_LANKA = { south: 5.7, west: 79.4, north: 10.0, east: 82.1 };
const OUTSIDE_MESSAGE = 'This location is outside Sri Lanka. Choose a place in Sri Lanka.';

function inSriLanka(latitude: number, longitude: number): boolean {
  return (
    latitude >= SRI_LANKA.south &&
    latitude <= SRI_LANKA.north &&
    longitude >= SRI_LANKA.west &&
    longitude <= SRI_LANKA.east
  );
}

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
    if (!inSriLanka(lat, lng)) {
      setGpsMessage(OUTSIDE_MESSAGE);
      return;
    }
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
      if (!inSriLanka(position.coords.latitude, position.coords.longitude)) {
        setGpsMessage(OUTSIDE_MESSAGE);
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
    <LatexScreen
      eyebrow="Step 1 of 3"
      title="New Batch"
      subtitle="Choose where the latex was collected, then enter the batch."
    >
        <Card>
          <View style={styles.cardHead}>
            <Text style={styles.cardNumber}>1</Text>
            <Text style={styles.cardTitle}>Location</Text>
          </View>
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
              <View>
                <Text style={styles.drcLabel}>Dry rubber content</Text>
                <Text style={styles.drcValue}>{formatDrc(locationDrc.drc)}</Text>
              </View>
              <Text style={styles.drcPlace}>📍 {locationDrc.location}</Text>
            </View>
          ) : null}
          {drcError ? <Text style={styles.errorText}>{drcError}</Text> : null}
          {errors.location && !locationDrc && !drcLoading ? (
            <Text style={styles.errorText}>{errors.location}</Text>
          ) : null}
        </Card>

        <Card>
          <View style={styles.cardHead}>
            <Text style={styles.cardNumber}>2</Text>
            <Text style={styles.cardTitle}>Latex volume</Text>
          </View>
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
        </Card>

        <Card>
          <View style={styles.cardHead}>
            <Text style={styles.cardNumber}>3</Text>
            <Text style={styles.cardTitle}>What do you need?</Text>
          </View>
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
        </Card>

        {submitError ? <Banner tone="danger" text={submitError} /> : null}

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
    </LatexScreen>
  );
}

const styles = StyleSheet.create({
  cardHead: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  cardNumber: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: colors.primary,
    color: colors.textOnPrimary,
    textAlign: 'center',
    textAlignVertical: 'center',
    lineHeight: 26,
    fontSize: 13,
    fontWeight: '800',
    overflow: 'hidden',
  },
  rangeText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  drcRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  drcBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: '#E4F2E8',
    borderRadius: radii.md,
    padding: spacing.md,
  },
  drcLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  drcValue: {
    marginTop: 2,
    fontSize: 26,
    fontWeight: '800',
    color: colors.primary,
  },
  drcPlace: {
    flexShrink: 1,
    fontSize: 12,
    color: colors.textSecondary,
    textAlign: 'right',
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
    borderWidth: 1.5,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  optionSelected: {
    borderColor: colors.primary,
    backgroundColor: '#E4F2E8',
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
