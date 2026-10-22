import DateTimePicker from '@react-native-community/datetimepicker';
import * as Location from 'expo-location';
import { router } from 'expo-router';
import { useRef, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { FormField } from '../../components/FormField';
import { PrimaryButton } from '../../components/PrimaryButton';
import { SecondaryButton } from '../../components/SecondaryButton';
import { colors, radii, spacing, typography } from '../../constants/theme';
import { buildAdvisoryRecord, buildAdvisoryResult, weatherInputs } from '../../features/plantAdvisory/advisoryResult';
import { FieldMap } from '../../features/plantAdvisory/FieldMap';
import {
  advisoryTimeBounds,
  advisoryTimestamp,
  checkAdvisoryTime,
  FORECAST_LIMIT_DAYS,
  formatObservationHour,
  formatPickedDate,
  formatPickedTime,
  fromGps,
  fromMap,
  parseCoordinates,
  sriLankaNow,
  type FieldLocation,
} from '../../features/plantAdvisory/fieldLocation';
import {
  estimateDrc,
  estimatePrice,
  getTappingAdvisory,
  getWeatherSnapshot,
  saveAdvisoryRecord,
  toReadableMessage,
} from '../../services/plantAdvisoryApi';
import type { WeatherSnapshotResponse } from '../../types/plantAdvisory';

// GPS can hang indoors or on a cold start, so give up after a while
const GPS_TIMEOUT_MS = 20000;

const WEATHER_ROWS: { key: keyof WeatherSnapshotResponse; label: string; unit: string }[] = [
  { key: 'temperature_c', label: 'Temperature', unit: '°C' },
  { key: 'relative_humidity_pct', label: 'Humidity', unit: '%' },
  { key: 'wind_speed_kmh', label: 'Wind speed', unit: 'km/h' },
  { key: 'rainfall_last_6h_mm', label: 'Rain, last 6 h', unit: 'mm' },
  { key: 'rainfall_last_3days_mm', label: 'Rain, last 3 days', unit: 'mm' },
  { key: 'forecast_rain_next_6h_mm', label: 'Forecast rain, next 6 h', unit: 'mm' },
];

export default function FieldLocationScreen() {
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [fieldErrors, setFieldErrors] = useState<{ latitude?: string; longitude?: string }>({});
  const [gpsLoading, setGpsLoading] = useState(false);
  const [gpsMessage, setGpsMessage] = useState<string | null>(null);
  const [location, setLocation] = useState<FieldLocation | null>(null);
  // wall-clock Sri Lanka time chosen with the date and time pickers
  const [pickedTime, setPickedTime] = useState<Date>(() => sriLankaNow());
  const [pickerMode, setPickerMode] = useState<'date' | 'time' | null>(null);
  const [timeError, setTimeError] = useState<string | null>(null);
  // the coordinate boxes are a fallback; the map is the main way to choose
  const [manualEntry, setManualEntry] = useState(false);

  // bumped whenever the farmer types, so a slow GPS answer doesn't
  // overwrite coordinates they entered while it was searching
  const typedVersion = useRef(0);

  const [weather, setWeather] = useState<WeatherSnapshotResponse | null>(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  const [advisoryLoading, setAdvisoryLoading] = useState(false);
  const [advisoryError, setAdvisoryError] = useState<string | null>(null);

  // Any change to the location means the weather shown is for somewhere else.
  function clearWeather() {
    setWeather(null);
    setWeatherError(null);
    setAdvisoryError(null);
  }

  // Typed coordinates move the pin as soon as both are valid.
  function onTyped(latitudeText: string, longitudeText: string) {
    typedVersion.current += 1;
    setLatitude(latitudeText);
    setLongitude(longitudeText);
    setLocation(parseCoordinates(latitudeText, longitudeText).location);
    setFieldErrors({});
    clearWeather();
  }

  function onLatitudeChange(text: string) {
    onTyped(text, longitude);
  }

  function onLongitudeChange(text: string) {
    onTyped(latitude, text);
  }

  // A tap or drag on the map. It also counts as the farmer choosing, so a
  // GPS answer still on its way won't move the pin afterwards.
  function onMapPick(lat: number, lng: number) {
    typedVersion.current += 1;
    const picked = fromMap(lat, lng);
    setLocation(picked);
    setLatitude(String(picked.latitude));
    setLongitude(String(picked.longitude));
    setFieldErrors({});
    setGpsMessage(null);
    clearWeather();
  }

  function onTimeChange(next: Date) {
    setPickedTime(next);
    setTimeError(checkAdvisoryTime(next));
    clearWeather();
  }

  // The date picker only changes the day and the time picker only the
  // hour and minute, so each keeps the other half of the chosen time.
  function onPicked(value: Date) {
    const mode = pickerMode;
    // Android shows a dialog that is already closed by now; iOS shows
    // the picker inline, so hide it once a value is chosen
    setPickerMode(null);
    const next = new Date(pickedTime);
    if (mode === 'date') {
      next.setFullYear(value.getFullYear(), value.getMonth(), value.getDate());
    } else {
      next.setHours(value.getHours(), value.getMinutes(), 0, 0);
    }
    onTimeChange(next);
  }

  // Only runs when the farmer taps the button - Android shows its own
  // permission prompt the first time.
  async function useMyLocation() {
    setGpsLoading(true);
    setGpsMessage(null);
    clearWeather();
    const versionAtStart = typedVersion.current;

    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (permission.status !== 'granted') {
        setGpsMessage('Location permission was not given. You can type the coordinates instead.');
        return;
      }

      const servicesOn = await Location.hasServicesEnabledAsync();
      if (!servicesOn) {
        setGpsMessage('Location is turned off on this phone. Turn it on, or type the coordinates.');
        return;
      }

      const position = await Promise.race([
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), GPS_TIMEOUT_MS)),
      ]);

      if (typedVersion.current !== versionAtStart) {
        setGpsMessage('Your location was found, but you typed coordinates meanwhile, so those were kept.');
        return;
      }

      const found = fromGps(position.coords.latitude, position.coords.longitude, position.coords.accuracy);
      setLocation(found);
      setLatitude(String(found.latitude));
      setLongitude(String(found.longitude));
      setFieldErrors({});
    } catch {
      setGpsMessage('Your location could not be found. Try again, or type the coordinates.');
    } finally {
      setGpsLoading(false);
    }
  }

  async function getWeather() {
    // map, GPS and valid typed coordinates all set `location` already;
    // if it's empty, say what's missing
    const target = location;
    if (!target) {
      if (manualEntry) {
        setFieldErrors(parseCoordinates(latitude, longitude).errors);
      } else {
        setGpsMessage('Choose your field first: tap the map, use your current location, or enter coordinates.');
      }
      return;
    }

    // checked again here: a time that was fine when picked can drift
    // past the history limit if the screen stays open
    const rangeError = checkAdvisoryTime(pickedTime);
    setTimeError(rangeError);
    if (rangeError) {
      return;
    }

    setWeatherLoading(true);
    setWeatherError(null);
    setWeather(null);

    try {
      const snapshot = await getWeatherSnapshot({
        latitude: target.latitude,
        longitude: target.longitude,
        observation_timestamp: advisoryTimestamp(pickedTime),
      });
      setWeather(snapshot);
    } catch (error) {
      setWeatherError(toReadableMessage(error));
    } finally {
      setWeatherLoading(false);
    }
  }

  // DRC and tapping run together; price needs the DRC estimate, so it
  // goes after. Weather goes to all three. The latex condition isn't
  // asked for, and a location can't tell us the estate or soil, so the
  // backend fills those itself.
  async function generateAdvisory() {
    if (!location || !weather) {
      return;
    }

    setAdvisoryLoading(true);
    setAdvisoryError(null);

    const timestamp = weather.observation_timestamp;
    const inputs = weatherInputs(weather);

    try {
      const [drcResult, tappingResult] = await Promise.allSettled([
        estimateDrc({ observation_timestamp: timestamp, ...inputs }),
        getTappingAdvisory({ observation_timestamp: timestamp, ...inputs }),
      ]);

      // without a DRC there is no price and nothing worth saving
      if (drcResult.status === 'rejected') {
        setAdvisoryError(toReadableMessage(drcResult.reason));
        return;
      }
      const drc = drcResult.value;
      const tapping = tappingResult.status === 'fulfilled' ? tappingResult.value : null;
      const tappingError = tappingResult.status === 'rejected' ? toReadableMessage(tappingResult.reason) : null;

      let price = null;
      let priceError: string | null = null;
      try {
        price = await estimatePrice({
          observation_timestamp: timestamp,
          latex_drc_pct: drc.predicted_latex_drc_pct,
          temperature_c: weather.temperature_c,
          relative_humidity_pct: weather.relative_humidity_pct,
          rainfall_last_6h_mm: weather.rainfall_last_6h_mm,
        });
      } catch (error) {
        priceError = toReadableMessage(error);
      }

      // saved in the background for Component 2; a failed save doesn't
      // stop the farmer seeing the result
      try {
        await saveAdvisoryRecord(buildAdvisoryRecord(location, weather, drc, tapping, price));
      } catch (error) {
        console.warn('Advisory was not saved for Component 2:', toReadableMessage(error));
      }

      const result = buildAdvisoryResult(
        location, weather, null, drc, tapping, tappingError, price, priceError,
      );
      router.push({ pathname: '/plant-advisory/result', params: { data: JSON.stringify(result) } });
    } finally {
      setAdvisoryLoading(false);
    }
  }

  const SOURCE_TEXT = { map: 'Pin on the map', gps: 'From your current location', typed: 'Typed in' } as const;
  const locationLine = location
    ? `${SOURCE_TEXT[location.source]}: ${location.latitude}, ${location.longitude}` +
      (location.accuracy != null ? ` (about ±${location.accuracy} m)` : '')
    : null;

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
        <Text style={styles.title}>Your Field</Text>
        <Text style={styles.subtitle}>Choose where your rubber field is to get its weather.</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Field location</Text>
          <Text style={styles.hint}>Tap the map to drop a pin on your field, then drag the pin to adjust it.</Text>

          <FieldMap pin={location} onPick={onMapPick} onUnavailable={() => setManualEntry(true)} />

          {locationLine ? (
            <Text style={styles.locationText}>{locationLine}</Text>
          ) : (
            <Text style={styles.hint}>No location chosen yet.</Text>
          )}

          <SecondaryButton
            label={gpsLoading ? 'Finding your location...' : 'Use My Current Location'}
            onPress={() => {
              if (!gpsLoading) {
                useMyLocation();
              }
            }}
          />
          {gpsLoading ? <ActivityIndicator color={colors.primary} /> : null}
          {gpsMessage ? <Text style={styles.warningText}>{gpsMessage}</Text> : null}

          {manualEntry ? (
            <>
              <FormField
                label="Latitude"
                value={latitude}
                onChangeText={onLatitudeChange}
                placeholder="e.g. 6.9271"
                keyboardType="numbers-and-punctuation"
                error={fieldErrors.latitude}
              />
              <FormField
                label="Longitude"
                value={longitude}
                onChangeText={onLongitudeChange}
                placeholder="e.g. 79.8612"
                keyboardType="numbers-and-punctuation"
                error={fieldErrors.longitude}
              />
            </>
          ) : (
            <Pressable accessibilityRole="button" onPress={() => setManualEntry(true)}>
              <Text style={styles.linkText}>Enter coordinates instead</Text>
            </Pressable>
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>When</Text>
          <Text style={styles.hint}>
            Advisory date and time, in Sri Lanka time. Future times, up to {FORECAST_LIMIT_DAYS} days
            ahead, use the forecast.
          </Text>

          <View style={styles.pickerRow}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Date, ${formatPickedDate(pickedTime)}. Tap to change.`}
              onPress={() => setPickerMode('date')}
              style={[styles.option, styles.pickerButton, pickerMode === 'date' && styles.optionSelected]}
            >
              <View style={styles.optionText}>
                <Text style={styles.optionCaption}>Date</Text>
                <Text style={styles.optionLabel}>{formatPickedDate(pickedTime)}</Text>
              </View>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Time, ${formatPickedTime(pickedTime)}. Tap to change.`}
              onPress={() => setPickerMode('time')}
              style={[styles.option, styles.pickerButton, pickerMode === 'time' && styles.optionSelected]}
            >
              <View style={styles.optionText}>
                <Text style={styles.optionCaption}>Time</Text>
                <Text style={styles.optionLabel}>{formatPickedTime(pickedTime)}</Text>
              </View>
            </Pressable>
          </View>

          {pickerMode ? (
            <DateTimePicker
              value={pickedTime}
              mode={pickerMode}
              display={Platform.OS === 'ios' ? (pickerMode === 'date' ? 'inline' : 'spinner') : 'default'}
              minimumDate={pickerMode === 'date' ? advisoryTimeBounds().min : undefined}
              maximumDate={pickerMode === 'date' ? advisoryTimeBounds().max : undefined}
              onValueChange={(_, value) => onPicked(value)}
              onDismiss={() => setPickerMode(null)}
            />
          ) : null}

          {timeError ? <Text style={styles.timeError}>{timeError}</Text> : null}

          <Pressable accessibilityRole="button" onPress={() => onTimeChange(sriLankaNow())}>
            <Text style={styles.linkText}>Use current time</Text>
          </Pressable>
        </View>

        <PrimaryButton
          label={weatherLoading ? 'Getting weather...' : 'Get Weather'}
          onPress={() => {
            if (!weatherLoading && !gpsLoading) {
              getWeather();
            }
          }}
        />

        {weatherLoading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Getting the weather for your field...</Text>
          </View>
        ) : null}

        {weatherError ? (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Weather could not be retrieved</Text>
            <Text style={styles.errorMessage}>{weatherError}</Text>
            <SecondaryButton label="Try Again" onPress={getWeather} />
          </View>
        ) : null}

        {weather ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Weather</Text>
            <Text style={styles.weatherMeta}>
              {formatObservationHour(weather.observation_timestamp)} · {weather.provider}
            </Text>

            <View style={styles.weatherGrid}>
              {WEATHER_ROWS.map((row) => {
                const value = weather[row.key];
                return (
                  <View key={row.key} style={styles.weatherTile}>
                    <Text style={styles.weatherLabel}>{row.label}</Text>
                    {typeof value === 'number' ? (
                      <Text style={styles.weatherValue}>
                        {Number(value.toFixed(2))}
                        <Text style={styles.weatherUnit}> {row.unit}</Text>
                      </Text>
                    ) : (
                      // say it's missing rather than showing a zero
                      <Text style={styles.missingText}>Not available</Text>
                    )}
                  </View>
                );
              })}
            </View>

            {weather.data_quality.observation_in_future ? (
              <Text style={styles.warningText}>
                Forecast: this is a future time, so every value is a forecast.
              </Text>
            ) : null}
          </View>
        ) : null}

        {weather ? (
          <PrimaryButton
            label={advisoryLoading ? 'Generating advisory...' : 'Generate Advisory'}
            onPress={() => {
              if (!advisoryLoading) {
                generateAdvisory();
              }
            }}
          />
        ) : null}

        {advisoryLoading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>Running the models...</Text>
          </View>
        ) : null}

        {advisoryError ? (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>The advisory could not be generated</Text>
            <Text style={styles.errorMessage}>{advisoryError}</Text>
            <SecondaryButton label="Try Again" onPress={generateAdvisory} />
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

// same layout language as the Let's Set the Latex screens
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
    paddingBottom: spacing.xl,
    gap: spacing.lg,
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
  card: {
    backgroundColor: colors.surface,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    gap: spacing.md,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  // option cards match the "What do you need?" choice on the Latex batch screen
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
  pickerRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  pickerButton: {
    flex: 1,
  },
  timeError: {
    fontSize: 13,
    color: colors.critical,
    lineHeight: 19,
  },
  hint: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 19,
  },
  linkText: {
    textAlign: 'center',
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary,
    paddingVertical: spacing.xs,
  },
  locationText: {
    fontSize: 13,
    color: colors.primary,
    fontWeight: '600',
  },
  warningText: {
    fontSize: 13,
    color: colors.accent,
    lineHeight: 19,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  loadingText: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  errorCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.critical,
    padding: spacing.md,
    gap: spacing.sm,
  },
  errorTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.critical,
  },
  errorMessage: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 19,
  },
  weatherMeta: {
    marginTop: -spacing.sm,
    fontSize: 13,
    color: colors.textSecondary,
  },
  weatherGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  weatherTile: {
    width: '48%',
    flexGrow: 1,
    backgroundColor: colors.background,
    borderRadius: radii.sm,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.sm,
  },
  weatherLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  weatherValue: {
    marginTop: 2,
    fontSize: 20,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  weatherUnit: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.textSecondary,
  },
  missingText: {
    marginTop: 4,
    fontSize: 13,
    fontWeight: '600',
    color: colors.critical,
  },
});
