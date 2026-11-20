import Constants from 'expo-constants';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BrandMark } from '../components/BrandMark';
import { ChevronIcon, StatusDot } from '../components/UiIcons';
import { colors, radii, spacing } from '../constants/theme';
import { COMPONENTS } from '../features/home/componentCatalog';
import {
  clearProfile,
  getInitials,
  ROLE_LABELS,
  useProfile,
} from '../features/profile/profileStorage';
import { MOISTURE_API_BASE_URL } from '../services/moistureApi';
import { SMOKEHOUSE_API_BASE_URL } from '../services/smokehouseApi';

type ServiceState = 'checking' | 'online' | 'offline';

async function ping(url: string): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export default function Settings() {
  const { profile, preferences, reload, updatePreferences } = useProfile();
  const [mainApi, setMainApi] = useState<ServiceState>('checking');
  const [acousticApi, setAcousticApi] = useState<ServiceState>('checking');

  const checkServices = useCallback(() => {
    setMainApi('checking');
    setAcousticApi('checking');
    void ping(`${SMOKEHOUSE_API_BASE_URL}/health`).then((ok) => setMainApi(ok ? 'online' : 'offline'));
    void ping(`${MOISTURE_API_BASE_URL}/api/v1/thud-test/health`).then((ok) =>
      setAcousticApi(ok ? 'online' : 'offline'),
    );
  }, []);

  useFocusEffect(
    useCallback(() => {
      void reload();
      checkServices();
    }, [reload, checkServices]),
  );

  const confirmReset = () => {
    Alert.alert(
      'Reset account?',
      'This removes your profile and preferences from this device. Your saved predictions on the server are not affected.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            void clearProfile().then(() => {
              if (router.canDismiss()) router.dismissAll();
              router.replace('/');
            });
          },
        },
      ],
    );
  };

  const version = Constants.expoConfig?.version ?? '1.0.0';

  return (
    <View style={styles.root}>
      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        <View style={styles.topBar}>
          <Pressable accessibilityRole="button" hitSlop={10} onPress={() => router.back()}>
            <Text style={styles.back}>‹ Back</Text>
          </Pressable>
          <Text style={styles.screenTitle}>Settings</Text>
          <View style={styles.topBarSpacer} />
        </View>

        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
          {/* Profile */}
          <View style={styles.profileCard}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{getInitials(profile?.fullName ?? '')}</Text>
            </View>
            <View style={styles.flex}>
              <Text style={styles.profileName} numberOfLines={1}>
                {profile?.fullName ?? 'Guest user'}
              </Text>
              <Text style={styles.profileMeta} numberOfLines={1}>
                {profile ? ROLE_LABELS[profile.role] : 'No account created yet'}
              </Text>
              {profile ? (
                <Text style={styles.profileMeta} numberOfLines={1}>
                  {profile.estateName} · {profile.district}
                </Text>
              ) : null}
            </View>
          </View>

          <Section title="Account">
            <Row
              label={profile ? 'Edit profile' : 'Create account'}
              hint={profile ? 'Name, role, estate and district' : 'Personalise your Home screen'}
              onPress={() =>
                router.push(profile ? { pathname: '/onboarding', params: { mode: 'edit' } } : '/onboarding')
              }
            />
            {profile ? (
              <Row
                label="Member since"
                value={new Date(profile.createdAt).toLocaleDateString(undefined, {
                  day: 'numeric',
                  month: 'short',
                  year: 'numeric',
                })}
                last
              />
            ) : null}
          </Section>

          <Section title="Home screen">
            <ToggleRow
              label="Auto-play slideshow"
              hint="Move through the forecast and tool slides automatically"
              value={preferences.autoPlaySlides}
              onChange={(value) => void updatePreferences({ autoPlaySlides: value })}
            />
            <ToggleRow
              label="Show latest predictions"
              hint="Recent result from each tool on Home"
              value={preferences.showLatestPredictions}
              onChange={(value) => void updatePreferences({ showLatestPredictions: value })}
              last
            />
          </Section>

          <Section
            title="Connection"
            action={
              <Pressable hitSlop={8} onPress={checkServices}>
                <Text style={styles.sectionAction}>Check again</Text>
              </Pressable>
            }
          >
            <ServiceRow
              label="Main server"
              hint="Plant advisory, latex & smokehouse"
              url={SMOKEHOUSE_API_BASE_URL}
              state={mainApi}
            />
            <ServiceRow
              label="Acoustic service"
              hint="Moisture / thud-test analysis"
              url={MOISTURE_API_BASE_URL}
              state={acousticApi}
              last
            />
          </Section>

          <Section title="Smart tools">
            {COMPONENTS.map((info, i) => (
              <Row
                key={info.key}
                label={`${info.number} · ${info.shortTitle}`}
                hint={info.subtitle}
                onPress={() => router.push(info.route)}
                last={i === COMPONENTS.length - 1}
              />
            ))}
          </Section>

          <Section title="About">
            <Row label="App" value="RubberSense" />
            <Row label="Version" value={version} />
            <Row label="Data privacy" value="Profile stored on device" last />
          </Section>

          <Pressable
            accessibilityRole="button"
            onPress={confirmReset}
            style={({ pressed }) => [styles.resetButton, pressed && styles.resetPressed]}
          >
            <Text style={styles.resetLabel}>Reset account</Text>
          </Pressable>

          <View style={styles.footerBrand}>
            <BrandMark size={36} />
            <Text style={styles.footerText}>RubberSense · Sense. Predict. Improve.</Text>
          </View>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>{title}</Text>
        {action}
      </View>
      <View style={styles.sectionCard}>{children}</View>
    </View>
  );
}

function Row({
  label,
  hint,
  value,
  onPress,
  last,
}: {
  label: string;
  hint?: string;
  value?: string;
  onPress?: () => void;
  last?: boolean;
}) {
  const content = (
    <>
      <View style={styles.flex}>
        <Text style={styles.rowLabel}>{label}</Text>
        {hint ? <Text style={styles.rowHint}>{hint}</Text> : null}
      </View>
      {value ? (
        <Text style={styles.rowValue} numberOfLines={1}>
          {value}
        </Text>
      ) : null}
      {onPress ? <ChevronIcon size={20} tone={colors.textSecondary} /> : null}
    </>
  );

  if (!onPress) {
    return <View style={[styles.row, !last && styles.rowDivider]}>{content}</View>;
  }

  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.row, !last && styles.rowDivider, pressed && styles.rowPressed]}
    >
      {content}
    </Pressable>
  );
}

function ToggleRow({
  label,
  hint,
  value,
  onChange,
  last,
}: {
  label: string;
  hint: string;
  value: boolean;
  onChange: (value: boolean) => void;
  last?: boolean;
}) {
  return (
    <View style={[styles.row, !last && styles.rowDivider]}>
      <View style={styles.flex}>
        <Text style={styles.rowLabel}>{label}</Text>
        <Text style={styles.rowHint}>{hint}</Text>
      </View>
      <Switch
        value={value}
        onValueChange={onChange}
        trackColor={{ false: colors.border, true: colors.primaryLight }}
        thumbColor={value ? colors.primary : '#F2F2F2'}
      />
    </View>
  );
}

function ServiceRow({
  label,
  hint,
  url,
  state,
  last,
}: {
  label: string;
  hint: string;
  url: string;
  state: ServiceState;
  last?: boolean;
}) {
  const tone =
    state === 'online' ? colors.success : state === 'offline' ? colors.critical : colors.warning;
  const text = state === 'online' ? 'Online' : state === 'offline' ? 'Offline' : 'Checking…';

  return (
    <View style={[styles.row, !last && styles.rowDivider]}>
      <View style={styles.flex}>
        <Text style={styles.rowLabel}>{label}</Text>
        <Text style={styles.rowHint}>{hint}</Text>
        <Text style={styles.rowUrl}>{url}</Text>
      </View>
      <View style={styles.statusPill}>
        <StatusDot tone={tone} />
        <Text style={[styles.statusText, { color: tone }]}>{text}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  safeArea: {
    flex: 1,
  },
  flex: {
    flex: 1,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  back: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.primary,
    minWidth: 60,
  },
  screenTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  topBarSpacer: {
    minWidth: 60,
  },
  scroll: {
    paddingHorizontal: spacing.lg - 4,
    paddingBottom: spacing.xl,
  },
  profileCard: {
    marginTop: spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.md,
    borderRadius: radii.lg,
    backgroundColor: colors.primary,
  },
  avatar: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 20,
    fontWeight: '800',
    color: colors.primary,
  },
  profileName: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textOnPrimary,
  },
  profileMeta: {
    marginTop: 2,
    fontSize: 13,
    color: '#CDE8D6',
  },
  section: {
    marginTop: spacing.lg,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.xs,
  },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
    color: colors.textSecondary,
    textTransform: 'uppercase',
  },
  sectionAction: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.primary,
  },
  sectionCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md + 2,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm + 4,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 6,
  },
  rowDivider: {
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  rowPressed: {
    backgroundColor: '#EEF6F0',
  },
  rowLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  rowHint: {
    marginTop: 2,
    fontSize: 12,
    color: colors.textSecondary,
  },
  rowUrl: {
    marginTop: 2,
    fontSize: 11,
    color: colors.primaryLight,
  },
  rowValue: {
    maxWidth: '50%',
    fontSize: 13,
    color: colors.textSecondary,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radii.pill,
    backgroundColor: colors.background,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '700',
  },
  resetButton: {
    marginTop: spacing.lg,
    minHeight: 50,
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.critical,
    alignItems: 'center',
    justifyContent: 'center',
  },
  resetPressed: {
    backgroundColor: '#FBECEA',
  },
  resetLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.critical,
  },
  footerBrand: {
    marginTop: spacing.xl,
    alignItems: 'center',
    gap: spacing.sm,
  },
  footerText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
});
