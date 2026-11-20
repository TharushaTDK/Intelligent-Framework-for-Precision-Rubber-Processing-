import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
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

import { BrandMark } from '../components/BrandMark';
import { FormField } from '../components/FormField';
import { PrimaryButton } from '../components/PrimaryButton';
import { colors, radii, spacing } from '../constants/theme';
import { COMPONENTS } from '../features/home/componentCatalog';
import {
  loadProfile,
  ROLE_LABELS,
  RUBBER_DISTRICTS,
  saveProfile,
  type UserRole,
} from '../features/profile/profileStorage';

const STEPS = ['About you', 'Your estate', 'All set'] as const;
const ROLES = Object.keys(ROLE_LABELS) as UserRole[];

export default function Onboarding() {
  const params = useLocalSearchParams<{ mode?: string }>();
  const editing = params.mode === 'edit';

  const [step, setStep] = useState(0);
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState<UserRole>('smallholder');
  const [estateName, setEstateName] = useState('');
  const [district, setDistrict] = useState('');
  const [createdAt, setCreatedAt] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!editing) return;
    void loadProfile().then((profile) => {
      if (!profile) return;
      setFullName(profile.fullName);
      setRole(profile.role);
      setEstateName(profile.estateName);
      setDistrict(profile.district);
      setCreatedAt(profile.createdAt);
    });
  }, [editing]);

  const validate = (current: number): boolean => {
    const next: Record<string, string> = {};
    if (current === 0 && fullName.trim().length < 2) {
      next.fullName = 'Please enter your name.';
    }
    if (current === 1) {
      if (estateName.trim().length < 2) next.estateName = 'Please enter your estate or plantation name.';
      if (!district) next.district = 'Please choose your district.';
    }
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const goNext = async () => {
    if (!validate(step)) return;
    if (step < STEPS.length - 1) {
      setStep(step + 1);
      return;
    }
    setSaving(true);
    try {
      await saveProfile({
        fullName: fullName.trim(),
        estateName: estateName.trim(),
        district,
        role,
        createdAt: createdAt ?? new Date().toISOString(),
      });
      if (editing && router.canGoBack()) {
        router.back();
      } else {
        router.replace('/home');
      }
    } finally {
      setSaving(false);
    }
  };

  const goBack = () => {
    if (step > 0) {
      setErrors({});
      setStep(step - 1);
    } else if (router.canGoBack()) {
      router.back();
    }
  };

  const finalLabel = editing ? 'Save changes' : 'Create account';

  return (
    <View style={styles.root}>
      <View style={styles.blob} />
      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        <KeyboardAvoidingView
          style={styles.flex}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <View style={styles.topBar}>
            <Pressable accessibilityRole="button" hitSlop={10} onPress={goBack}>
              <Text style={styles.back}>‹ Back</Text>
            </Pressable>
            <Text style={styles.stepCount}>
              Step {step + 1} of {STEPS.length}
            </Text>
          </View>

          <View style={styles.progressTrack}>
            {STEPS.map((label, i) => (
              <View key={label} style={[styles.progressSegment, i <= step && styles.progressActive]} />
            ))}
          </View>

          <ScrollView
            contentContainerStyle={styles.scroll}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            {step === 0 ? (
              <View style={styles.stepBody}>
                <Text style={styles.title}>{editing ? 'Edit your profile' : 'Create your account'}</Text>
                <Text style={styles.subtitle}>
                  Tell us a little about yourself so RubberSense can tailor advice to you.
                </Text>

                <FormField
                  label="Full name"
                  value={fullName}
                  onChangeText={setFullName}
                  placeholder="e.g. Nimal Perera"
                  error={errors.fullName}
                />

                <Text style={styles.fieldLabel}>I am a…</Text>
                <View style={styles.optionList}>
                  {ROLES.map((item) => (
                    <Pressable
                      key={item}
                      accessibilityRole="radio"
                      accessibilityState={{ selected: role === item }}
                      onPress={() => setRole(item)}
                      style={[styles.option, role === item && styles.optionSelected]}
                    >
                      <View style={[styles.radio, role === item && styles.radioSelected]}>
                        {role === item ? <View style={styles.radioDot} /> : null}
                      </View>
                      <Text style={[styles.optionLabel, role === item && styles.optionLabelSelected]}>
                        {ROLE_LABELS[item]}
                      </Text>
                    </Pressable>
                  ))}
                </View>
              </View>
            ) : null}

            {step === 1 ? (
              <View style={styles.stepBody}>
                <Text style={styles.title}>Your estate</Text>
                <Text style={styles.subtitle}>
                  We show your estate and district on the Home screen.
                </Text>

                <FormField
                  label="Estate / plantation name"
                  value={estateName}
                  onChangeText={setEstateName}
                  placeholder="e.g. Nimal's Estate"
                  error={errors.estateName}
                />

                <Text style={styles.fieldLabel}>District</Text>
                <View style={styles.chips}>
                  {RUBBER_DISTRICTS.map((item) => (
                    <Pressable
                      key={item}
                      accessibilityRole="radio"
                      accessibilityState={{ selected: district === item }}
                      onPress={() => setDistrict(item)}
                      style={[styles.chip, district === item && styles.chipSelected]}
                    >
                      <Text style={[styles.chipLabel, district === item && styles.chipLabelSelected]}>
                        {item}
                      </Text>
                    </Pressable>
                  ))}
                </View>
                {errors.district ? <Text style={styles.error}>{errors.district}</Text> : null}
              </View>
            ) : null}

            {step === 2 ? (
              <View style={styles.stepBody}>
                <View style={styles.summaryHero}>
                  <BrandMark size={72} />
                  <Text style={styles.title}>
                    {editing ? 'Review your changes' : `Welcome, ${fullName.trim().split(/\s+/)[0]}!`}
                  </Text>
                  <Text style={[styles.subtitle, styles.center]}>
                    Here’s your profile. Your details are stored only on this device.
                  </Text>
                </View>

                <View style={styles.summaryCard}>
                  <SummaryRow label="Name" value={fullName.trim()} />
                  <SummaryRow label="Role" value={ROLE_LABELS[role]} />
                  <SummaryRow label="Estate" value={estateName.trim()} />
                  <SummaryRow label="District" value={`${district}, Sri Lanka`} last />
                </View>

                {!editing ? (
                  <>
                    <Text style={styles.fieldLabel}>Your smart tools</Text>
                    <View style={styles.toolList}>
                      {COMPONENTS.map((info) => (
                        <View key={info.key} style={styles.toolRow}>
                          <View style={styles.toolIcon}>{info.icon(16)}</View>
                          <View style={styles.flex}>
                            <Text style={styles.toolTitle}>{info.shortTitle}</Text>
                            <Text style={styles.toolSubtitle}>{info.subtitle}</Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  </>
                ) : null}
              </View>
            ) : null}
          </ScrollView>

          <View style={styles.footer}>
            <PrimaryButton
              label={saving ? 'Saving…' : step === STEPS.length - 1 ? finalLabel : 'Continue'}
              onPress={() => void goNext()}
            />
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
}

function SummaryRow({ label, value, last }: { label: string; value: string; last?: boolean }) {
  return (
    <View style={[styles.summaryRow, !last && styles.summaryDivider]}>
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={styles.summaryValue} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
    overflow: 'hidden',
  },
  blob: {
    position: 'absolute',
    top: -110,
    right: -90,
    width: 240,
    height: 240,
    borderRadius: 120,
    backgroundColor: colors.primaryLight,
    opacity: 0.2,
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
    paddingTop: spacing.sm,
  },
  back: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.primary,
  },
  stepCount: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  progressTrack: {
    flexDirection: 'row',
    gap: 6,
    paddingHorizontal: spacing.lg,
    marginTop: spacing.md,
  },
  progressSegment: {
    flex: 1,
    height: 5,
    borderRadius: 3,
    backgroundColor: colors.border,
  },
  progressActive: {
    backgroundColor: colors.primary,
  },
  scroll: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xl,
  },
  stepBody: {
    gap: spacing.md,
  },
  title: {
    fontSize: 26,
    fontWeight: '800',
    color: colors.textPrimary,
  },
  subtitle: {
    marginTop: -spacing.sm,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textSecondary,
  },
  center: {
    textAlign: 'center',
    marginTop: 0,
  },
  fieldLabel: {
    marginTop: spacing.xs,
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  optionList: {
    gap: spacing.sm,
    marginTop: -spacing.xs,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm + 4,
    padding: spacing.md - 2,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  optionSelected: {
    borderColor: colors.primary,
    backgroundColor: '#EEF6F0',
  },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
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
  optionLabel: {
    fontSize: 15,
    color: colors.textPrimary,
  },
  optionLabelSelected: {
    fontWeight: '700',
    color: colors.primary,
  },
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginTop: -spacing.xs,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radii.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  chipSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  chipLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  chipLabelSelected: {
    color: colors.textOnPrimary,
  },
  error: {
    fontSize: 12,
    color: colors.critical,
    marginTop: -spacing.sm,
  },
  summaryHero: {
    alignItems: 'center',
    gap: spacing.sm + 4,
  },
  summaryCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.sm + 4,
    gap: spacing.md,
  },
  summaryDivider: {
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  summaryLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  summaryValue: {
    flexShrink: 1,
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  toolList: {
    gap: spacing.sm,
    marginTop: -spacing.xs,
  },
  toolRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm + 4,
    padding: spacing.sm + 4,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  toolIcon: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: '#EAF6EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  toolTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textPrimary,
  },
  toolSubtitle: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  footer: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
  },
});
