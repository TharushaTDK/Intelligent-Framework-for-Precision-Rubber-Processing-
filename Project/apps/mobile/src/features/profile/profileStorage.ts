import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useState } from 'react';

/**
 * Local (on-device) user profile and app preferences.
 *
 * There is no account API on the backend, so the profile lives only
 * on this phone. It personalises the Home screen and is never sent
 * to the server.
 */

export type UserRole = 'smallholder' | 'estate_manager' | 'factory_supervisor' | 'researcher';

export interface UserProfile {
  fullName: string;
  estateName: string;
  district: string;
  role: UserRole;
  createdAt: string;
}

export interface AppPreferences {
  autoPlaySlides: boolean;
  showLatestPredictions: boolean;
}

export const ROLE_LABELS: Record<UserRole, string> = {
  smallholder: 'Smallholder farmer',
  estate_manager: 'Estate manager',
  factory_supervisor: 'Factory supervisor',
  researcher: 'Researcher / Officer',
};

export const RUBBER_DISTRICTS = [
  'Kalutara',
  'Ratnapura',
  'Kegalle',
  'Colombo',
  'Gampaha',
  'Galle',
  'Matara',
  'Kurunegala',
  'Kandy',
  'Monaragala',
  'Badulla',
  'Ampara',
];

export const DEFAULT_PREFERENCES: AppPreferences = {
  autoPlaySlides: true,
  showLatestPredictions: true,
};

const PROFILE_KEY = 'rubbersense.profile.v1';
const PREFERENCES_KEY = 'rubbersense.preferences.v1';

export async function loadProfile(): Promise<UserProfile | null> {
  try {
    const raw = await AsyncStorage.getItem(PROFILE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<UserProfile>;
    if (!parsed.fullName || !parsed.estateName) {
      return null;
    }
    return parsed as UserProfile;
  } catch {
    return null;
  }
}

export async function saveProfile(profile: UserProfile): Promise<void> {
  await AsyncStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

export async function clearProfile(): Promise<void> {
  await AsyncStorage.multiRemove([PROFILE_KEY, PREFERENCES_KEY]);
}

export async function loadPreferences(): Promise<AppPreferences> {
  try {
    const raw = await AsyncStorage.getItem(PREFERENCES_KEY);
    if (!raw) {
      return DEFAULT_PREFERENCES;
    }
    return { ...DEFAULT_PREFERENCES, ...(JSON.parse(raw) as Partial<AppPreferences>) };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

export async function savePreferences(preferences: AppPreferences): Promise<void> {
  await AsyncStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences));
}

export function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return 'RS';
  }
  const first = parts[0][0] ?? '';
  const last = parts.length > 1 ? parts[parts.length - 1][0] ?? '' : '';
  return (first + last).toUpperCase();
}

/** Loads the profile and preferences; call `reload` when the screen regains focus. */
export function useProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [preferences, setPreferences] = useState<AppPreferences>(DEFAULT_PREFERENCES);
  const [loaded, setLoaded] = useState(false);

  const reload = useCallback(async () => {
    const [nextProfile, nextPreferences] = await Promise.all([loadProfile(), loadPreferences()]);
    setProfile(nextProfile);
    setPreferences(nextPreferences);
    setLoaded(true);
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const updatePreferences = useCallback(async (patch: Partial<AppPreferences>) => {
    setPreferences((current) => {
      const next = { ...current, ...patch };
      void savePreferences(next);
      return next;
    });
  }, []);

  return { profile, preferences, loaded, reload, updatePreferences };
}
