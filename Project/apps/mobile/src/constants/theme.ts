import { Platform, type ViewStyle } from 'react-native';

export const colors = {
  background: '#F6FAF4',
  surface: '#FFFFFF',
  primary: '#1F5D3C',
  primaryDark: '#143E28',
  primaryLight: '#8FC2A0',
  accent: '#C98A4B',
  textPrimary: '#122018',
  textSecondary: '#5B6B61',
  textOnPrimary: '#FFFFFF',
  border: '#DCE8DF',
  success: '#2D9D5F',
  warning: '#E0A530',
  critical: '#C0392B',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const radii = {
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  pill: 999,
} as const;

export const typography = {
  display: {
    fontSize: 34,
    fontWeight: '700' as const,
    letterSpacing: 0.2,
  },
  title: {
    fontSize: 22,
    fontWeight: '700' as const,
  },
  tagline: {
    fontSize: 14,
    fontWeight: '700' as const,
    letterSpacing: 2.5,
  },
  body: {
    fontSize: 16,
    fontWeight: '400' as const,
  },
  caption: {
    fontSize: 12,
    fontWeight: '500' as const,
    letterSpacing: 1,
  },
  button: {
    fontSize: 17,
    fontWeight: '700' as const,
    letterSpacing: 0.3,
  },
} as const;

export const shadows = {
  soft: Platform.select<ViewStyle>({
    ios: {
      shadowColor: colors.primaryDark,
      shadowOpacity: 0.12,
      shadowRadius: 12,
      shadowOffset: { width: 0, height: 6 },
    },
    android: {
      elevation: 4,
    },
    default: {},
  }) as ViewStyle,
} as const;

export const theme = {
  colors,
  spacing,
  radii,
  typography,
  shadows,
} as const;

export type Theme = typeof theme;
