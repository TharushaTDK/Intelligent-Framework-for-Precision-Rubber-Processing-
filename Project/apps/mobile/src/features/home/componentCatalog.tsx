import type { ReactNode } from 'react';

import { DropletMark, LeafMark, ThermometerMark, WaveMark } from '../../components/ComponentMarks';
import { colors } from '../../constants/theme';
import type { InsightKey } from './useLatestInsights';

export type ComponentRoute = '/plant-advisory' | '/latex' | '/smokehouse' | '/moisture';

export interface ComponentInfo {
  key: InsightKey;
  number: string;
  title: string;
  shortTitle: string;
  subtitle: string;
  description: string;
  highlights: string[];
  route: ComponentRoute;
  emptyHint: string;
  palette: { from: string; to: string; ink: string };
  icon: (size: number, tone?: string) => ReactNode;
}

export const COMPONENTS: ComponentInfo[] = [
  {
    key: 'plant',
    number: '01',
    title: "How's Today Your Rubber Plant?",
    shortTitle: 'Plant Advisory',
    subtitle: 'Weather & tapping advisory',
    description:
      'Live weather for your field turns into a tapping suitability score, latex DRC estimate and an expected market price.',
    highlights: ['Tapping suitability', 'DRC estimate', 'Price outlook'],
    route: '/plant-advisory',
    emptyHint: 'Pick your field to get today’s tapping advice.',
    palette: { from: '#2E8B57', to: '#1F5D3C', ink: '#E7F5EC' },
    icon: (size, tone = colors.primary) => <LeafMark size={size} tone={tone} />,
  },
  {
    key: 'latex',
    number: '02',
    title: "Let's Set the Latex",
    shortTitle: 'Latex Coagulation',
    subtitle: 'Coagulation decision support',
    description:
      'Get the right water and formic-acid amounts for each latex batch, plus the expected coagulation time.',
    highlights: ['Water & acid dose', 'Set time', 'Batch history'],
    route: '/latex',
    emptyHint: 'Plan your first latex batch.',
    palette: { from: '#3E9C8F', to: '#1E5F5A', ink: '#E3F4F2' },
    icon: (size, tone = colors.primary) => <DropletMark size={size} tone={tone} />,
  },
  {
    key: 'smokehouse',
    number: '03',
    title: 'Smart Smokehouse',
    shortTitle: 'Smart Smokehouse',
    subtitle: 'Drying & fire-risk monitor',
    description:
      'Track each smokehouse layer, predict the remaining drying time and catch fire-risk conditions early.',
    highlights: ['Drying time', 'Fire-risk alerts', 'Layer tracking'],
    route: '/smokehouse',
    emptyHint: 'Start a batch to monitor drying.',
    palette: { from: '#C98A4B', to: '#7A4E25', ink: '#FBEFE2' },
    icon: (size, tone = colors.primary) => <ThermometerMark size={size} tone={tone} />,
  },
  {
    key: 'moisture',
    number: '04',
    title: "How's the Moisture?",
    shortTitle: 'Moisture Check',
    subtitle: 'Acoustic sheet assessment',
    description:
      'Tap a processed rubber sheet and let acoustic analysis check its quality — no lab equipment needed.',
    highlights: ['Thud-test capture', 'Quality check', 'Saved results'],
    route: '/moisture',
    emptyHint: 'Run a thud test on a sheet.',
    palette: { from: '#4F7FA8', to: '#2C4D6B', ink: '#E5EEF6' },
    icon: (size, tone = colors.primary) => <WaveMark size={size} tone={tone} />,
  },
];
