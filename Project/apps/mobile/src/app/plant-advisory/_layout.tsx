import { Stack } from 'expo-router';

// keeps the overview underneath when the app opens straight onto a later
// screen (e.g. Expo Go restoring the last route), so Back always leads there
export const unstable_settings = {
  initialRouteName: 'index',
};

export default function PlantAdvisoryLayout() {
  return <Stack screenOptions={{ headerShown: false }} />;
}
