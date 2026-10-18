import { Stack } from 'expo-router';

import { SmokehouseFlowProvider } from '../../features/smokehouse/SmokehouseFlowContext';

export default function SmokehouseLayout() {
  return (
    <SmokehouseFlowProvider>
      <Stack screenOptions={{ headerShown: false }} />
    </SmokehouseFlowProvider>
  );
}
