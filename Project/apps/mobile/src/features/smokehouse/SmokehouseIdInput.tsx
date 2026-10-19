import { Pressable, StyleSheet, Text, View } from 'react-native';

import { FormField } from '../../components/FormField';
import { KNOWN_SMOKEHOUSE_IDS } from '../../constants/knownSmokehouses';
import { colors, radii, spacing, typography } from '../../constants/theme';

type SmokehouseIdInputProps = {
  label?: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  error?: string;
};

export function SmokehouseIdInput({
  label = 'Smokehouse ID',
  value,
  onChangeText,
  placeholder = 'e.g. SH01',
  error,
}: SmokehouseIdInputProps) {
  const trimmedLower = value.trim().toLowerCase();

  const suggestions = trimmedLower
    ? KNOWN_SMOKEHOUSE_IDS.filter((id) => {
        const idLower = id.toLowerCase();
        return idLower.startsWith(trimmedLower) && idLower !== trimmedLower;
      })
    : [];

  return (
    <View>
      <FormField
        label={label}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        error={error}
      />
      {suggestions.length > 0 ? (
        <View style={styles.suggestionBox}>
          {suggestions.map((id, index) => (
            <Pressable
              key={id}
              accessibilityRole="button"
              onPress={() => onChangeText(id)}
              style={[
                styles.suggestionRow,
                index === suggestions.length - 1 && styles.suggestionRowLast,
              ]}
            >
              <Text style={styles.suggestionText}>{id}</Text>
            </Pressable>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  suggestionBox: {
    marginTop: spacing.xs,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    overflow: 'hidden',
  },
  suggestionRow: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  suggestionRowLast: {
    borderBottomWidth: 0,
  },
  suggestionText: {
    fontSize: typography.body.fontSize,
    fontWeight: '600',
    color: colors.textPrimary,
  },
});
