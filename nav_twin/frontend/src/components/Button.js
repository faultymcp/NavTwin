
import React from 'react';
import { TouchableOpacity, Text } from 'react-native';
export default function Button({ title, onPress, variant='primary', disabled }){
  const style = {
    primary: { backgroundColor: '#4f46e5', color: 'white' },
    ghost: { backgroundColor: 'white', borderWidth: 1, borderColor: '#ddd', color: '#111' },
    success: { backgroundColor: '#10b981', color: 'white' },
  }[variant];
  return (
    <TouchableOpacity onPress={onPress} disabled={disabled} style={{ borderRadius: 12, paddingVertical: 12, paddingHorizontal: 16, alignSelf: 'flex-start', backgroundColor: style.backgroundColor, borderWidth: style.borderWidth || 0, borderColor: style.borderColor || 'transparent', opacity: disabled ? 0.6 : 1 }}>
      <Text style={{ color: style.color, fontWeight: '600' }}>{title}</Text>
    </TouchableOpacity>
  );
}
