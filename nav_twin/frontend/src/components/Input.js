
import React from 'react';
import { View, Text, TextInput } from 'react-native';
export default function Input({ label, value, onChangeText, placeholder, secureTextEntry, keyboardType }){
  return (
    <View style={{ marginBottom: 12 }}>
      <Text style={{ color: '#6b7280', marginBottom: 6 }}>{label}</Text>
      <TextInput value={value} onChangeText={onChangeText} placeholder={placeholder} secureTextEntry={secureTextEntry} keyboardType={keyboardType}
        style={{ borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10, backgroundColor: 'white' }} />
    </View>
  );
}
