
import React from 'react';
import { View, Text } from 'react-native';
export default function Card({ title, right, children }){
  return (
    <View style={{ backgroundColor: 'white', borderRadius: 16, padding: 16, shadowColor: '#000', shadowOpacity: 0.05, shadowOffset: { width: 0, height: 2 }, shadowRadius: 6, elevation: 2, marginBottom: 12 }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <Text style={{ fontSize: 18, fontWeight: '600' }}>{title}</Text>
        {right}
      </View>
      {children}
    </View>
  );
}
