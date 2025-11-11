
import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
export default function EmojiScale({ value, onChange, caption, emojis=['😌','🙂','😐','😣','😖'] }){
  return (
    <View style={{ marginBottom: 10 }}>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
        <Text style={{ color: '#6b7280' }}>{caption}</Text>
        <Text style={{ color: '#6b7280' }}>{value}/5</Text>
      </View>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
        {emojis.map((e, idx)=>{
          const score = idx+1;
          const active = value===score;
          return (
            <TouchableOpacity key={idx} onPress={()=>onChange(score)} style={{ padding: 12, borderRadius: 12, borderWidth: 1, borderColor: '#e5e7eb', backgroundColor: active ? '#4f46e5' : 'white' }}>
              <Text style={{ fontSize: 22, color: active ? 'white' : '#111' }}>{e}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}
