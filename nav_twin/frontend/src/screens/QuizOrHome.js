
import React, { useEffect } from 'react';
import { View, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { api } from '../api';
export default function QuizOrHome({ navigation }){
  useEffect(()=>{(async()=>{
    try{
      const raw = await AsyncStorage.getItem('navtwin_auth'); const tok = JSON.parse(raw || '{}');
      const prof = await api('/api/profile', { token: tok?.access_token });
      if (prof?.sensory_profile && (prof.sensory_profile.crowd_sensitivity || prof.crowd_sensitivity)) navigation.replace('Home');
      else navigation.replace('Quiz');
    }catch{ navigation.replace('Login'); }
  })();}, [navigation]);
  return <View style={{flex:1,alignItems:'center',justifyContent:'center'}}><ActivityIndicator/></View>;
}
