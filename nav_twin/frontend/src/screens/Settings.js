
import React, { useEffect, useState } from 'react';
import { View, Text, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card'; import Button from '../components/Button'; import Input from '../components/Input'; import EmojiScale from '../components/EmojiScale'; import { api } from '../api';
export default function Settings(){
  const [profile,setProfile]=useState(null); const [loading,setLoading]=useState(false);
  useEffect(()=>{(async()=>{ try{ const raw=await AsyncStorage.getItem('navtwin_auth'); const tok=JSON.parse(raw||'{}'); const p=await api('/api/profile',{token:tok?.access_token}); const sp=p?.sensory_profile||p||{}; setProfile({ username:p?.username||'', crowd_sensitivity:sp.crowd_sensitivity||3, noise_sensitivity:sp.noise_sensitivity||3, visual_sensitivity:sp.visual_sensitivity||3, transfer_anxiety:sp.transfer_anxiety||3, instruction_detail_level:sp.instruction_detail_level||'moderate', prefers_visual_over_text: sp.prefers_visual_over_text ?? true }); }catch(e){} })();},[]);
  async function save(){ try{ setLoading(true); const raw=await AsyncStorage.getItem('navtwin_auth'); const tok=JSON.parse(raw||'{}'); await api('/api/profile',{method:'PUT',token:tok?.access_token,body:profile}); Alert.alert('Saved','Profile updated'); }catch(e){ Alert.alert('Save failed', e.message);} finally{ setLoading(false);} }
  if(!profile) return <View style={{flex:1,alignItems:'center',justifyContent:'center'}}><Text>Loading…</Text></View>;
  return (<View style={{flex:1,backgroundColor:'#f6f7fb'}}><View style={{padding:16}}><Card title="Profile"><Input label="Username" value={profile.username} onChangeText={v=>setProfile({...profile,username:v})}/><EmojiScale value={profile.crowd_sensitivity} onChange={v=>setProfile({...profile,crowd_sensitivity:v})} caption="Crowd sensitivity"/><EmojiScale value={profile.noise_sensitivity} onChange={v=>setProfile({...profile,noise_sensitivity:v})} caption="Noise sensitivity"/><EmojiScale value={profile.visual_sensitivity} onChange={v=>setProfile({...profile,visual_sensitivity:v})} caption="Visual sensitivity"/><EmojiScale value={profile.transfer_anxiety} onChange={v=>setProfile({...profile,transfer_anxiety:v})} caption="Transfer anxiety"/><Button title={loading?'Saving…':'Save profile'} onPress={save}/></Card></View></View>);
}
