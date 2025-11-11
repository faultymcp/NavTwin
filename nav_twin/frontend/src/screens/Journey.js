import React, { useEffect, useState } from 'react';
import { View, Text, Alert, FlatList } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card';
import Button from '../components/Button';
import { api } from '../api';

function StepItem({ item, onComplete }) {
  return (
    <View style={{ marginBottom: 10, padding: 12, borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 12 }}>
      <Text style={{ fontWeight: '600', marginBottom: 6 }}>{item.title || 'Step'}</Text>
      <Text style={{ color: '#6b7280', marginBottom: 8 }}>{item.description || '—'}</Text>
      {!item.completed && <Button title="Mark done ✅" onPress={() => onComplete(item)} />}
      {item.completed && <Text style={{ color: '#10b981', marginTop: 4 }}>Completed</Text>}
    </View>
  );
}

export default function Journey({ route, navigation }) {
  const { routeData, origin, destination } = route.params || {};
  const [journeyId, setJourneyId] = useState(null);
  const [milestones, setMilestones] = useState([]);
  const [points, setPoints] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');

  useEffect(() => {
    startJourney();
  }, []);

  async function startJourney() {
    try {
      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');

      // FIX: Use origin/destination from Plan.js if available, otherwise extract from routeData
      let originCoords, destCoords;

      if (origin?.lat && origin?.lng) {
        // Use the origin from Plan.js (already has correct format)
        originCoords = { lat: origin.lat, lng: origin.lng };
      } else {
        // Extract from routeData and convert latitude/longitude to lat/lng
        const startLoc = routeData?.legs?.[0]?.start_location || routeData?.legs?.[0]?.startLocation;
        originCoords = {
          lat: startLoc?.latitude || startLoc?.lat || 51.5308,
          lng: startLoc?.longitude || startLoc?.lng || -0.1238
        };
      }

      if (destination?.lat && destination?.lng) {
        // Use the destination from Plan.js
        destCoords = { lat: destination.lat, lng: destination.lng };
      } else {
        // Extract from routeData
        const endLoc = routeData?.legs?.slice(-1)?.[0]?.end_location || routeData?.legs?.slice(-1)?.[0]?.endLocation;
        destCoords = {
          lat: endLoc?.latitude || endLoc?.lat || 51.5048,
          lng: endLoc?.longitude || endLoc?.lng || -0.0863
        };
      }

      console.log('🚀 Starting journey with coords:', { originCoords, destCoords });

      const res = await api('/api/journeys/start', {
        method: 'POST',
        token: tok?.access_token,
        body: {
          origin_lat: originCoords.lat,
          origin_lng: originCoords.lng,
          origin_address: origin?.name || 'Origin',
          destination_lat: destCoords.lat,
          destination_lng: destCoords.lng,
          destination_address: destination?.name || 'Destination'
        }
      });

      setJourneyId(res.journey_id);

      const gen = await api(`/api/journeys/${res.journey_id}/milestones/generate`, {
        method: 'POST',
        token: tok?.access_token,
        body: { route_data: routeData }
      });

      setMilestones((gen.milestones || []).map((m, i) => ({ ...m, _localId: i })));
    } catch (e) {
      console.error('❌ Journey start error:', e);
      Alert.alert('Start failed', e.message);
    }
  }

  async function completeMilestone(m) {
    try {
      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');
      const res = await api(`/api/journeys/${journeyId}/milestones/${m?.id || m?._localId || 0}/complete`, {
        method: 'POST',
        token: tok?.access_token,
        body: { milestone: m, all_milestones: milestones }
      });
      const awarded = res?.data?.points_awarded || 0;
      setPoints(p => p + awarded);
      setStatusMsg(`+${awarded} points`);
      setMilestones(ms => ms.map(x => x._localId === m._localId ? { ...x, completed: true } : x));
    } catch (e) {
      Alert.alert('Complete failed', e.message);
    }
  }

  async function finish() {
    try {
      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');
      await api(`/api/journeys/${journeyId}/complete`, {
        method: 'POST',
        token: tok?.access_token,
        body: { success: true, rating: 4 }
      });
      Alert.alert('🎉 Journey complete', `Great job! Total points: ${points}`);
      navigation.popToTop();
    } catch (e) {
      Alert.alert('Finish failed', e.message);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <View style={{ padding: 16 }}>
        <Card title="Journey" right={<Text style={{ color: '#10b981' }}>{statusMsg}</Text>}>
          <Text style={{ color: '#6b7280', marginBottom: 8 }}>Points: {points}</Text>
          <Button title="Finish Journey" variant="success" onPress={finish} />
        </Card>
        <Card title="Your steps">
          <FlatList
            data={milestones}
            keyExtractor={(item) => String(item._localId ?? item.id ?? Math.random())}
            renderItem={({ item }) => <StepItem item={item} onComplete={completeMilestone} />}
          />
        </Card>
      </View>
    </View>
  );
}