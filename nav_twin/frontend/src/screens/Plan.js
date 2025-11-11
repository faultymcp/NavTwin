// src/screens/Plan.js
import React, { useMemo, useRef, useState } from 'react';
import { View, Text, Alert, TouchableOpacity, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';
import { GooglePlacesAutocomplete } from 'react-native-google-places-autocomplete';

import Card from '../components/Card';
import Button from '../components/Button';
import { api } from '../api';

const GOOGLE_MAPS_API_KEY = Constants?.expoConfig?.extra?.GOOGLE_MAPS_API_KEY;

export default function Plan({ navigation }) {
  const originRef = useRef(null);
  const destRef = useRef(null);

  // Store both the text and the resolved coords
  const [origin, setOrigin] = useState({ text: '', lat: null, lng: null });
  const [destination, setDestination] = useState({ text: '', lat: null, lng: null });
  const [mode, setMode] = useState('TRANSIT'); // 'TRANSIT' | 'WALK' | 'DRIVE'
  const [loading, setLoading] = useState(false);
  const [routes, setRoutes] = useState([]);

  const countryFilter = useMemo(() => ({ components: 'country:gb' }), []);

  const onPickPlace = (setter) => (data, details) => {
    if (!details?.geometry?.location) {
      return Alert.alert('Place error', 'Could not read coordinates for that place.');
    }
    setter({
      text: data.description,
      lat: details.geometry.location.lat,
      lng: details.geometry.location.lng,
    });
  };

  async function plan() {
    try {
      if (!origin.lat || !origin.lng || !destination.lat || !destination.lng) {
        Alert.alert('Missing places', 'Please choose both origin and destination from the suggestions.');
        return;
      }
      setLoading(true);

      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');

      const data = await api('/api/routes/plan', {
        method: 'POST',
        token: tok?.access_token,
        body: {
          origin_lat: Number(origin.lat),
          origin_lng: Number(origin.lng),
          destination_lat: Number(destination.lat),
          destination_lng: Number(destination.lng),
          mode,
        },
      });

      setRoutes(Array.isArray(data?.routes) ? data.routes : []);
      if (!data?.routes?.length) {
        Alert.alert('No routes', 'No routes found for that journey. Try adjusting locations or mode.');
      }
    } catch (e) {
      Alert.alert('Route error', e?.message || 'Failed to get routes.');
    } finally {
      setLoading(false);
    }
  }

  const ModeChip = ({ label }) => (
    <TouchableOpacity
      onPress={() => setMode(label)}
      style={{
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: '#e5e7eb',
        backgroundColor: mode === label ? '#4f46e5' : 'white',
        marginRight: 8,
      }}
    >
      <Text style={{ color: mode === label ? 'white' : '#111' }}>{label}</Text>
    </TouchableOpacity>
  );

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <ScrollView contentContainerStyle={{ padding: 16 }}>
        <Card title="Plan a journey">
          {!GOOGLE_MAPS_API_KEY ? (
            <Text style={{ color: '#b91c1c', marginBottom: 12 }}>
              ⚠️ Missing Google API key. Add it to app.json → expo.extra.GOOGLE_MAPS_API_KEY.
            </Text>
          ) : null}

          <View style={{ marginBottom: 12 }}>
            <Text style={{ marginBottom: 6, fontWeight: '600' }}>Origin</Text>
            <GooglePlacesAutocomplete
              ref={originRef}
              placeholder="Search origin (e.g., University of Essex)"
              fetchDetails
              enablePoweredByContainer={false}
              debounce={200}
              minLength={2}
              onPress={onPickPlace(setOrigin)}
              query={{
                key: GOOGLE_MAPS_API_KEY,
                language: 'en',
                ...countryFilter, // limit to GB; remove if you want global
              }}
              styles={{
                textInput: { height: 44, borderColor: '#e5e7eb', borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, backgroundColor: 'white' },
                listView: { zIndex: 1000 }, // ensure suggestions overlay
              }}
            />
          </View>

          <View style={{ marginBottom: 12 }}>
            <Text style={{ marginBottom: 6, fontWeight: '600' }}>Destination</Text>
            <GooglePlacesAutocomplete
              ref={destRef}
              placeholder="Search destination (e.g., Colchester Station)"
              fetchDetails
              enablePoweredByContainer={false}
              debounce={200}
              minLength={2}
              onPress={onPickPlace(setDestination)}
              query={{
                key: GOOGLE_MAPS_API_KEY,
                language: 'en',
                ...countryFilter,
              }}
              styles={{
                textInput: { height: 44, borderColor: '#e5e7eb', borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, backgroundColor: 'white' },
                listView: { zIndex: 1000 },
              }}
            />
          </View>

          <View style={{ flexDirection: 'row', marginTop: 6, marginBottom: 12 }}>
            {['TRANSIT', 'WALK', 'DRIVE'].map((m) => (
              <ModeChip key={m} label={m} />
            ))}
          </View>

          <Button title={loading ? 'Finding…' : 'Find routes'} onPress={plan} />
        </Card>

        {routes.map((r, idx) => (
          <Card
            key={idx}
            title={`${r.name || 'Route'} • ${((r.duration_seconds || r.duration || 0) / 60).toFixed(1)} min`}
            right={
              <View style={{ flexDirection: 'row' }}>
                <View style={{ marginRight: 8 }}>
                  <Button
                    title="Map"
                    variant="ghost"
                    onPress={() => navigation.navigate('Map', { routeData: r })}
                  />
                </View>
                <Button
                  title="Start"
                  onPress={() =>
                    navigation.navigate('Journey', {
                      routeData: r,
                      origin: {
                        lat: origin.lat,
                        lng: origin.lng,
                        address: origin.text,
                      },
                      destination: {
                        lat: destination.lat,
                        lng: destination.lng,
                        address: destination.text,
                      },
                    })
                  }
                />
              </View>
            }
          >
            <Text style={{ color: '#6b7280', marginBottom: 6 }}>
              Modes: {(r.transport_modes || []).join(', ')}
            </Text>
            {(r.legs?.[0]?.steps || []).slice(0, 6).map((s, i) => (
              <Text key={i} style={{ marginBottom: 4 }}>
                • {s.html_instructions || s.travel_mode || s.name || 'Step'}
              </Text>
            ))}
          </Card>
        ))}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
