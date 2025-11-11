import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  TextInput,
  FlatList,
} from 'react-native';
import * as Location from 'expo-location';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

const GOOGLE_API_KEY = Constants?.expoConfig?.extra?.GOOGLE_MAPS_API_KEY || 'AIzaSyCcxpsmr0SaXcA0j2pDItYF8NyEwut-rm0';
const API_BASE_URL = Constants?.expoConfig?.extra?.API_BASE_URL || 'http://192.168.0.39:8000';

export default function LocationSearchScreen({ navigation, route }) {
  const [startLocation, setStartLocation] = useState(null);
  const [destination, setDestination] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentLocationLoading, setCurrentLocationLoading] = useState(false);
  const [userId, setUserId] = useState(null);
  const [transportMode, setTransportMode] = useState('transit'); // 'walking', 'transit', 'driving'

  // For autocomplete
  const [startInput, setStartInput] = useState('');
  const [destInput, setDestInput] = useState('');
  const [startSuggestions, setStartSuggestions] = useState([]);
  const [destSuggestions, setDestSuggestions] = useState([]);
  const [showStartSuggestions, setShowStartSuggestions] = useState(false);
  const [showDestSuggestions, setShowDestSuggestions] = useState(false);

  useEffect(() => {
    loadUserId();
  }, []);

  const loadUserId = async () => {
    try {
      // Get auth token from storage (saved by Login screen)
      const raw = await AsyncStorage.getItem('navtwin_auth');
      if (raw) {
        const auth = JSON.parse(raw);
        // The token response includes user_id
        if (auth?.user_id) {
          setUserId(auth.user_id);
          console.log('✅ User ID loaded:', auth.user_id);
        }
      }
    } catch (error) {
      console.error('Error loading user ID:', error);
    }
  };

  // Search places using Google Places API
  const searchPlaces = async (input, isStart) => {
    if (!input || input.length < 2) {
      if (isStart) setStartSuggestions([]);
      else setDestSuggestions([]);
      return;
    }

    try {
      const url = `https://maps.googleapis.com/maps/api/place/autocomplete/json?input=${encodeURIComponent(input)}&key=${GOOGLE_API_KEY}&components=country:gb`;
      const response = await fetch(url);
      const data = await response.json();

      if (data.status === 'OK' && data.predictions) {
        if (isStart) {
          setStartSuggestions(data.predictions);
          setShowStartSuggestions(true);
        } else {
          setDestSuggestions(data.predictions);
          setShowDestSuggestions(true);
        }
      }
    } catch (error) {
      console.error('Places search error:', error);
    }
  };

  // Get place details (coordinates)
  const getPlaceDetails = async (placeId, description, isStart) => {
    try {
      const url = `https://maps.googleapis.com/maps/api/place/details/json?place_id=${placeId}&key=${GOOGLE_API_KEY}&fields=geometry`;
      const response = await fetch(url);
      const data = await response.json();

      if (data.status === 'OK' && data.result?.geometry?.location) {
        const location = {
          address: description,
          latitude: data.result.geometry.location.lat,
          longitude: data.result.geometry.location.lng,
        };

        if (isStart) {
          setStartLocation(location);
          setStartInput(description);
          setShowStartSuggestions(false);
        } else {
          setDestination(location);
          setDestInput(description);
          setShowDestSuggestions(false);
        }
      }
    } catch (error) {
      console.error('Place details error:', error);
      Alert.alert('Error', 'Could not get location details');
    }
  };

  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371000;
    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δφ = ((lat2 - lat1) * Math.PI) / 180;
    const Δλ = ((lon2 - lon1) * Math.PI) / 180;

    const a =
      Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  };

  const handleUseCurrentLocation = async () => {
    setCurrentLocationLoading(true);

    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required.');
        setCurrentLocationLoading(false);
        return;
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });

      setStartLocation({
        address: 'Current Location',
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      });
      setStartInput('Current Location');

      Alert.alert('Success', 'Using your current location');
    } catch (error) {
      console.error('Location error:', error);
      Alert.alert('Location Error', 'Could not get your location');
    } finally {
      setCurrentLocationLoading(false);
    }
  };

  const handleStartJourney = async () => {
    if (!startLocation || !destination) {
      Alert.alert('Missing Information', 'Please select both start and destination.');
      return;
    }

    const distance = calculateDistance(
      startLocation.latitude,
      startLocation.longitude,
      destination.latitude,
      destination.longitude
    );

    if (distance > 2000) {
      Alert.alert(
        'Long Journey',
        `This journey is ${(distance / 1000).toFixed(1)}km. Shorter journeys work better.\n\nContinue anyway?`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Continue', onPress: () => planJourney() },
        ]
      );
    } else {
      planJourney();
    }
  };

  const planJourney = async () => {
    setLoading(true);

    try {
      // Get auth token (saved by Login screen)
      const raw = await AsyncStorage.getItem('navtwin_auth');
      const auth = JSON.parse(raw || '{}');
      
      console.log('🚀 Planning journey with user:', userId);
      console.log('🌐 Calling backend at:', API_BASE_URL);

      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

      const response = await fetch(`${API_BASE_URL}/api/journeys/plan`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${auth?.access_token}`
        },
        body: JSON.stringify({
          start_latitude: startLocation.latitude,
          start_longitude: startLocation.longitude,
          destination_latitude: destination.latitude,
          destination_longitude: destination.longitude,
          user_id: userId,
          start_address: startLocation.address,
          destination_address: destination.address,
          transport_mode: transportMode, // Send user's choice!
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json();
      console.log('✅ Journey response:', data.status);

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to plan journey');
      }

      if (data.status === 'success') {
        setLoading(false);
        
        // Build personalization message
        let message = `Optimized for ${data.user_anxiety_level} anxiety\n• ${data.journey.total_milestones} manageable milestones\n• Distance: ${data.distance_km}km`;
        
        // Add note if journey is long
        if (data.is_long_journey) {
          message += '\n\n⚠️ This is a longer journey. Take breaks if needed!';
        }
        
        // Show personalization info before navigating
        Alert.alert(
          '🧠 Journey Personalized!',
          message,
          [
            {
              text: 'Start Navigation',
              onPress: () => navigation.navigate('Navigation', {
                journeyData: data.journey,
                startLocation: startLocation,
                destination: destination,
              })
            }
          ]
        );
      }
    } catch (error) {
      setLoading(false);
      
      if (error.name === 'AbortError') {
        console.error('⏰ Request timeout');
        Alert.alert(
          'Connection Timeout',
          `Backend not responding. Is it running?\n\nBackend URL: ${API_BASE_URL}\n\nCheck:\n1. Backend is running (python main.py)\n2. Phone is on same WiFi\n3. URL is correct in app.json`
        );
      } else {
        console.error('❌ Journey planning error:', error);
        Alert.alert('Planning Failed', error.message || 'Unable to plan journey');
      }
    }
  };

  return (
    <ScrollView
      style={styles.container}
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={styles.scrollContent}
    >
      <View style={styles.header}>
        <Text style={styles.title}>Plan Your Journey</Text>
        <Text style={styles.subtitle}>Let's find the best route 🗺️</Text>
      </View>

      {/* Starting Point */}
      <View style={styles.searchContainer}>
        <Text style={styles.label}>📍 Starting Point</Text>
        <View style={styles.inputRow}>
          <TextInput
            style={styles.textInput}
            placeholder="Where are you now?"
            value={startInput}
            onChangeText={(text) => {
              setStartInput(text);
              searchPlaces(text, true);
            }}
            onFocus={() => setShowStartSuggestions(true)}
          />
          {currentLocationLoading ? (
            <ActivityIndicator size="small" color="#2196F3" style={styles.locationIcon} />
          ) : (
            <TouchableOpacity
              style={styles.locationIcon}
              onPress={handleUseCurrentLocation}
            >
              <Text style={styles.locationIconText}>📍</Text>
            </TouchableOpacity>
          )}
        </View>

        {showStartSuggestions && startSuggestions.length > 0 && (
          <View style={styles.suggestionsContainer}>
            {startSuggestions.map((item) => (
              <TouchableOpacity
                key={item.place_id}
                style={styles.suggestionItem}
                onPress={() => getPlaceDetails(item.place_id, item.description, true)}
              >
                <Text style={styles.suggestionText}>{item.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {startLocation && (
          <View style={styles.selectedContainer}>
            <Text style={styles.selectedText}>✓ {startLocation.address}</Text>
          </View>
        )}
      </View>

      {/* Destination */}
      <View style={styles.searchContainer}>
        <Text style={styles.label}>🎯 Destination</Text>
        <TextInput
          style={styles.textInput}
          placeholder="Where do you want to go?"
          value={destInput}
          onChangeText={(text) => {
            setDestInput(text);
            searchPlaces(text, false);
          }}
          onFocus={() => setShowDestSuggestions(true)}
        />

        {showDestSuggestions && destSuggestions.length > 0 && (
          <View style={styles.suggestionsContainer}>
            {destSuggestions.map((item) => (
              <TouchableOpacity
                key={item.place_id}
                style={styles.suggestionItem}
                onPress={() => getPlaceDetails(item.place_id, item.description, false)}
              >
                <Text style={styles.suggestionText}>{item.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {destination && (
          <View style={styles.selectedContainer}>
            <Text style={styles.selectedText}>✓ {destination.address}</Text>
          </View>
        )}
      </View>

      {/* Transport Mode Selector */}
      <View style={styles.modeContainer}>
        <Text style={styles.label}>🚌 Choose Transport Mode</Text>
        <View style={styles.modeButtons}>
          <TouchableOpacity
            style={[
              styles.modeButton,
              transportMode === 'walking' && styles.modeButtonActive
            ]}
            onPress={() => setTransportMode('walking')}
          >
            <Text style={[
              styles.modeButtonText,
              transportMode === 'walking' && styles.modeButtonTextActive
            ]}>
              🚶 Walking
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.modeButton,
              transportMode === 'transit' && styles.modeButtonActive
            ]}
            onPress={() => setTransportMode('transit')}
          >
            <Text style={[
              styles.modeButtonText,
              transportMode === 'transit' && styles.modeButtonTextActive
            ]}>
              🚌 Public Transport
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.modeButton,
              transportMode === 'driving' && styles.modeButtonActive
            ]}
            onPress={() => setTransportMode('driving')}
          >
            <Text style={[
              styles.modeButtonText,
              transportMode === 'driving' && styles.modeButtonTextActive
            ]}>
              🚗 Driving
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Distance Info */}
      {startLocation && destination && (
        <View style={styles.infoCard}>
          <Text style={styles.infoText}>
            📏 Distance:{' '}
            {(
              calculateDistance(
                startLocation.latitude,
                startLocation.longitude,
                destination.latitude,
                destination.longitude
              ) / 1000
            ).toFixed(2)}{' '}
            km
          </Text>
        </View>
      )}

      {/* Start Button */}
      <TouchableOpacity
        style={[
          styles.startButton,
          (!startLocation || !destination || loading) && styles.startButtonDisabled,
        ]}
        onPress={handleStartJourney}
        disabled={!startLocation || !destination || loading}
      >
        {loading ? (
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <ActivityIndicator size="small" color="white" />
            <Text style={[styles.startButtonText, { marginLeft: 10 }]}>
              Planning your journey...
            </Text>
          </View>
        ) : (
          <Text style={styles.startButtonText}>Start Navigation</Text>
        )}
      </TouchableOpacity>

      {/* Tips */}
      <View style={styles.tipsContainer}>
        <Text style={styles.tipsTitle}>💡 Tips:</Text>
        <Text style={styles.tipText}>• Shorter journeys work best</Text>
        <Text style={styles.tipText}>• We'll break it into milestones</Text>
        <Text style={styles.tipText}>• Earn rewards as you go! 🎉</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 30,
    marginTop: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  searchContainer: {
    marginBottom: 25,
  },
  modeContainer: {
    marginBottom: 25,
  },
  modeButtons: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  modeButton: {
    flex: 1,
    minWidth: 100,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    backgroundColor: 'white',
    alignItems: 'center',
  },
  modeButtonActive: {
    backgroundColor: '#2196F3',
    borderColor: '#2196F3',
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  modeButtonTextActive: {
    color: 'white',
  },
  label: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    height: 56,
    borderRadius: 12,
    paddingHorizontal: 16,
    backgroundColor: 'white',
    fontSize: 16,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    color: '#333',
  },
  locationIcon: {
    width: 50,
    height: 56,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  locationIconText: {
    fontSize: 24,
  },
  suggestionsContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    marginTop: 8,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    maxHeight: 200,
  },
  suggestionItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  suggestionText: {
    fontSize: 15,
    color: '#333',
  },
  selectedContainer: {
    marginTop: 10,
    padding: 12,
    backgroundColor: '#e8f5e9',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  selectedText: {
    color: '#2e7d32',
    fontSize: 14,
    fontWeight: '500',
  },
  infoCard: {
    backgroundColor: '#fff3e0',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#ff9800',
  },
  infoText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#e65100',
  },
  startButton: {
    backgroundColor: '#2196F3',
    paddingVertical: 18,
    borderRadius: 14,
    alignItems: 'center',
    marginTop: 10,
    elevation: 4,
    shadowColor: '#2196F3',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  startButtonDisabled: {
    backgroundColor: '#bdbdbd',
    elevation: 0,
    shadowOpacity: 0,
  },
  startButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  tipsContainer: {
    marginTop: 30,
    padding: 16,
    backgroundColor: '#e3f2fd',
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#2196F3',
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1565c0',
    marginBottom: 10,
  },
  tipText: {
    fontSize: 14,
    color: '#1976d2',
    marginBottom: 6,
  },
});