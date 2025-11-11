import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Alert, RefreshControl } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card';
import Button from '../components/Button';
import { api } from '../api';

export default function Profile({ navigation }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadProfile();
  }, []);

  async function loadProfile() {
    try {
      setLoading(true);
      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');

      const data = await api('/api/profile', {
        method: 'GET',
        token: tok?.access_token,
      });

      setProfile(data);
    } catch (e) {
      Alert.alert('Error', 'Failed to load profile: ' + e.message);
    } finally {
      setLoading(false);
    }
  }

  async function onRefresh() {
    setRefreshing(true);
    await loadProfile();
    setRefreshing(false);
  }

  async function logout() {
    await AsyncStorage.removeItem('navtwin_auth');
    navigation.replace('Login');
  }

  if (loading && !profile) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#f6f7fb' }}>
        <Text style={{ color: '#6b7280' }}>Loading profile...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={{ flex: 1, backgroundColor: '#f6f7fb' }}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      <View style={{ padding: 16 }}>
        {/* User Info Card */}
        <Card title="👤 Your Profile">
          <Text style={{ fontSize: 18, fontWeight: '600', marginBottom: 4 }}>
            {profile?.user?.username || 'User'}
          </Text>
          <Text style={{ color: '#6b7280', marginBottom: 12 }}>
            {profile?.user?.email || ''}
          </Text>
          
          <View style={{ flexDirection: 'row', gap: 8, marginTop: 8 }}>
            <Button
              title="Edit Quiz"
              variant="ghost"
              onPress={() => navigation.navigate('Quiz')}
            />
            <Button
              title="Logout"
              variant="ghost"
              onPress={logout}
            />
          </View>
        </Card>

        {/* Stats Card */}
        <Card title="📊 Your Stats">
          <View style={{ flexDirection: 'row', justifyContent: 'space-around' }}>
            <View style={{ alignItems: 'center' }}>
              <Text style={{ fontSize: 24, fontWeight: '700', color: '#4f46e5' }}>
                {profile?.stats?.total_journeys || 0}
              </Text>
              <Text style={{ color: '#6b7280', fontSize: 12 }}>Journeys</Text>
            </View>
            <View style={{ alignItems: 'center' }}>
              <Text style={{ fontSize: 24, fontWeight: '700', color: '#10b981' }}>
                {profile?.stats?.total_points || 0}
              </Text>
              <Text style={{ color: '#6b7280', fontSize: 12 }}>Points</Text>
            </View>
            <View style={{ alignItems: 'center' }}>
              <Text style={{ fontSize: 24, fontWeight: '700', color: '#f59e0b' }}>
                {profile?.stats?.level || 1}
              </Text>
              <Text style={{ color: '#6b7280', fontSize: 12 }}>Level</Text>
            </View>
          </View>
        </Card>

        {/* Sensory Preferences Card */}
        <Card title="🧠 Sensory Preferences">
          <Text style={{ color: '#6b7280', marginBottom: 12, fontSize: 13 }}>
            These settings help us personalize your navigation experience
          </Text>

          <SensitivityRow
            icon="👥"
            label="Crowd Sensitivity"
            value={profile?.sensory_profile?.crowd_sensitivity || 3}
          />

          <SensitivityRow
            icon="🔊"
            label="Noise Sensitivity"
            value={profile?.sensory_profile?.noise_sensitivity || 3}
          />

          <SensitivityRow
            icon="💡"
            label="Visual Sensitivity"
            value={profile?.sensory_profile?.visual_sensitivity || 3}
          />

          <SensitivityRow
            icon="✋"
            label="Touch Sensitivity"
            value={profile?.sensory_profile?.touch_sensitivity || 3}
          />

          <SensitivityRow
            icon="🔄"
            label="Transfer Anxiety"
            value={profile?.sensory_profile?.transfer_anxiety || 3}
          />

          <SensitivityRow
            icon="⏱️"
            label="Time Pressure Tolerance"
            value={profile?.sensory_profile?.time_pressure_tolerance || 3}
          />

          <SensitivityRow
            icon="🗺️"
            label="Familiarity Preference"
            value={profile?.sensory_profile?.preference_for_familiarity || 3}
          />
        </Card>

        {/* Communication Preferences Card */}
        <Card title="💬 Communication Preferences">
          <PreferenceRow
            label="Instruction Detail Level"
            value={getInstructionLevelLabel(profile?.sensory_profile?.instruction_detail_level)}
          />
          <PreferenceRow
            label="Prefers Visual Over Text"
            value={profile?.sensory_profile?.prefers_visual_over_text ? 'Yes ✓' : 'No ✗'}
          />
          <PreferenceRow
            label="Wants Voice Guidance"
            value={profile?.sensory_profile?.wants_voice_guidance ? 'Yes ✓' : 'No ✗'}
          />
        </Card>

        {/* Navigation Style Card */}
        <Card title="✨ Your Navigation Style">
          <View style={{ padding: 12, backgroundColor: '#f3f4f6', borderRadius: 8 }}>
            <Text style={{ fontSize: 16, fontWeight: '600', marginBottom: 8 }}>
              {getNavigationStyle(profile?.sensory_profile)}
            </Text>
            <Text style={{ color: '#6b7280', fontSize: 14 }}>
              {getNavigationDescription(profile?.sensory_profile)}
            </Text>
          </View>
        </Card>

        {/* Tips Card */}
        <Card title="💡 Tips">
          <Text style={{ color: '#6b7280', fontSize: 14, lineHeight: 20 }}>
            • Your preferences help us recommend the best routes for you{'\n'}
            • Routes are scored based on your sensitivities{'\n'}
            • Complete journeys to earn points and level up!{'\n'}
            • Update your preferences anytime from "Edit Quiz"
          </Text>
        </Card>
      </View>
    </ScrollView>
  );
}

// Helper Components
function SensitivityRow({ icon, label, value }) {
  function getSensitivityLabel(val) {
    if (val <= 1.5) return 'Very Low';
    if (val <= 2.5) return 'Low';
    if (val <= 3.5) return 'Moderate';
    if (val <= 4.5) return 'High';
    return 'Very High';
  }

  function getSensitivityColor(val) {
    if (val <= 2) return '#10b981';
    if (val <= 3) return '#3b82f6';
    if (val <= 4) return '#f59e0b';
    return '#ef4444';
  }

  return (
    <View
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 12,
        paddingBottom: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
      }}
    >
      <Text style={{ fontSize: 20, marginRight: 12 }}>{icon}</Text>
      <View style={{ flex: 1 }}>
        <Text style={{ fontWeight: '500', marginBottom: 4 }}>{label}</Text>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View
            style={{
              flex: 1,
              height: 8,
              backgroundColor: '#e5e7eb',
              borderRadius: 4,
              overflow: 'hidden',
              marginRight: 8,
            }}
          >
            <View
              style={{
                width: `${(value / 5) * 100}%`,
                height: '100%',
                backgroundColor: getSensitivityColor(value),
              }}
            />
          </View>
          <Text style={{ fontSize: 12, color: '#6b7280', width: 70 }}>
            {getSensitivityLabel(value)}
          </Text>
        </View>
      </View>
    </View>
  );
}

function PreferenceRow({ label, value }) {
  return (
    <View
      style={{
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
        paddingBottom: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
      }}
    >
      <Text style={{ color: '#6b7280' }}>{label}</Text>
      <Text style={{ fontWeight: '600' }}>{value}</Text>
    </View>
  );
}

function getInstructionLevelLabel(level) {
  const labels = {
    'brief': 'Brief & Quick',
    'moderate': 'Moderate Detail',
    'detailed': 'Very Detailed',
  };
  return labels[level] || level;
}

function getNavigationStyle(profile) {
  if (!profile) return 'Balanced Navigator 🎯';

  const avgSensitivity =
    (profile.crowd_sensitivity +
      profile.noise_sensitivity +
      profile.visual_sensitivity +
      profile.transfer_anxiety) /
    4;

  if (avgSensitivity >= 4.5) return 'Comfort Seeker 🛋️';
  if (avgSensitivity >= 3.5) return 'Mindful Traveler 🧘';
  if (avgSensitivity >= 2.5) return 'Balanced Navigator 🎯';
  if (avgSensitivity >= 1.5) return 'Confident Explorer 🚀';
  return 'Adventurous Spirit 🌟';
}

function getNavigationDescription(profile) {
  if (!profile) return 'We balance efficiency with comfort in your routes.';

  const avgSensitivity =
    (profile.crowd_sensitivity +
      profile.noise_sensitivity +
      profile.visual_sensitivity +
      profile.transfer_anxiety) /
    4;

  if (avgSensitivity >= 4.5)
    return 'You prefer quiet, calm, and direct routes. We prioritize your comfort above all else.';
  if (avgSensitivity >= 3.5)
    return 'You value comfort while staying efficient. We find the sweet spot for you.';
  if (avgSensitivity >= 2.5)
    return 'We balance efficiency with comfort in your routes.';
  if (avgSensitivity >= 1.5)
    return 'You handle challenges well. We optimize for speed while respecting your preferences.';
  return 'You thrive on variety! We show you the fastest routes with confidence.';
}