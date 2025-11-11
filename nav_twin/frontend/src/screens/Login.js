import React, { useState } from 'react';
import { View, Text, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card';
import Button from '../components/Button';
import Input from '../components/Input';
import { api } from '../api';

export default function Login({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  async function signIn() {
    try {
      if (!email || !password) {
        Alert.alert('Missing fields', 'Please enter email and password');
        return;
      }

      setLoading(true);

      // Login
      const tok = await api('/api/auth/login', {
        method: 'POST',
        body: { email, password },
      });

      await AsyncStorage.setItem('navtwin_auth', JSON.stringify(tok));

      // Check if user has completed sensory profile
      const profile = await api('/api/profile', {
        method: 'GET',
        token: tok?.access_token,
      });

      // If no sensory profile, go to quiz
      if (!profile?.sensory_profile || Object.keys(profile.sensory_profile).length === 0) {
        Alert.alert(
          'Welcome! 👋',
          'Let\'s personalize your navigation experience',
          [{ text: 'Start Quiz', onPress: () => navigation.replace('Quiz') }]
        );
      } else {
        // Profile complete, go to home
        navigation.replace('Home');
      }
    } catch (e) {
      Alert.alert('Sign in failed', e.message || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <View style={{ padding: 16, paddingTop: 60 }}>
        {/* App Logo/Header */}
        <View style={{ alignItems: 'center', marginBottom: 32 }}>
          <Text style={{ fontSize: 36, fontWeight: 'bold', color: '#2196F3', marginBottom: 8 }}>
            NavTwin
          </Text>
          <Text style={{ fontSize: 16, color: '#6b7280', textAlign: 'center' }}>
            Personalized navigation for{'\n'}neurodivergent travelers 🧠
          </Text>
        </View>

        <Card title="Welcome back">
          <Input
            label="Email"
            value={email}
            onChangeText={setEmail}
            placeholder="you@example.com"
            keyboardType="email-address"
            autoCapitalize="none"
          />

          <Input
            label="Password"
            value={password}
            onChangeText={setPassword}
            placeholder="••••••••"
            secureTextEntry
          />

          <Button
            title={loading ? 'Signing in…' : 'Sign in'}
            onPress={signIn}
            disabled={loading}
          />

          <View style={{ height: 8 }} />

          <Button
            title="Create an account"
            variant="ghost"
            onPress={() => navigation.navigate('Register')}
          />
        </Card>

        {/* Feature Highlights */}
        <View style={{ marginTop: 24, padding: 16, backgroundColor: 'white', borderRadius: 12 }}>
          <Text style={{ fontWeight: '600', marginBottom: 12, color: '#333' }}>
            ✨ What makes NavTwin special:
          </Text>
          <Text style={{ color: '#6b7280', marginBottom: 6, fontSize: 14 }}>
            🎯 Routes based on YOUR sensory needs
          </Text>
          <Text style={{ color: '#6b7280', marginBottom: 6, fontSize: 14 }}>
            🎮 Gamified milestones to reduce anxiety
          </Text>
          <Text style={{ color: '#6b7280', fontSize: 14 }}>
            💪 Build confidence with each journey
          </Text>
        </View>
      </View>
    </View>
  );
}