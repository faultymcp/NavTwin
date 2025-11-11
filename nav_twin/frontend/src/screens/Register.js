import React, { useState } from 'react';
import { View, Text, Alert, ScrollView } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card';
import Button from '../components/Button';
import Input from '../components/Input';
import { api } from '../api';

export default function Register({ navigation }) {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);

  async function signUp() {
    try {
      if (!email || !username || !password || !confirmPassword) {
        Alert.alert('Missing fields', 'Please fill in all fields');
        return;
      }

      if (password !== confirmPassword) {
        Alert.alert('Password mismatch', 'Passwords do not match');
        return;
      }

      if (password.length < 6) {
        Alert.alert('Weak password', 'Password must be at least 6 characters');
        return;
      }

      setLoading(true);

      const response = await api('/api/auth/register', {
        method: 'POST',
        body: { email, username, password },
      });

      // Auto login after registration
      const tok = await api('/api/auth/login', {
        method: 'POST',
        body: { email, password },
      });

      await AsyncStorage.setItem('navtwin_auth', JSON.stringify(tok));
      
      Alert.alert(
        'Account Created! 🎉',
        'Now let\'s personalize your experience',
        [{ text: 'Continue', onPress: () => navigation.replace('Quiz') }]
      );
    } catch (e) {
      Alert.alert('Registration failed', e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <View style={{ padding: 16, paddingTop: 40 }}>
        <Card title="Create Your Account">
          <Text style={{ color: '#6b7280', marginBottom: 16, lineHeight: 20 }}>
            Join NavTwin to get personalized navigation designed for neurodivergent travelers 🧠
          </Text>

          <Input
            label="Username"
            value={username}
            onChangeText={setUsername}
            placeholder="Choose a username"
            autoCapitalize="none"
          />

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

          <Input
            label="Confirm Password"
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="••••••••"
            secureTextEntry
          />

          <Button
            title={loading ? 'Creating account…' : 'Create Account'}
            onPress={signUp}
            disabled={loading}
          />

          <View style={{ height: 8 }} />

          <Button
            title="Already have an account? Sign in"
            variant="ghost"
            onPress={() => navigation.navigate('Login')}
          />
        </Card>
      </View>
    </ScrollView>
  );
}