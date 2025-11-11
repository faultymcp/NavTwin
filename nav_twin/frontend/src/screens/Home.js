import React from 'react';
import { View, Text, ScrollView } from 'react-native';
import Card from '../components/Card';
import Button from '../components/Button';

export default function Home({ navigation }) {
  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <View style={{ padding: 16 }}>
        {/* Welcome Card */}
        <Card title="Welcome to NavTwin! 🎯">
          <Text style={{ color: '#6b7280', marginBottom: 12, lineHeight: 20 }}>
            Your personalized navigation assistant designed for neurodivergent travelers.
          </Text>
          <Text style={{ color: '#6b7280', fontSize: 14, lineHeight: 20 }}>
            • AI-powered route planning{'\n'}
            • Gamified milestones{'\n'}
            • Anxiety-reducing guidance{'\n'}
            • Real-time stress support
          </Text>
        </Card>

        {/* Quick Actions */}
        <Card title="Quick Actions">
          <View style={{ gap: 8 }}>
            <Button
              title="🗺️ Plan a Journey"
              onPress={() => navigation.navigate('LocationSearch')}
            />
            <Button
              title="👤 View My Profile"
              variant="ghost"
              onPress={() => navigation.navigate('Profile')}
            />
            <Button
              title="⚙️ Settings"
              variant="ghost"
              onPress={() => navigation.navigate('Settings')}
            />
          </View>
        </Card>

        {/* Features Card */}
        <Card title="✨ How NavTwin Helps You">
          <FeatureItem
            icon="🎮"
            title="Gamification"
            description="Turn navigation into a game! Earn points and badges for completing milestones."
          />
          <FeatureItem
            icon="🧠"
            title="Personalization"
            description="Routes tailored to YOUR sensitivities. Less crowds, fewer transfers, calm journeys."
          />
          <FeatureItem
            icon="📍"
            title="Step-by-Step"
            description="Complex journeys broken into simple, clear instructions. No confusion."
          />
          <FeatureItem
            icon="💪"
            title="Confidence Building"
            description="Track your progress, celebrate achievements, build independence."
          />
        </Card>

        {/* Stats Preview Card */}
        <Card title="📊 Your Progress">
          <View style={{ flexDirection: 'row', justifyContent: 'space-around', paddingVertical: 8 }}>
            <StatItem label="Journeys" value="0" />
            <StatItem label="Points" value="0" />
            <StatItem label="Level" value="1" />
          </View>
          <Button
            title="View Full Profile"
            variant="ghost"
            onPress={() => navigation.navigate('Profile')}
          />
        </Card>

        {/* Getting Started */}
        <Card title="🚀 Getting Started">
          <Text style={{ color: '#6b7280', marginBottom: 12, lineHeight: 20 }}>
            1. Complete your sensory profile quiz (if you haven't){'\n'}
            2. Plan your first journey{'\n'}
            3. Follow step-by-step milestones{'\n'}
            4. Earn points and level up!
          </Text>
          <Button
            title="Plan Your First Journey"
            onPress={() => navigation.navigate('Plan')}
          />
        </Card>
      </View>
    </ScrollView>
  );
}

function FeatureItem({ icon, title, description }) {
  return (
    <View
      style={{
        flexDirection: 'row',
        marginBottom: 16,
        paddingBottom: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
      }}
    >
      <Text style={{ fontSize: 24, marginRight: 12 }}>{icon}</Text>
      <View style={{ flex: 1 }}>
        <Text style={{ fontWeight: '600', marginBottom: 4 }}>{title}</Text>
        <Text style={{ color: '#6b7280', fontSize: 14, lineHeight: 18 }}>{description}</Text>
      </View>
    </View>
  );
}

function StatItem({ label, value }) {
  return (
    <View style={{ alignItems: 'center' }}>
      <Text style={{ fontSize: 24, fontWeight: '700', color: '#4f46e5' }}>{value}</Text>
      <Text style={{ color: '#6b7280', fontSize: 12 }}>{label}</Text>
    </View>
  );
}