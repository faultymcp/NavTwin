import React, { useState } from 'react';
import { View, Text, ScrollView, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Card from '../components/Card';
import Button from '../components/Button';
import { api } from '../api';

export default function Quiz({ navigation }) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);

  const questions = [
    {
      id: 'crowd_sensitivity',
      question: 'How do you feel about crowded spaces?',
      description: 'Bus stations, busy streets, shopping centers',
      options: [
        { label: 'Very comfortable - I enjoy crowds', value: 1 },
        { label: 'Somewhat comfortable', value: 2 },
        { label: 'Neutral', value: 3 },
        { label: 'Uncomfortable - prefer to avoid', value: 4 },
        { label: 'Very uncomfortable - causes anxiety', value: 5 },
      ],
    },
    {
      id: 'noise_sensitivity',
      question: 'How sensitive are you to noise?',
      description: 'Traffic, construction, loud conversations',
      options: [
        { label: 'Not bothered at all', value: 1 },
        { label: 'Slightly bothered', value: 2 },
        { label: 'Moderately sensitive', value: 3 },
        { label: 'Very sensitive', value: 4 },
        { label: 'Extremely sensitive - need quiet', value: 5 },
      ],
    },
    {
      id: 'visual_sensitivity',
      question: 'How do you handle busy visual environments?',
      description: 'Bright lights, many signs, complex intersections',
      options: [
        { label: 'No issues', value: 1 },
        { label: 'Minor distraction', value: 2 },
        { label: 'Can be overwhelming', value: 3 },
        { label: 'Very overwhelming', value: 4 },
        { label: 'Extremely difficult to process', value: 5 },
      ],
    },
    {
      id: 'transfer_anxiety',
      question: 'How do you feel about changing transport modes?',
      description: 'Switching buses, changing platforms, transfers',
      options: [
        { label: 'Easy and comfortable', value: 1 },
        { label: 'Manageable', value: 2 },
        { label: 'Somewhat stressful', value: 3 },
        { label: 'Very stressful', value: 4 },
        { label: 'Causes significant anxiety', value: 5 },
      ],
    },
    {
      id: 'time_pressure_tolerance',
      question: 'How do you handle time pressure during travel?',
      description: 'Tight connections, rushing to catch transport',
      options: [
        { label: 'Comfortable with tight schedules', value: 1 },
        { label: 'Can manage okay', value: 2 },
        { label: 'Prefer more buffer time', value: 3 },
        { label: 'Need extra time to feel safe', value: 4 },
        { label: 'Time pressure causes severe anxiety', value: 5 },
      ],
    },
    {
      id: 'preference_for_familiarity',
      question: 'How important is familiarity in routes?',
      description: 'Taking the same route vs trying new ways',
      options: [
        { label: 'Love exploring new routes', value: 1 },
        { label: 'Open to variety', value: 2 },
        { label: 'Prefer familiar routes', value: 3 },
        { label: 'Strongly prefer familiar routes', value: 4 },
        { label: 'Need familiar routes for comfort', value: 5 },
      ],
    },
    {
      id: 'instruction_detail_level',
      question: 'What level of detail do you prefer in instructions?',
      description: 'Navigation guidance style',
      options: [
        { label: 'Brief - just key directions', value: 'brief' },
        { label: 'Moderate - balanced detail', value: 'moderate' },
        { label: 'Detailed - every step explained', value: 'detailed' },
      ],
    },
    {
      id: 'prefers_visual_over_text',
      question: 'Do you prefer visual maps or text instructions?',
      options: [
        { label: 'Prefer visual maps/diagrams', value: true },
        { label: 'Prefer text descriptions', value: false },
      ],
    },
    {
      id: 'wants_voice_guidance',
      question: 'Would you like voice guidance during navigation?',
      options: [
        { label: 'Yes, voice guidance helps me', value: true },
        { label: 'No, I prefer silent navigation', value: false },
      ],
    },
  ];

  const handleAnswer = (value) => {
    const newAnswers = { ...answers, [questions[currentQuestion].id]: value };
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      submitQuiz(newAnswers);
    }
  };

  const submitQuiz = async (finalAnswers) => {
    try {
      setLoading(true);

      const raw = await AsyncStorage.getItem('navtwin_auth');
      const tok = JSON.parse(raw || '{}');

      await api('/api/profile/sensory', {
        method: 'POST',
        token: tok?.access_token,
        body: finalAnswers,
      });

      Alert.alert(
        'Profile Complete! 🎉',
        'Your personalized navigation experience is ready!',
        [{ text: 'Start Exploring', onPress: () => navigation.replace('Home') }]
      );
    } catch (e) {
      Alert.alert('Error', 'Failed to save profile: ' + e.message);
      setLoading(false);
    }
  };

  const goBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const currentQ = questions[currentQuestion];
  const progress = ((currentQuestion + 1) / questions.length) * 100;

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#f6f7fb' }}>
        <Text style={{ fontSize: 18, fontWeight: '600', marginBottom: 8 }}>Personalizing your experience...</Text>
        <Text style={{ color: '#6b7280' }}>This will just take a moment ✨</Text>
      </View>
    );
  }

  return (
    <ScrollView style={{ flex: 1, backgroundColor: '#f6f7fb' }}>
      <View style={{ padding: 16, paddingTop: 40 }}>
        {/* Progress Bar */}
        <View style={{ marginBottom: 24 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 }}>
            <Text style={{ fontWeight: '600', color: '#333' }}>
              Question {currentQuestion + 1} of {questions.length}
            </Text>
            <Text style={{ color: '#6b7280' }}>{Math.round(progress)}%</Text>
          </View>
          <View style={{ height: 8, backgroundColor: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
            <View style={{ width: `${progress}%`, height: '100%', backgroundColor: '#4f46e5' }} />
          </View>
        </View>

        <Card title={currentQ.question}>
          {currentQ.description && (
            <Text style={{ color: '#6b7280', marginBottom: 20, fontSize: 14, lineHeight: 20 }}>
              {currentQ.description}
            </Text>
          )}

          {currentQ.options.map((option, index) => (
            <View key={index} style={{ marginBottom: 8 }}>
              <Button
                title={option.label}
                variant="ghost"
                onPress={() => handleAnswer(option.value)}
              />
            </View>
          ))}

          {currentQuestion > 0 && (
            <View style={{ marginTop: 16 }}>
              <Button title="← Back" variant="ghost" onPress={goBack} />
            </View>
          )}
        </Card>

        <View style={{ marginTop: 16, padding: 16, backgroundColor: '#e0e7ff', borderRadius: 12 }}>
          <Text style={{ color: '#4338ca', fontSize: 13, lineHeight: 18 }}>
            💡 Your answers help us create routes that match your comfort level and reduce anxiety during navigation.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}