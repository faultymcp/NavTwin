import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import * as Location from 'expo-location';

export default function NavigationScreen({ route, navigation }) {
  const { journeyData, startLocation, destination } = route.params;

  const [currentMilestoneIndex, setCurrentMilestoneIndex] = useState(0);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [totalPoints, setTotalPoints] = useState(0);
  const [completedMilestones, setCompletedMilestones] = useState([]);

  useEffect(() => {
    let locationSubscription;

    async function startTracking() {
      // Request permission
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location permission is required for navigation');
        return;
      }

      // Start watching location
      locationSubscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          distanceInterval: 10, // Update every 10 meters
          timeInterval: 5000, // Update every 5 seconds
        },
        (location) => {
          setCurrentLocation({
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
          });
        }
      );
    }

    startTracking();

    return () => {
      if (locationSubscription) {
        locationSubscription.remove();
      }
    };
  }, []);

  const currentMilestone = journeyData?.milestones?.[currentMilestoneIndex];
  const totalMilestones = journeyData?.milestones?.length || 0;
  const progressPercentage = totalMilestones > 0 
    ? ((currentMilestoneIndex / totalMilestones) * 100).toFixed(0)
    : 0;

  const handleCompleteMilestone = () => {
    if (!currentMilestone) return;

    // Award points
    const points = currentMilestone.gamification?.points || 10;
    setTotalPoints(totalPoints + points);
    setCompletedMilestones([...completedMilestones, currentMilestone]);

    // Show reward message
    Alert.alert(
      '🎉 Milestone Completed!',
      currentMilestone.reward_message || 'Great job! Keep going!',
      [
        {
          text: 'Continue',
          onPress: () => {
            if (currentMilestoneIndex < totalMilestones - 1) {
              setCurrentMilestoneIndex(currentMilestoneIndex + 1);
            } else {
              // Journey complete!
              handleJourneyComplete();
            }
          },
        },
      ]
    );
  };

  const handleJourneyComplete = () => {
    Alert.alert(
      '🏆 Journey Complete!',
      `Congratulations! You've completed your journey!\n\nTotal Points: ${totalPoints}\nMilestones: ${completedMilestones.length}/${totalMilestones}`,
      [
        {
          text: 'Finish',
          onPress: () => navigation.navigate('LocationSearch'),
        },
      ]
    );
  };

  const handleCancelJourney = () => {
    Alert.alert(
      'Cancel Journey?',
      'Are you sure you want to cancel this journey?',
      [
        { text: 'No', style: 'cancel' },
        {
          text: 'Yes',
          style: 'destructive',
          onPress: () => navigation.goBack(),
        },
      ]
    );
  };

  if (!journeyData || !journeyData.milestones) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Error loading journey data</Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleCancelJourney}>
          <Text style={styles.cancelButton}>✕ Cancel</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Navigation</Text>
        <Text style={styles.pointsBadge}>⭐ {totalPoints}</Text>
      </View>

      {/* Progress Bar */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          <View
            style={[styles.progressFill, { width: `${progressPercentage}%` }]}
          />
        </View>
        <Text style={styles.progressText}>
          {currentMilestoneIndex + 1} of {totalMilestones} milestones
        </Text>
      </View>

      {/* Journey Info Card */}
      <View style={styles.journeyInfoCard}>
        {/* Personalization Banner */}
        <View style={styles.personalizationBanner}>
          <Text style={styles.personalizationTitle}>
            🧠 AI-Personalized Journey • {journeyData.transport_mode === 'transit' ? '🚌 Public Transport' : journeyData.transport_mode === 'walking' ? '🚶 Walking' : '🚗 Driving'}
          </Text>
          <Text style={styles.personalizationText}>
            Optimized for your sensory profile • {totalMilestones} anxiety-appropriate milestones
          </Text>
        </View>
        
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>From:</Text>
          <Text style={styles.infoValue}>{startLocation.address}</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>To:</Text>
          <Text style={styles.infoValue}>{destination.address}</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Distance:</Text>
          <Text style={styles.infoValue}>
            {(journeyData.total_distance_meters / 1000).toFixed(2)} km
          </Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Difficulty:</Text>
          <Text style={styles.infoValue}>{journeyData.difficulty_rating}</Text>
        </View>
      </View>

      {/* Current Milestone */}
      <ScrollView style={styles.milestoneContainer}>
        <View style={styles.currentMilestoneCard}>
          <Text style={styles.milestoneNumber}>
            Milestone {currentMilestoneIndex + 1}
          </Text>
          <Text style={styles.milestoneInstruction}>
            {cleanHTMLInstruction(currentMilestone?.instruction || 'Continue ahead')}
          </Text>
          <View style={styles.milestoneDetails}>
            <Text style={styles.milestoneDetail}>
              📍 {currentMilestone?.distance_meters || 0}m
            </Text>
            <Text style={styles.milestoneDetail}>
              🎯 {currentMilestone?.gamification?.points || 10} points
            </Text>
          </View>
        </View>

        {/* Upcoming Milestones */}
        {currentMilestoneIndex < totalMilestones - 1 && (
          <View style={styles.upcomingContainer}>
            <Text style={styles.upcomingTitle}>Coming Up:</Text>
            {journeyData.milestones
              .slice(currentMilestoneIndex + 1, currentMilestoneIndex + 3)
              .map((milestone, index) => (
                <View key={index} style={styles.upcomingMilestone}>
                  <Text style={styles.upcomingText}>
                    {index + 2}. {cleanHTMLInstruction(milestone.instruction)}
                  </Text>
                </View>
              ))}
          </View>
        )}
      </ScrollView>

      {/* Complete Milestone Button */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.completeMilestoneButton}
          onPress={handleCompleteMilestone}
        >
          <Text style={styles.completeMilestoneButtonText}>
            ✓ Complete Milestone
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// Helper function to clean HTML from instructions
function cleanHTMLInstruction(htmlString) {
  if (!htmlString) return '';
  return htmlString
    .replace(/<[^>]*>/g, '') // Remove HTML tags
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .trim();
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 50,
    backgroundColor: '#2196F3',
  },
  cancelButton: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  headerTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  pointsBadge: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  progressContainer: {
    padding: 16,
    backgroundColor: 'white',
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4CAF50',
    borderRadius: 4,
  },
  progressText: {
    marginTop: 8,
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  journeyInfoCard: {
    backgroundColor: 'white',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  personalizationBanner: {
    backgroundColor: '#e0e7ff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4f46e5',
  },
  personalizationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4338ca',
    marginBottom: 4,
  },
  personalizationText: {
    fontSize: 12,
    color: '#4338ca',
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  infoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    width: 80,
  },
  infoValue: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  milestoneContainer: {
    flex: 1,
    padding: 16,
  },
  currentMilestoneCard: {
    backgroundColor: '#2196F3',
    padding: 20,
    borderRadius: 16,
    marginBottom: 20,
    elevation: 4,
    shadowColor: '#2196F3',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  milestoneNumber: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    opacity: 0.9,
    marginBottom: 8,
  },
  milestoneInstruction: {
    fontSize: 22,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 16,
    lineHeight: 30,
  },
  milestoneDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  milestoneDetail: {
    fontSize: 14,
    color: 'white',
    fontWeight: '600',
  },
  upcomingContainer: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    elevation: 2,
  },
  upcomingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
    marginBottom: 12,
  },
  upcomingMilestone: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  upcomingText: {
    fontSize: 14,
    color: '#333',
  },
  footer: {
    padding: 16,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  completeMilestoneButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 18,
    borderRadius: 12,
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#4CAF50',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  completeMilestoneButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  errorText: {
    fontSize: 16,
    color: '#f44336',
    textAlign: 'center',
    marginTop: 100,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 16,
    margin: 20,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});