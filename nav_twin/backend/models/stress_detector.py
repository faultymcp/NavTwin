"""
COMPONENT 4: Real-Time Stress Detection System
Monitors physiological and behavioral signals to detect elevated stress during journeys
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class StressReading:
    """Single stress measurement"""
    timestamp: datetime
    heart_rate: Optional[float] = None
    movement_intensity: Optional[float] = None
    location_dwelling_time: Optional[float] = None
    app_interaction_pattern: Optional[str] = None
    route_deviation: Optional[float] = None
    calculated_stress_level: float = 0.0


class StressDetectionSystem:
    """
    AI-powered stress detection using multiple signals:
    - Physiological: Heart rate, movement patterns
    - Behavioral: App usage, location dwelling, route deviations
    - Contextual: Crowd levels, noise, unexpected events
    
    Provides real-time intervention recommendations
    """
    
    def __init__(self, user_profile: Dict[str, Any], baseline_data: Optional[Dict] = None):
        self.profile = user_profile
        self.baseline_hr = baseline_data.get('baseline_hr', 75) if baseline_data else 75
        self.baseline_movement = baseline_data.get('baseline_movement', 0.5) if baseline_data else 0.5
        
        # Stress thresholds (personalized)
        self.stress_thresholds = {
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.7,
            'critical': 0.85
        }
        
        # Recent readings for trend analysis
        self.recent_readings: List[StressReading] = []
        self.max_history = 50  # Keep last 50 readings
        
        # Intervention history
        self.interventions_used = []
        
    def detect_stress(
        self,
        current_data: Dict[str, Any],
        journey_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main stress detection function
        
        Args:
            current_data: Real-time sensor/behavioral data
            journey_context: Current journey situation
            
        Returns:
            Stress analysis with level, factors, and recommendations
        """
        
        # Calculate stress components
        physiological_stress = self._calculate_physiological_stress(current_data)
        behavioral_stress = self._calculate_behavioral_stress(current_data)
        contextual_stress = self._calculate_contextual_stress(journey_context)
        
        # Combine into overall stress score (0-1)
        overall_stress = self._combine_stress_signals(
            physiological_stress,
            behavioral_stress,
            contextual_stress
        )
        
        # Detect stress level
        stress_level = self._classify_stress_level(overall_stress)
        
        # Identify contributing factors
        stress_factors = self._identify_stress_factors(
            current_data,
            journey_context,
            physiological_stress,
            behavioral_stress,
            contextual_stress
        )
        
        # Generate interventions if needed
        interventions = []
        if stress_level in ['high', 'critical']:
            interventions = self._generate_interventions(
                stress_level,
                stress_factors,
                journey_context
            )
        
        # Create stress reading
        reading = StressReading(
            timestamp=datetime.now(),
            heart_rate=current_data.get('heart_rate'),
            movement_intensity=current_data.get('movement_intensity'),
            location_dwelling_time=current_data.get('dwelling_time'),
            app_interaction_pattern=current_data.get('interaction_pattern'),
            route_deviation=current_data.get('route_deviation'),
            calculated_stress_level=overall_stress
        )
        
        self._add_reading(reading)
        
        # Detect trends
        trend = self._analyze_stress_trend()
        
        return {
            'stress_level': stress_level,
            'stress_score': round(overall_stress, 2),
            'components': {
                'physiological': round(physiological_stress, 2),
                'behavioral': round(behavioral_stress, 2),
                'contextual': round(contextual_stress, 2)
            },
            'factors': stress_factors,
            'trend': trend,
            'interventions': interventions,
            'timestamp': reading.timestamp.isoformat()
        }
    
    def _calculate_physiological_stress(self, data: Dict) -> float:
        """Calculate stress from physiological signals"""
        
        stress_signals = []
        
        # Heart rate elevation
        if 'heart_rate' in data and data['heart_rate']:
            hr = data['heart_rate']
            hr_elevation = (hr - self.baseline_hr) / self.baseline_hr
            hr_stress = np.clip(hr_elevation / 0.3, 0, 1)  # 30% elevation = max stress
            stress_signals.append(hr_stress * 0.6)  # Weight: 60%
        
        # Movement intensity (restlessness indicator)
        if 'movement_intensity' in data and data['movement_intensity']:
            movement = data['movement_intensity']
            movement_deviation = abs(movement - self.baseline_movement) / self.baseline_movement
            movement_stress = np.clip(movement_deviation, 0, 1)
            stress_signals.append(movement_stress * 0.4)  # Weight: 40%
        
        if stress_signals:
            return np.mean(stress_signals)
        return 0.0
    
    def _calculate_behavioral_stress(self, data: Dict) -> float:
        """Calculate stress from behavioral patterns"""
        
        stress_signals = []
        
        # Location dwelling (stopping/hesitating)
        if 'dwelling_time' in data and data['dwelling_time']:
            dwelling = data['dwelling_time']  # seconds
            if dwelling > 30:  # More than 30s stationary
                dwelling_stress = min(dwelling / 120, 1.0)  # Max at 2 minutes
                stress_signals.append(dwelling_stress * 0.4)
        
        # App interaction patterns
        if 'interaction_pattern' in data:
            pattern = data['interaction_pattern']
            if pattern == 'frequent_checking':  # Checking app repeatedly
                stress_signals.append(0.7 * 0.3)
            elif pattern == 'help_seeking':  # Looking for help features
                stress_signals.append(0.8 * 0.3)
        
        # Route deviations
        if 'route_deviation' in data and data['route_deviation']:
            deviation = data['route_deviation']  # meters off route
            if deviation > 50:
                deviation_stress = min(deviation / 200, 1.0)
                stress_signals.append(deviation_stress * 0.3)
        
        if stress_signals:
            return np.mean(stress_signals)
        return 0.0
    
    def _calculate_contextual_stress(self, context: Dict) -> float:
        """Calculate stress from journey context"""
        
        stress_signals = []
        
        # Crowd levels
        crowd_level = context.get('current_crowd_level', 0)
        crowd_sensitivity = self.profile.get('crowd_sensitivity', 3.0)
        crowd_stress = (crowd_level / 5.0) * (crowd_sensitivity / 5.0)
        stress_signals.append(crowd_stress * 0.3)
        
        # Noise levels
        noise_level = context.get('current_noise_level', 0)
        noise_sensitivity = self.profile.get('noise_sensitivity', 3.0)
        noise_stress = (noise_level / 5.0) * (noise_sensitivity / 5.0)
        stress_signals.append(noise_stress * 0.25)
        
        # Transfer anxiety
        if context.get('approaching_transfer', False):
            transfer_anxiety = self.profile.get('transfer_anxiety', 3.0)
            stress_signals.append((transfer_anxiety / 5.0) * 0.25)
        
        # Time pressure
        if context.get('running_late', False):
            time_tolerance = self.profile.get('time_pressure_tolerance', 3.0)
            time_stress = 1.0 - (time_tolerance / 5.0)
            stress_signals.append(time_stress * 0.2)
        
        if stress_signals:
            return np.mean(stress_signals)
        return 0.0
    
    def _combine_stress_signals(
        self,
        physiological: float,
        behavioral: float,
        contextual: float
    ) -> float:
        """Combine stress signals with adaptive weighting"""
        
        # Base weights
        weights = {
            'physiological': 0.4,
            'behavioral': 0.35,
            'contextual': 0.25
        }
        
        # Adjust weights based on available data
        if physiological == 0:  # No physiological data
            weights['behavioral'] = 0.6
            weights['contextual'] = 0.4
        
        combined = (
            physiological * weights['physiological'] +
            behavioral * weights['behavioral'] +
            contextual * weights['contextual']
        )
        
        return np.clip(combined, 0, 1)
    
    def _classify_stress_level(self, stress_score: float) -> str:
        """Classify stress into categories"""
        
        if stress_score < self.stress_thresholds['low']:
            return 'calm'
        elif stress_score < self.stress_thresholds['moderate']:
            return 'low'
        elif stress_score < self.stress_thresholds['high']:
            return 'moderate'
        elif stress_score < self.stress_thresholds['critical']:
            return 'high'
        else:
            return 'critical'
    
    def _identify_stress_factors(
        self,
        current_data: Dict,
        context: Dict,
        phys_stress: float,
        behav_stress: float,
        context_stress: float
    ) -> List[Dict[str, Any]]:
        """Identify specific factors contributing to stress"""
        
        factors = []
        
        # Check each potential stressor
        if phys_stress > 0.5:
            if current_data.get('heart_rate', 0) > self.baseline_hr * 1.2:
                factors.append({
                    'factor': 'elevated_heart_rate',
                    'severity': 'high',
                    'description': 'Heart rate is elevated'
                })
        
        if behav_stress > 0.5:
            if current_data.get('dwelling_time', 0) > 60:
                factors.append({
                    'factor': 'hesitation',
                    'severity': 'moderate',
                    'description': 'Prolonged stopping detected'
                })
        
        if context.get('current_crowd_level', 0) > 3:
            if self.profile.get('crowd_sensitivity', 3) > 3:
                factors.append({
                    'factor': 'crowding',
                    'severity': 'high',
                    'description': 'Area is crowded'
                })
        
        if context.get('approaching_transfer'):
            if self.profile.get('transfer_anxiety', 3) > 3:
                factors.append({
                    'factor': 'transfer_anxiety',
                    'severity': 'moderate',
                    'description': 'Approaching a transfer point'
                })
        
        if context.get('running_late'):
            factors.append({
                'factor': 'time_pressure',
                'severity': 'moderate',
                'description': 'Running behind schedule'
            })
        
        return factors
    
    def _generate_interventions(
        self,
        stress_level: str,
        factors: List[Dict],
        context: Dict
    ) -> List[Dict[str, Any]]:
        """Generate personalized intervention recommendations"""
        
        interventions = []
        
        # Breathing exercises (always helpful for high stress)
        if stress_level in ['high', 'critical']:
            interventions.append({
                'type': 'breathing_exercise',
                'priority': 'high',
                'title': 'Take a moment to breathe',
                'description': 'Try 4-7-8 breathing: Inhale for 4, hold for 7, exhale for 8',
                'duration_seconds': 60
            })
        
        # Factor-specific interventions
        for factor in factors:
            if factor['factor'] == 'crowding':
                interventions.append({
                    'type': 'route_alternative',
                    'priority': 'high',
                    'title': 'Quieter route available',
                    'description': 'We found a less crowded alternative. Would you like to switch?',
                    'action': 'show_alternative_route'
                })
            
            elif factor['factor'] == 'transfer_anxiety':
                interventions.append({
                    'type': 'extra_guidance',
                    'priority': 'moderate',
                    'title': 'Transfer coming up',
                    'description': 'I\'ll guide you step-by-step through this transfer. You\'ve got this!',
                    'action': 'enable_detailed_guidance'
                })
            
            elif factor['factor'] == 'time_pressure':
                interventions.append({
                    'type': 'reassurance',
                    'priority': 'low',
                    'title': 'You\'re doing fine',
                    'description': 'Minor delays are normal. You\'re still on track.',
                    'action': 'show_updated_eta'
                })
            
            elif factor['factor'] == 'hesitation':
                interventions.append({
                    'type': 'assistance_offer',
                    'priority': 'high',
                    'title': 'Need help?',
                    'description': 'You can ask me anything. Or I can call for assistance.',
                    'action': 'open_help_dialog'
                })
        
        # Limit to top 3 interventions
        return sorted(interventions, key=lambda x: {'high': 3, 'moderate': 2, 'low': 1}[x['priority']], reverse=True)[:3]
    
    def _add_reading(self, reading: StressReading):
        """Add reading to history"""
        self.recent_readings.append(reading)
        if len(self.recent_readings) > self.max_history:
            self.recent_readings.pop(0)
    
    def _analyze_stress_trend(self) -> Dict[str, Any]:
        """Analyze stress trend over recent readings"""
        
        if len(self.recent_readings) < 5:
            return {'direction': 'stable', 'confidence': 'low'}
        
        recent_5 = [r.calculated_stress_level for r in self.recent_readings[-5:]]
        recent_10 = [r.calculated_stress_level for r in self.recent_readings[-10:]] if len(self.recent_readings) >= 10 else recent_5
        
        # Calculate trend
        avg_recent = np.mean(recent_5)
        avg_earlier = np.mean(recent_10[:len(recent_10)//2]) if len(recent_10) > 5 else avg_recent
        
        change = avg_recent - avg_earlier
        
        if abs(change) < 0.1:
            direction = 'stable'
        elif change > 0.1:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate confidence based on consistency
        std_dev = np.std(recent_5)
        confidence = 'high' if std_dev < 0.15 else 'moderate' if std_dev < 0.25 else 'low'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'change': round(change, 2)
        }
    
    def get_stress_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress report for journey"""
        
        if not self.recent_readings:
            return {'status': 'no_data'}
        
        stress_scores = [r.calculated_stress_level for r in self.recent_readings]
        
        return {
            'average_stress': round(np.mean(stress_scores), 2),
            'max_stress': round(np.max(stress_scores), 2),
            'min_stress': round(np.min(stress_scores), 2),
            'readings_count': len(self.recent_readings),
            'high_stress_episodes': sum(1 for s in stress_scores if s > 0.7),
            'calm_periods': sum(1 for s in stress_scores if s < 0.3),
            'overall_assessment': self._get_overall_assessment(stress_scores)
        }
    
    def _get_overall_assessment(self, scores: List[float]) -> str:
        """Provide overall journey stress assessment"""
        
        avg = np.mean(scores)
        max_stress = np.max(scores)
        high_episodes = sum(1 for s in scores if s > 0.7)
        
        if avg < 0.3 and max_stress < 0.5:
            return 'excellent'  # Calm throughout
        elif avg < 0.5 and high_episodes == 0:
            return 'good'  # Generally manageable
        elif avg < 0.6 and high_episodes < 2:
            return 'moderate'  # Some challenging moments
        elif avg < 0.7 or high_episodes < 4:
            return 'challenging'  # Frequently stressful
        else:
            return 'difficult'  # Very stressful journey


# Example usage
if __name__ == "__main__":
    # Test with sample data
    user_profile = {
        'crowd_sensitivity': 4.5,
        'noise_sensitivity': 3.5,
        'transfer_anxiety': 4.0,
        'time_pressure_tolerance': 2.5
    }
    
    detector = StressDetectionSystem(user_profile)
    
    # Simulate stress detection
    current_data = {
        'heart_rate': 95,  # Elevated
        'movement_intensity': 0.8,
        'dwelling_time': 45,  # Hesitating
        'interaction_pattern': 'frequent_checking'
    }
    
    journey_context = {
        'current_crowd_level': 4,
        'current_noise_level': 3,
        'approaching_transfer': True,
        'running_late': False
    }
    
    result = detector.detect_stress(current_data, journey_context)
    
    print("Stress Detection Result:")
    print(f"Level: {result['stress_level']}")
    print(f"Score: {result['stress_score']}")
    print(f"Factors: {len(result['factors'])} identified")
    print(f"Interventions: {len(result['interventions'])} suggested")
    
    if result['interventions']:
        print("\nTop intervention:")
        print(f"  {result['interventions'][0]['title']}")
        print(f"  {result['interventions'][0]['description']}")
