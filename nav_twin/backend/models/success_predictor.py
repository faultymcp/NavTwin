"""
COMPONENT 5: Journey Success Predictor
Predicts probability of successful journey completion and identifies risk factors
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SuccessPrediction:
    """Journey success prediction result"""
    probability: float  # 0-1
    confidence: float  # 0-1
    risk_level: str  # 'low', 'moderate', 'high', 'critical'
    risk_factors: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    timestamp: datetime


class JourneySuccessPredictor:
    """
    ML-based prediction of journey completion success
    
    Uses:
    - Historical journey data (completion rate, patterns)
    - Current user state (stress, fatigue, confidence)
    - Route characteristics (complexity, length, transfers)
    - Environmental conditions (crowds, delays, weather)
    - Time factors (rush hour, familiarity with time)
    
    Provides early warnings and preventive interventions
    """
    
    def __init__(self, user_history: Dict[str, Any], user_profile: Dict[str, Any]):
        self.history = user_history
        self.profile = user_profile
        
        # Calculate user's baseline success patterns
        self.baseline_success_rate = user_history.get('overall_success_rate', 0.85)
        self.successful_patterns = user_history.get('successful_patterns', {})
        self.failure_patterns = user_history.get('failure_patterns', {})
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.15,      # <15% failure probability
            'moderate': 0.30,  # 15-30% failure probability
            'high': 0.50,      # 30-50% failure probability
            'critical': 0.50   # >50% failure probability
        }
    
    def predict_success(
        self,
        route_data: Dict[str, Any],
        current_state: Dict[str, Any],
        environmental_conditions: Dict[str, Any]
    ) -> SuccessPrediction:
        """
        Main prediction function
        
        Args:
            route_data: Route characteristics (duration, transfers, complexity)
            current_state: User's current physical/mental state
            environmental_conditions: Current external conditions
            
        Returns:
            SuccessPrediction with probability, risks, and recommendations
        """
        
        # Calculate component scores
        route_risk = self._assess_route_risk(route_data)
        state_risk = self._assess_user_state_risk(current_state)
        environmental_risk = self._assess_environmental_risk(environmental_conditions)
        historical_risk = self._assess_historical_risk(route_data, current_state)
        
        # Combine risks with weighted average
        combined_risk = self._combine_risk_factors(
            route_risk,
            state_risk,
            environmental_risk,
            historical_risk
        )
        
        # Convert risk to success probability
        success_probability = 1.0 - combined_risk
        
        # Calculate confidence in prediction
        confidence = self._calculate_confidence(route_data, current_state)
        
        # Classify risk level
        risk_level = self._classify_risk_level(combined_risk)
        
        # Identify specific risk factors
        risk_factors = self._identify_risk_factors(
            route_data,
            current_state,
            environmental_conditions,
            route_risk,
            state_risk,
            environmental_risk
        )
        
        # Identify protective factors
        protective_factors = self._identify_protective_factors(
            route_data,
            current_state,
            environmental_conditions
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level,
            risk_factors,
            protective_factors,
            route_data
        )
        
        return SuccessPrediction(
            probability=success_probability,
            confidence=confidence,
            risk_level=risk_level,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _assess_route_risk(self, route_data: Dict) -> float:
        """Assess risk from route characteristics"""
        
        risk_scores = []
        
        # Duration risk
        duration_minutes = route_data.get('duration_minutes', 0)
        if duration_minutes > 60:
            duration_risk = min((duration_minutes - 60) / 60, 1.0)
            risk_scores.append(duration_risk * 0.2)
        
        # Transfer risk
        transfers = route_data.get('transfer_count', 0)
        transfer_anxiety = self.profile.get('transfer_anxiety', 3.0)
        transfer_risk = (transfers / 3.0) * (transfer_anxiety / 5.0)
        risk_scores.append(transfer_risk * 0.3)
        
        # Complexity risk
        complexity = route_data.get('complexity_score', 0)  # 0-1
        risk_scores.append(complexity * 0.25)
        
        # Unfamiliarity risk
        familiarity = route_data.get('familiarity_score', 0)  # 0-1, higher = more familiar
        familiarity_preference = self.profile.get('preference_for_familiarity', 3.0)
        unfamiliarity_risk = (1.0 - familiarity) * (familiarity_preference / 5.0)
        risk_scores.append(unfamiliarity_risk * 0.25)
        
        return np.clip(np.mean(risk_scores), 0, 1)
    
    def _assess_user_state_risk(self, state: Dict) -> float:
        """Assess risk from user's current state"""
        
        risk_scores = []
        
        # Stress level risk
        current_stress = state.get('current_stress_level', 0)  # 0-1
        risk_scores.append(current_stress * 0.4)
        
        # Fatigue risk
        fatigue_level = state.get('fatigue_level', 0)  # 0-1
        risk_scores.append(fatigue_level * 0.2)
        
        # Confidence risk
        confidence = state.get('user_confidence', 0.7)  # 0-1
        confidence_risk = 1.0 - confidence
        risk_scores.append(confidence_risk * 0.25)
        
        # Recent experience
        recent_failures = state.get('recent_failure_count', 0)
        if recent_failures > 0:
            failure_impact = min(recent_failures / 3.0, 1.0)
            risk_scores.append(failure_impact * 0.15)
        
        return np.clip(np.mean(risk_scores), 0, 1)
    
    def _assess_environmental_risk(self, conditions: Dict) -> float:
        """Assess risk from environmental conditions"""
        
        risk_scores = []
        
        # Weather risk
        weather = conditions.get('weather_condition', 'clear')
        weather_risks = {
            'clear': 0.0,
            'rain': 0.3,
            'heavy_rain': 0.6,
            'snow': 0.7,
            'extreme': 0.9
        }
        risk_scores.append(weather_risks.get(weather, 0.0) * 0.2)
        
        # Crowd level risk
        crowd_level = conditions.get('predicted_crowd_level', 0)  # 0-5
        crowd_sensitivity = self.profile.get('crowd_sensitivity', 3.0)
        crowd_risk = (crowd_level / 5.0) * (crowd_sensitivity / 5.0)
        risk_scores.append(crowd_risk * 0.3)
        
        # Service disruptions
        if conditions.get('service_disruptions', False):
            risk_scores.append(0.8 * 0.25)
        
        # Time of day risk
        time_of_day = conditions.get('time_of_day', 'midday')
        if time_of_day == 'rush_hour':
            risk_scores.append(0.4 * 0.15)
        
        # Delays predicted
        predicted_delay = conditions.get('predicted_delay_minutes', 0)
        if predicted_delay > 5:
            delay_risk = min(predicted_delay / 20, 1.0)
            risk_scores.append(delay_risk * 0.1)
        
        return np.clip(np.mean(risk_scores), 0, 1)
    
    def _assess_historical_risk(self, route_data: Dict, state: Dict) -> float:
        """Assess risk based on historical patterns"""
        
        # Check if similar routes were successful
        similar_success_rate = self.history.get('similar_routes_success_rate', self.baseline_success_rate)
        
        # Check time-of-day patterns
        current_time = datetime.now().hour
        time_success_rate = self.history.get('time_success_rates', {}).get(str(current_time), self.baseline_success_rate)
        
        # Check transfer pattern success
        transfers = route_data.get('transfer_count', 0)
        transfer_success_rate = self.history.get('transfer_success_rates', {}).get(str(transfers), self.baseline_success_rate)
        
        # Average success rates
        avg_success = np.mean([similar_success_rate, time_success_rate, transfer_success_rate])
        
        # Convert to risk (inverse of success)
        historical_risk = 1.0 - avg_success
        
        return np.clip(historical_risk, 0, 1)
    
    def _combine_risk_factors(
        self,
        route_risk: float,
        state_risk: float,
        env_risk: float,
        hist_risk: float
    ) -> float:
        """Combine risk factors with adaptive weighting"""
        
        # Base weights
        weights = {
            'route': 0.25,
            'state': 0.35,
            'environmental': 0.25,
            'historical': 0.15
        }
        
        # Increase weight of historical data if user has extensive history
        journey_count = self.history.get('total_journeys', 0)
        if journey_count > 50:
            weights['historical'] = 0.25
            weights['route'] = 0.2
            weights['state'] = 0.3
            weights['environmental'] = 0.25
        
        combined = (
            route_risk * weights['route'] +
            state_risk * weights['state'] +
            env_risk * weights['environmental'] +
            hist_risk * weights['historical']
        )
        
        return np.clip(combined, 0, 1)
    
    def _calculate_confidence(self, route_data: Dict, state: Dict) -> float:
        """Calculate confidence in the prediction"""
        
        confidence_factors = []
        
        # Data availability
        data_completeness = self._assess_data_completeness(route_data, state)
        confidence_factors.append(data_completeness * 0.3)
        
        # Historical data volume
        journey_count = self.history.get('total_journeys', 0)
        history_confidence = min(journey_count / 100, 1.0)
        confidence_factors.append(history_confidence * 0.4)
        
        # Pattern consistency
        pattern_consistency = self.history.get('pattern_consistency_score', 0.7)
        confidence_factors.append(pattern_consistency * 0.3)
        
        return np.clip(np.mean(confidence_factors), 0, 1)
    
    def _assess_data_completeness(self, route_data: Dict, state: Dict) -> float:
        """Assess how complete our input data is"""
        
        required_fields = [
            'duration_minutes', 'transfer_count', 'complexity_score',
            'current_stress_level', 'user_confidence'
        ]
        
        available = sum(1 for field in required_fields if route_data.get(field) is not None or state.get(field) is not None)
        
        return available / len(required_fields)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify overall risk level"""
        
        failure_probability = risk_score
        
        if failure_probability < self.risk_thresholds['low']:
            return 'low'
        elif failure_probability < self.risk_thresholds['moderate']:
            return 'moderate'
        elif failure_probability < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _identify_risk_factors(
        self,
        route_data: Dict,
        state: Dict,
        conditions: Dict,
        route_risk: float,
        state_risk: float,
        env_risk: float
    ) -> List[Dict[str, Any]]:
        """Identify specific risk factors"""
        
        factors = []
        
        # Route-related risks
        if route_data.get('transfer_count', 0) > 2:
            factors.append({
                'category': 'route',
                'factor': 'multiple_transfers',
                'severity': 'high' if route_data['transfer_count'] > 3 else 'moderate',
                'description': f"{route_data['transfer_count']} transfers required",
                'impact': 0.3
            })
        
        if route_data.get('duration_minutes', 0) > 60:
            factors.append({
                'category': 'route',
                'factor': 'long_duration',
                'severity': 'moderate',
                'description': f"{route_data['duration_minutes']} minute journey",
                'impact': 0.2
            })
        
        # State-related risks
        if state.get('current_stress_level', 0) > 0.6:
            factors.append({
                'category': 'user_state',
                'factor': 'elevated_stress',
                'severity': 'high',
                'description': 'Currently experiencing high stress',
                'impact': 0.4
            })
        
        if state.get('fatigue_level', 0) > 0.6:
            factors.append({
                'category': 'user_state',
                'factor': 'fatigue',
                'severity': 'moderate',
                'description': 'User reports feeling tired',
                'impact': 0.2
            })
        
        # Environmental risks
        if conditions.get('predicted_crowd_level', 0) > 3:
            if self.profile.get('crowd_sensitivity', 3) > 3:
                factors.append({
                    'category': 'environment',
                    'factor': 'high_crowds',
                    'severity': 'high',
                    'description': 'Heavy crowding expected',
                    'impact': 0.3
                })
        
        if conditions.get('service_disruptions'):
            factors.append({
                'category': 'environment',
                'factor': 'service_disruption',
                'severity': 'high',
                'description': 'Active service disruptions reported',
                'impact': 0.4
            })
        
        # Sort by impact
        factors.sort(key=lambda x: x['impact'], reverse=True)
        
        return factors[:5]  # Top 5 risk factors
    
    def _identify_protective_factors(
        self,
        route_data: Dict,
        state: Dict,
        conditions: Dict
    ) -> List[Dict[str, Any]]:
        """Identify factors that increase success probability"""
        
        factors = []
        
        # Route familiarity
        if route_data.get('familiarity_score', 0) > 0.7:
            factors.append({
                'factor': 'route_familiarity',
                'strength': 'high',
                'description': 'You\'ve taken this route before',
                'benefit': 0.3
            })
        
        # User confidence
        if state.get('user_confidence', 0) > 0.7:
            factors.append({
                'factor': 'high_confidence',
                'strength': 'moderate',
                'description': 'Feeling confident about the journey',
                'benefit': 0.2
            })
        
        # Low stress
        if state.get('current_stress_level', 0) < 0.3:
            factors.append({
                'factor': 'calm_state',
                'strength': 'moderate',
                'description': 'Currently in a calm state',
                'benefit': 0.2
            })
        
        # Good timing
        if conditions.get('time_of_day') == 'off_peak':
            factors.append({
                'factor': 'good_timing',
                'strength': 'moderate',
                'description': 'Traveling during quieter hours',
                'benefit': 0.15
            })
        
        # Recent success
        recent_successes = state.get('recent_success_count', 0)
        if recent_successes > 2:
            factors.append({
                'factor': 'positive_momentum',
                'strength': 'moderate',
                'description': f"{recent_successes} successful recent journeys",
                'benefit': 0.2
            })
        
        return factors
    
    def _generate_recommendations(
        self,
        risk_level: str,
        risk_factors: List[Dict],
        protective_factors: List[Dict],
        route_data: Dict
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Critical risk - suggest alternatives
        if risk_level == 'critical':
            recommendations.append({
                'priority': 'critical',
                'type': 'route_change',
                'title': 'Consider alternative route',
                'description': 'This route has high risk factors. Let me find a better option.',
                'action': 'suggest_alternative_routes'
            })
        
        # High risk - preventive measures
        if risk_level in ['high', 'critical']:
            recommendations.append({
                'priority': 'high',
                'type': 'preparation',
                'title': 'Extra preparation recommended',
                'description': 'Review the route carefully before starting.',
                'action': 'show_detailed_preview'
            })
        
        # Factor-specific recommendations
        for factor in risk_factors:
            if factor['factor'] == 'multiple_transfers':
                recommendations.append({
                    'priority': 'moderate',
                    'type': 'guidance',
                    'title': 'Transfer assistance available',
                    'description': 'I\'ll give you detailed guidance at each transfer point.',
                    'action': 'enable_transfer_guidance'
                })
            
            elif factor['factor'] == 'elevated_stress':
                recommendations.append({
                    'priority': 'high',
                    'type': 'wellbeing',
                    'title': 'Start when ready',
                    'description': 'Take a few minutes to calm down before beginning.',
                    'action': 'suggest_delay'
                })
            
            elif factor['factor'] == 'high_crowds':
                recommendations.append({
                    'priority': 'moderate',
                    'type': 'timing',
                    'title': 'Consider delaying',
                    'description': 'Crowds expected to decrease in 30 minutes.',
                    'action': 'suggest_optimal_departure_time'
                })
        
        # Leverage protective factors
        if protective_factors:
            strongest = protective_factors[0]
            recommendations.append({
                'priority': 'low',
                'type': 'encouragement',
                'title': 'You\'ve got this!',
                'description': strongest['description'] + ' - that\'s in your favor.',
                'action': 'show_positive_reinforcement'
            })
        
        return sorted(recommendations, key=lambda x: {'critical': 4, 'high': 3, 'moderate': 2, 'low': 1}[x['priority']], reverse=True)[:4]
    
    def update_with_outcome(self, journey_id: str, success: bool, failure_reason: Optional[str] = None):
        """Update predictor with actual journey outcome for learning"""
        
        # This would update the ML model in a real implementation
        # For now, update basic statistics
        
        if success:
            self.baseline_success_rate = (
                self.baseline_success_rate * 0.95 + 0.05  # Incremental update
            )
        else:
            self.baseline_success_rate = (
                self.baseline_success_rate * 0.95  # Incremental decrease
            )
        
        # Store patterns for future learning
        # In production, this would retrain the ML model


# Example usage
if __name__ == "__main__":
    # Test with sample data
    user_history = {
        'overall_success_rate': 0.82,
        'total_journeys': 45,
        'similar_routes_success_rate': 0.88,
        'pattern_consistency_score': 0.75
    }
    
    user_profile = {
        'crowd_sensitivity': 4.5,
        'transfer_anxiety': 4.0,
        'preference_for_familiarity': 4.5
    }
    
    predictor = JourneySuccessPredictor(user_history, user_profile)
    
    # Predict for a challenging journey
    route_data = {
        'duration_minutes': 45,
        'transfer_count': 3,
        'complexity_score': 0.7,
        'familiarity_score': 0.2  # Unfamiliar route
    }
    
    current_state = {
        'current_stress_level': 0.6,
        'fatigue_level': 0.4,
        'user_confidence': 0.5,
        'recent_failure_count': 1
    }
    
    environmental_conditions = {
        'weather_condition': 'rain',
        'predicted_crowd_level': 4,
        'service_disruptions': False,
        'time_of_day': 'rush_hour',
        'predicted_delay_minutes': 5
    }
    
    prediction = predictor.predict_success(
        route_data,
        current_state,
        environmental_conditions
    )
    
    print("Journey Success Prediction:")
    print(f"Success Probability: {prediction.probability:.1%}")
    print(f"Confidence: {prediction.confidence:.1%}")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"\nRisk Factors: {len(prediction.risk_factors)}")
    for factor in prediction.risk_factors[:3]:
        print(f"  - {factor['description']} (impact: {factor['impact']})")
    print(f"\nRecommendations: {len(prediction.recommendations)}")
    for rec in prediction.recommendations[:2]:
        print(f"  - {rec['title']}: {rec['description']}")
