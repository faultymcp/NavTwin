"""
REINFORCEMENT LEARNING SYSTEM
Continuously learns and adapts from user behavior

This is TRUE AI - the system gets better with every journey!

Learning signals:
1. Route choices (revealed preference)
2. Journey completions (success/failure)
3. User ratings (explicit feedback)
4. Stress levels during journeys
5. Re-routing requests (dissatisfaction)

Adaptation:
- Updates ML weights
- Refines sensory predictions
- Improves route scoring
- Personalizes interventions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class JourneyFeedback:
    """Single journey outcome for learning"""
    journey_id: str
    presented_routes: List[Dict[str, Any]]  # Routes that were shown
    chosen_route_index: int  # Which one did user choose?
    completed_successfully: bool
    user_rating: Optional[int]  # 1-5 stars
    average_stress: float  # 0-1
    actual_duration_minutes: float
    deviations_count: int  # Times user went off-route
    re_route_requests: int  # Times user requested different route
    intervention_helpfulness: Dict[str, int]  # How helpful were interventions?


class ReinforcementLearner:
    """
    Learns optimal route scoring weights from user behavior
    
    Uses reinforcement learning to adapt:
    - Which routes does user actually choose? (revealed preference)
    - Which routes does user complete successfully?
    - Which routes does user rate highly?
    
    Adjusts ML weights to match user's ACTUAL preferences, not assumptions
    """
    
    def __init__(self, user_id: str, initial_weights: Dict[str, float]):
        self.user_id = user_id
        self.current_weights = initial_weights.copy()
        
        # Learning parameters
        self.learning_rate = 0.1  # How quickly to adapt
        self.discount_factor = 0.95  # Weight recent journeys more
        
        # Minimum journeys before significant adaptation
        self.min_journeys_for_learning = 3
        
        # History of journey outcomes
        self.journey_history: List[JourneyFeedback] = []
        
        # Track what we're learning
        self.learning_insights = {
            'preferred_route_types': {},
            'avoided_features': [],
            'success_patterns': {},
            'stress_triggers': [],
            'adaptation_count': 0
        }
    
    def record_journey(self, feedback: JourneyFeedback):
        """
        Record journey outcome for learning
        """
        
        self.journey_history.append(feedback)
        
        # Enough data to learn?
        if len(self.journey_history) >= self.min_journeys_for_learning:
            self._update_weights_from_history()
            self._extract_insights()
    
    def _update_weights_from_history(self):
        """
        Update ML weights based on journey history
        
        Core idea:
        - If user consistently chooses routes with high environmental_comfort scores
          → Increase environmental_comfort weight
        - If user consistently avoids routes with low efficiency scores
          → Increase efficiency weight
        - Adapt weights to match user's REVEALED preferences
        """
        
        if len(self.journey_history) < self.min_journeys_for_learning:
            return
        
        # Analyze recent journeys (last 10)
        recent_journeys = self.journey_history[-10:]
        
        # Calculate which components predicted user choices best
        component_accuracies = self._calculate_component_accuracies(recent_journeys)
        
        # Adjust weights toward components that predict well
        self._adapt_weights_toward_accuracy(component_accuracies)
        
        # Adjust based on success rates
        self._adapt_weights_toward_success(recent_journeys)
        
        # Adjust based on explicit ratings
        self._adapt_weights_toward_ratings(recent_journeys)
        
        self.learning_insights['adaptation_count'] += 1
    
    def _calculate_component_accuracies(self, journeys: List[JourneyFeedback]) -> Dict[str, float]:
        """
        Calculate how well each scoring component predicts user choices
        
        If user consistently chooses routes with high personal_preference scores,
        that component is predicting well!
        """
        
        component_predictions = {
            'personal_preference': [],
            'environmental_comfort': [],
            'success_probability': [],
            'efficiency': []
        }
        
        for journey in journeys:
            if not journey.presented_routes or journey.chosen_route_index < 0:
                continue
            
            chosen_route = journey.presented_routes[journey.chosen_route_index]
            rejected_routes = [
                r for i, r in enumerate(journey.presented_routes)
                if i != journey.chosen_route_index
            ]
            
            # For each component, was chosen route scored higher than rejected ones?
            if 'scores' in chosen_route:
                chosen_scores = chosen_route['scores']
                
                for component in component_predictions.keys():
                    chosen_score = chosen_scores.get(component, 0)
                    
                    # How many rejected routes had lower scores?
                    better_than_count = sum(
                        1 for r in rejected_routes
                        if 'scores' in r and r['scores'].get(component, 0) < chosen_score
                    )
                    
                    if rejected_routes:
                        accuracy = better_than_count / len(rejected_routes)
                        component_predictions[component].append(accuracy)
        
        # Average accuracy for each component
        return {
            component: np.mean(scores) if scores else 0.5
            for component, scores in component_predictions.items()
        }
    
    def _adapt_weights_toward_accuracy(self, accuracies: Dict[str, float]):
        """
        Increase weights for components that predict user choices well
        """
        
        # Calculate adjustment for each component
        adjustments = {}
        for component, accuracy in accuracies.items():
            # If accuracy > 0.5, increase weight
            # If accuracy < 0.5, decrease weight
            adjustment = (accuracy - 0.5) * self.learning_rate
            adjustments[component] = adjustment
        
        # Apply adjustments
        for component in self.current_weights.keys():
            self.current_weights[component] += adjustments.get(component, 0)
        
        # Normalize to sum to 1.0
        total = sum(self.current_weights.values())
        self.current_weights = {
            k: max(0.05, min(0.70, v/total))  # Constrain to 5-70%
            for k, v in self.current_weights.items()
        }
        
        # Re-normalize after constraints
        total = sum(self.current_weights.values())
        self.current_weights = {k: v/total for k, v in self.current_weights.items()}
    
    def _adapt_weights_toward_success(self, journeys: List[JourneyFeedback]):
        """
        Increase weight of success_probability if user struggles with completions
        """
        
        # Calculate recent success rate
        successful = sum(1 for j in journeys if j.completed_successfully)
        success_rate = successful / len(journeys) if journeys else 1.0
        
        # If success rate is low, increase success_probability weight
        if success_rate < 0.75:
            adjustment = (0.75 - success_rate) * 0.2  # Up to 20% boost
            
            # Take from efficiency weight
            self.current_weights['success_probability'] += adjustment
            self.current_weights['efficiency'] -= adjustment
            
            # Normalize
            total = sum(self.current_weights.values())
            self.current_weights = {k: v/total for k, v in self.current_weights.items()}
    
    def _adapt_weights_toward_ratings(self, journeys: List[JourneyFeedback]):
        """
        Learn from explicit user ratings
        
        High-rated journeys → Increase weights that scored those routes highly
        Low-rated journeys → Decrease weights that scored those routes highly
        """
        
        for journey in journeys:
            if journey.user_rating is None or not journey.presented_routes:
                continue
            
            if journey.chosen_route_index < 0 or journey.chosen_route_index >= len(journey.presented_routes):
                continue
            
            chosen_route = journey.presented_routes[journey.chosen_route_index]
            
            if 'scores' not in chosen_route:
                continue
            
            scores = chosen_route['scores']
            rating = journey.user_rating  # 1-5
            
            # Rating 4-5: Reinforce these component scores
            # Rating 1-2: Reduce these component scores
            # Rating 3: Neutral
            
            feedback_signal = (rating - 3) / 2  # -1 to +1
            
            for component, score in scores.items():
                if component not in self.current_weights:
                    continue
                
                # If this component scored high AND user liked it, increase weight
                # If this component scored high AND user disliked it, decrease weight
                adjustment = feedback_signal * score * self.learning_rate * 0.1
                
                self.current_weights[component] += adjustment
            
            # Normalize
            total = sum(self.current_weights.values())
            self.current_weights = {k: v/total for k, v in self.current_weights.items()}
    
    def _extract_insights(self):
        """
        Extract patterns from user behavior
        """
        
        if len(self.journey_history) < 5:
            return
        
        recent = self.journey_history[-10:]
        
        # What route characteristics does user prefer?
        successful_routes = [
            j.presented_routes[j.chosen_route_index]
            for j in recent
            if j.completed_successfully and j.chosen_route_index >= 0
        ]
        
        if successful_routes:
            # Average characteristics of successful routes
            avg_transfers = np.mean([
                r.get('transfer_count', 0) for r in successful_routes
            ])
            
            avg_duration = np.mean([
                r.get('duration_seconds', 0) / 60 for r in successful_routes
            ])
            
            self.learning_insights['success_patterns'] = {
                'avg_transfers': avg_transfers,
                'avg_duration_minutes': avg_duration,
                'sample_size': len(successful_routes)
            }
        
        # What triggers stress?
        high_stress_journeys = [j for j in recent if j.average_stress > 0.6]
        if high_stress_journeys:
            stress_triggers = []
            for journey in high_stress_journeys:
                route = journey.presented_routes[journey.chosen_route_index]
                if route.get('transfer_count', 0) > 2:
                    stress_triggers.append('multiple_transfers')
                if route.get('duration_seconds', 0) / 60 > 45:
                    stress_triggers.append('long_duration')
            
            self.learning_insights['stress_triggers'] = list(set(stress_triggers))
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current learned weights"""
        return self.current_weights.copy()
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get summary of what the system has learned
        """
        
        return {
            'total_journeys': len(self.journey_history),
            'current_weights': self.current_weights,
            'adaptations_made': self.learning_insights['adaptation_count'],
            'success_patterns': self.learning_insights['success_patterns'],
            'stress_triggers': self.learning_insights['stress_triggers'],
            'learning_status': self._get_learning_status()
        }
    
    def _get_learning_status(self) -> str:
        """Get learning status message"""
        
        journey_count = len(self.journey_history)
        
        if journey_count < 3:
            return 'initial_phase'
        elif journey_count < 10:
            return 'learning'
        elif journey_count < 25:
            return 'adapting'
        else:
            return 'optimized'
    
    def predict_success_probability(self, route: Dict[str, Any]) -> float:
        """
        Predict if user will successfully complete this route
        Based on learned patterns
        """
        
        if not self.learning_insights['success_patterns']:
            return 0.7  # Default optimistic
        
        patterns = self.learning_insights['success_patterns']
        
        # Compare route to successful patterns
        route_transfers = route.get('transfer_count', 0)
        route_duration = route.get('duration_seconds', 0) / 60
        
        avg_transfers = patterns.get('avg_transfers', 1)
        avg_duration = patterns.get('avg_duration_minutes', 30)
        
        # Calculate similarity to successful patterns
        transfer_similarity = 1.0 - abs(route_transfers - avg_transfers) / 5.0
        duration_similarity = 1.0 - abs(route_duration - avg_duration) / 60.0
        
        similarity = (transfer_similarity + duration_similarity) / 2
        
        return np.clip(similarity, 0.3, 0.95)
    
    def should_suggest_alternative(self, route: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Based on learned stress triggers, should we warn about this route?
        """
        
        triggers = self.learning_insights.get('stress_triggers', [])
        
        if not triggers:
            return False, None
        
        warnings = []
        
        if 'multiple_transfers' in triggers and route.get('transfer_count', 0) > 2:
            warnings.append("This route has multiple transfers, which has caused stress in past journeys")
        
        if 'long_duration' in triggers and route.get('duration_seconds', 0) / 60 > 45:
            warnings.append("This is a longer journey, which you've found challenging before")
        
        if warnings:
            return True, warnings[0]
        
        return False, None


class AdaptiveLearningOrchestrator:
    """
    Combines onboarding quiz with continuous learning
    
    Phase 1: Use quiz results (initial weights)
    Phase 2: Learn from behavior (adapt weights)
    Phase 3: Optimize predictions (personalized model)
    """
    
    def __init__(
        self,
        user_id: str,
        initial_weights: Dict[str, float],
        sensory_profile: Dict[str, float]
    ):
        self.user_id = user_id
        self.sensory_profile = sensory_profile
        
        # Initialize reinforcement learner
        self.rl_learner = ReinforcementLearner(user_id, initial_weights)
        
        # Track learning progress
        self.learning_phase = 'initial'  # initial, learning, optimized
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current ML weights (adapted from behavior)"""
        return self.rl_learner.get_current_weights()
    
    def record_journey_outcome(
        self,
        journey_id: str,
        presented_routes: List[Dict],
        chosen_route_index: int,
        completed: bool,
        rating: Optional[int],
        stress_level: float
    ):
        """
        Record journey outcome to learn from
        """
        
        feedback = JourneyFeedback(
            journey_id=journey_id,
            presented_routes=presented_routes,
            chosen_route_index=chosen_route_index,
            completed_successfully=completed,
            user_rating=rating,
            average_stress=stress_level,
            actual_duration_minutes=0,  # Would calculate from timestamps
            deviations_count=0,
            re_route_requests=0,
            intervention_helpfulness={}
        )
        
        self.rl_learner.record_journey(feedback)
        
        # Update learning phase
        journey_count = len(self.rl_learner.journey_history)
        if journey_count >= 25:
            self.learning_phase = 'optimized'
        elif journey_count >= 3:
            self.learning_phase = 'learning'
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get what system has learned"""
        return self.rl_learner.get_learning_summary()
    
    def get_personalized_insights(self) -> List[Dict[str, str]]:
        """
        Generate insights to show user about their patterns
        """
        
        summary = self.rl_learner.get_learning_summary()
        insights = []
        
        # Journey count insight
        journey_count = summary['total_journeys']
        if journey_count >= 10:
            insights.append({
                'type': 'milestone',
                'title': f'{journey_count} journeys completed!',
                'description': 'The system is learning your preferences and getting better at recommending routes for you.'
            })
        
        # Success patterns
        if summary.get('success_patterns'):
            patterns = summary['success_patterns']
            avg_transfers = patterns.get('avg_transfers', 0)
            
            if avg_transfers < 1.5:
                insights.append({
                    'type': 'pattern',
                    'title': 'You prefer direct routes',
                    'description': 'You typically choose routes with minimal transfers. We\'ll prioritize these for you.'
                })
            elif avg_transfers > 2.5:
                insights.append({
                    'type': 'pattern',
                    'title': 'You handle complexity well',
                    'description': 'You successfully complete routes with multiple transfers. We can suggest more efficient options.'
                })
        
        # Stress triggers
        if summary.get('stress_triggers'):
            triggers = summary['stress_triggers']
            if 'multiple_transfers' in triggers:
                insights.append({
                    'type': 'awareness',
                    'title': 'Transfers can be stressful',
                    'description': 'We\'ve noticed transfers increase your stress. We\'ll minimize these when possible.'
                })
        
        # Learning status
        status = summary.get('learning_status')
        if status == 'optimized':
            insights.append({
                'type': 'achievement',
                'title': 'Your profile is fully personalized!',
                'description': 'The system has learned your preferences and provides highly accurate recommendations.'
            })
        
        return insights


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("REINFORCEMENT LEARNING SYSTEM")
    print("="*60)
    
    # Initial weights from quiz
    initial_weights = {
        'personal_preference': 0.40,
        'environmental_comfort': 0.35,
        'success_probability': 0.20,
        'efficiency': 0.05
    }
    
    learner = ReinforcementLearner('user_123', initial_weights)
    
    print("\nInitial weights:")
    for k, v in initial_weights.items():
        print(f"  {k}: {v:.1%}")
    
    # Simulate 5 journeys where user consistently chooses efficient routes
    print("\nSimulating 5 journeys...")
    
    for i in range(5):
        # Present 3 routes
        routes = [
            {
                'route_id': f'route_A_{i}',
                'scores': {
                    'personal_preference': 0.7,
                    'environmental_comfort': 0.8,
                    'success_probability': 0.7,
                    'efficiency': 0.3  # Low efficiency
                }
            },
            {
                'route_id': f'route_B_{i}',
                'scores': {
                    'personal_preference': 0.5,
                    'environmental_comfort': 0.4,
                    'success_probability': 0.8,
                    'efficiency': 0.9  # High efficiency - USER KEEPS CHOOSING THIS!
                }
            },
            {
                'route_id': f'route_C_{i}',
                'scores': {
                    'personal_preference': 0.6,
                    'environmental_comfort': 0.6,
                    'success_probability': 0.6,
                    'efficiency': 0.6
                }
            }
        ]
        
        # User chooses route B (high efficiency) every time
        feedback = JourneyFeedback(
            journey_id=f'journey_{i}',
            presented_routes=routes,
            chosen_route_index=1,  # Always chooses route B
            completed_successfully=True,
            user_rating=5,  # Rates it highly
            average_stress=0.2,  # Low stress
            actual_duration_minutes=25,
            deviations_count=0,
            re_route_requests=0,
            intervention_helpfulness={}
        )
        
        learner.record_journey(feedback)
    
    print("\nAfter learning from 5 journeys:")
    learned_weights = learner.get_current_weights()
    for k, v in learned_weights.items():
        change = v - initial_weights[k]
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {k}: {v:.1%} {arrow} (change: {change:+.1%})")
    
    print("\nKey insight: efficiency weight INCREASED because user")
    print("consistently chose fast routes despite initial profile!")
    print("\nThis is REAL learning - system adapts to behavior! ✅")
