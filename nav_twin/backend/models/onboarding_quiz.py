"""
ONBOARDING QUIZ SYSTEM
Comprehensive assessment to understand user's sensory profile BEFORE first journey

This establishes:
1. Sensory sensitivities (8 dimensions)
2. Past navigation experiences
3. Anxiety triggers
4. Preferences and coping strategies
5. Initial ML weights for route scoring
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QuizQuestion:
    """Single quiz question"""
    id: str
    question: str
    type: str  # 'scale', 'multiple_choice', 'multiple_select', 'scenario'
    category: str  # Which sensory dimension this measures
    options: Optional[List[str]] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    scenario: Optional[str] = None


class OnboardingQuiz:
    """
    Comprehensive onboarding assessment
    
    Purpose:
    - Understand user's sensory profile
    - Identify anxiety triggers
    - Establish baseline preferences
    - Set initial ML weights
    - Provide personalized recommendations from DAY ONE
    """
    
    def __init__(self):
        self.questions = self._create_quiz_questions()
    
    def _create_quiz_questions(self) -> List[QuizQuestion]:
        """Create comprehensive quiz questions"""
        
        questions = []
        
        # ====================================================================
        # SECTION 1: SENSORY SENSITIVITIES (Core Profile)
        # ====================================================================
        
        # Crowd Sensitivity
        questions.append(QuizQuestion(
            id='crowd_1',
            category='crowd_sensitivity',
            type='scale',
            question='How do you feel in crowded spaces like busy train stations?',
            min_value=1,
            max_value=5,
            options=[
                '1 - Completely comfortable',
                '2 - Slightly uncomfortable',
                '3 - Moderately uncomfortable',
                '4 - Very uncomfortable',
                '5 - Extremely distressing'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='crowd_2',
            category='crowd_sensitivity',
            type='scenario',
            scenario='You arrive at a train platform and it\'s packed with people. What do you do?',
            options=[
                'Board the train immediately - crowds don\'t bother me',
                'Wait for the next train if it\'s too crowded',
                'Find a less crowded spot on the platform',
                'Feel very anxious but push through',
                'Avoid the journey altogether if possible'
            ]
        ))
        
        # Noise Sensitivity
        questions.append(QuizQuestion(
            id='noise_1',
            category='noise_sensitivity',
            type='scale',
            question='How sensitive are you to loud or unexpected noises?',
            min_value=1,
            max_value=5,
            options=[
                '1 - Not bothered at all',
                '2 - Slightly noticeable',
                '3 - Moderately distracting',
                '4 - Very distressing',
                '5 - Overwhelming/painful'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='noise_2',
            category='noise_sensitivity',
            type='multiple_select',
            question='Which sounds bother you most during travel? (Select all that apply)',
            options=[
                'Train/bus announcements',
                'People talking loudly',
                'Construction/drilling',
                'Screeching brakes',
                'Emergency sirens',
                'Background music/TV',
                'None of these bother me'
            ]
        ))
        
        # Visual Sensitivity
        questions.append(QuizQuestion(
            id='visual_1',
            category='visual_sensitivity',
            type='scale',
            question='How do you react to bright lights, flashing screens, or visually busy environments?',
            min_value=1,
            max_value=5,
            options=[
                '1 - No issues',
                '2 - Slight discomfort',
                '3 - Moderately overwhelming',
                '4 - Very difficult to handle',
                '5 - Causes physical pain/headaches'
            ]
        ))
        
        # Touch/Physical Sensitivity
        questions.append(QuizQuestion(
            id='touch_1',
            category='touch_sensitivity',
            type='scale',
            question='How do you feel about being in close physical proximity to strangers (e.g., on a crowded bus)?',
            min_value=1,
            max_value=5,
            options=[
                '1 - No problem',
                '2 - Slightly uncomfortable',
                '3 - Moderately distressing',
                '4 - Very distressing',
                '5 - Unbearable'
            ]
        ))
        
        # Transfer Anxiety
        questions.append(QuizQuestion(
            id='transfer_1',
            category='transfer_anxiety',
            type='scale',
            question='How anxious do you feel about changing between different transport modes (bus to train, etc.)?',
            min_value=1,
            max_value=5,
            options=[
                '1 - No anxiety',
                '2 - Slight nervousness',
                '3 - Moderate anxiety',
                '4 - High anxiety',
                '5 - Severe anxiety/avoid transfers'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='transfer_2',
            category='transfer_anxiety',
            type='scenario',
            scenario='You need to transfer from a bus to a train in an unfamiliar station. How do you feel?',
            options=[
                'Confident - I can figure it out',
                'A bit nervous but manageable',
                'Worried about getting lost',
                'Very anxious - would prefer to avoid',
                'Would plan entire route to minimize transfers'
            ]
        ))
        
        # Time Pressure Tolerance
        questions.append(QuizQuestion(
            id='time_1',
            category='time_pressure_tolerance',
            type='scale',
            question='How do you handle running late or time pressure during travel?',
            min_value=1,
            max_value=5,
            options=[
                '1 - Handle it well',
                '2 - Slightly stressful',
                '3 - Moderately stressful',
                '4 - Very stressful',
                '5 - Extremely distressing'
            ]
        ))
        
        # Preference for Familiarity
        questions.append(QuizQuestion(
            id='familiar_1',
            category='preference_for_familiarity',
            type='scale',
            question='How important is it for you to take routes you know well?',
            min_value=1,
            max_value=5,
            options=[
                '1 - Don\'t mind trying new routes',
                '2 - Slightly prefer familiar routes',
                '3 - Moderately prefer familiar routes',
                '4 - Strongly prefer familiar routes',
                '5 - Only take routes I know well'
            ]
        ))
        
        # ====================================================================
        # SECTION 2: PAST EXPERIENCES
        # ====================================================================
        
        questions.append(QuizQuestion(
            id='experience_1',
            category='past_experience',
            type='multiple_choice',
            question='Have you had overwhelming or distressing experiences with public transport?',
            options=[
                'Never',
                'Rarely (1-2 times)',
                'Sometimes (3-5 times)',
                'Often (6-10 times)',
                'Frequently (10+ times)'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='experience_2',
            category='past_experience',
            type='multiple_select',
            question='What typically causes distress during your journeys? (Select all that apply)',
            options=[
                'Getting lost or confused',
                'Unexpected delays',
                'Crowded vehicles',
                'Loud noises',
                'Having to ask for help',
                'Missing connections',
                'Sensory overload',
                'Social anxiety',
                'None of these'
            ]
        ))
        
        # ====================================================================
        # SECTION 3: COPING STRATEGIES & PREFERENCES
        # ====================================================================
        
        questions.append(QuizQuestion(
            id='coping_1',
            category='coping_strategies',
            type='multiple_select',
            question='What helps you feel calmer during difficult journeys? (Select all that apply)',
            options=[
                'Listening to music/podcasts',
                'Having detailed step-by-step instructions',
                'Knowing exactly what to expect',
                'Having extra time (not rushing)',
                'Traveling during quieter times',
                'Having someone with me',
                'Taking breaks when needed',
                'Focusing on breathing exercises'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='instruction_1',
            category='instruction_preference',
            type='multiple_choice',
            question='What level of navigation instructions do you prefer?',
            options=[
                'Minimal - just key steps',
                'Moderate - important details',
                'Detailed - every single step',
                'Very detailed - with reassurance and tips'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='visual_1_pref',
            category='communication_preference',
            type='multiple_choice',
            question='How do you prefer to receive navigation information?',
            options=[
                'Visual only (text + maps)',
                'Mostly visual with some audio',
                'Balanced visual and audio',
                'Mostly audio with some visual',
                'Audio only (voice guidance)'
            ]
        ))
        
        # ====================================================================
        # SECTION 4: PRIORITIES & TRADE-OFFS
        # ====================================================================
        
        questions.append(QuizQuestion(
            id='priority_1',
            category='route_priorities',
            type='scenario',
            scenario='You have two route options:\nRoute A: 20 minutes, crowded, one transfer\nRoute B: 35 minutes, quiet, direct\nWhich do you choose?',
            options=[
                'Route A - Speed is most important',
                'Route B - Comfort is most important',
                'Depends on how I\'m feeling that day',
                'Would want to see more details first'
            ]
        ))
        
        questions.append(QuizQuestion(
            id='priority_2',
            category='route_priorities',
            type='multiple_choice',
            question='When choosing a route, what matters MOST to you?',
            options=[
                'Getting there fastest',
                'Avoiding crowds and noise',
                'Minimal complexity (fewer transfers)',
                'Routes I know well',
                'Success - routes I can actually complete'
            ]
        ))
        
        return questions
    
    def calculate_sensory_profile(self, answers: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate sensory profile from quiz answers
        
        Returns profile with scores 1-5 for each dimension
        """
        
        profile = {
            'crowd_sensitivity': 0.0,
            'noise_sensitivity': 0.0,
            'visual_sensitivity': 0.0,
            'touch_sensitivity': 0.0,
            'transfer_anxiety': 0.0,
            'time_pressure_tolerance': 0.0,
            'preference_for_familiarity': 0.0,
            'overall_sensitivity': 0.0
        }
        
        # Aggregate scores by category
        category_scores = {}
        category_counts = {}
        
        for question_id, answer in answers.items():
            question = next((q for q in self.questions if q.id == question_id), None)
            if not question:
                continue
            
            category = question.category
            
            # Convert answer to numerical score
            score = self._answer_to_score(answer, question)
            
            if category not in category_scores:
                category_scores[category] = []
            
            category_scores[category].append(score)
        
        # Average scores for each sensory dimension
        for dimension in profile.keys():
            if dimension == 'overall_sensitivity':
                continue
            
            if dimension in category_scores:
                profile[dimension] = np.mean(category_scores[dimension])
        
        # Calculate overall sensitivity
        sensitivity_dimensions = [
            'crowd_sensitivity',
            'noise_sensitivity',
            'visual_sensitivity',
            'touch_sensitivity'
        ]
        profile['overall_sensitivity'] = np.mean([
            profile[dim] for dim in sensitivity_dimensions
        ])
        
        return profile
    
    def _answer_to_score(self, answer: Any, question: QuizQuestion) -> float:
        """Convert quiz answer to numerical score (1-5)"""
        
        if question.type == 'scale':
            # Direct numerical answer
            return float(answer)
        
        elif question.type == 'multiple_choice':
            # Map option index to score
            if isinstance(answer, int):
                return float(answer + 1)  # 0-indexed to 1-5
            return 3.0  # Default middle
        
        elif question.type == 'scenario':
            # Map scenario choice to severity
            # First option = 1, last option = 5
            if isinstance(answer, int):
                num_options = len(question.options)
                return 1 + (answer / (num_options - 1)) * 4
            return 3.0
        
        elif question.type == 'multiple_select':
            # Count number of selections as indicator
            if isinstance(answer, list):
                return min(len(answer) / 2, 5.0)
            return 3.0
        
        return 3.0  # Default
    
    def calculate_initial_ml_weights(
        self,
        sensory_profile: Dict[str, float],
        answers: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate initial ML weights from quiz responses
        
        This determines how routes are scored for THIS user
        """
        
        # Get priority from answers
        priority_answer = answers.get('priority_2', 2)  # Default to comfort
        
        # Get overall sensitivity
        avg_sensitivity = sensory_profile.get('overall_sensitivity', 3.0)
        
        # Base weights on both priority and sensitivity
        if priority_answer == 0:  # Speed most important
            weights = {
                'personal_preference': 0.15,
                'environmental_comfort': 0.15,
                'success_probability': 0.20,
                'efficiency': 0.50  # Prioritize speed
            }
        
        elif priority_answer == 1:  # Comfort most important
            weights = {
                'personal_preference': 0.40,
                'environmental_comfort': 0.35,  # Prioritize comfort
                'success_probability': 0.20,
                'efficiency': 0.05
            }
        
        elif priority_answer == 2:  # Simplicity most important
            weights = {
                'personal_preference': 0.30,
                'environmental_comfort': 0.25,
                'success_probability': 0.35,  # Prioritize completion
                'efficiency': 0.10
            }
        
        elif priority_answer == 3:  # Familiarity most important
            weights = {
                'personal_preference': 0.45,  # Prioritize known routes
                'environmental_comfort': 0.25,
                'success_probability': 0.25,
                'efficiency': 0.05
            }
        
        else:  # Success most important
            weights = {
                'personal_preference': 0.30,
                'environmental_comfort': 0.25,
                'success_probability': 0.40,  # Prioritize completion
                'efficiency': 0.05
            }
        
        # Adjust based on sensitivity level
        if avg_sensitivity >= 4.0:
            # Very high sensitivity - boost comfort even more
            weights['environmental_comfort'] += 0.10
            weights['efficiency'] = max(0.05, weights['efficiency'] - 0.10)
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def generate_recommendations(
        self,
        sensory_profile: Dict[str, float],
        answers: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate personalized recommendations based on quiz
        """
        
        recommendations = []
        
        # High crowd sensitivity
        if sensory_profile.get('crowd_sensitivity', 0) >= 4.0:
            recommendations.append({
                'category': 'timing',
                'title': 'Travel during off-peak hours',
                'description': 'You\'re sensitive to crowds. We\'ll prioritize quieter routes and suggest optimal departure times.'
            })
        
        # High transfer anxiety
        if sensory_profile.get('transfer_anxiety', 0) >= 4.0:
            recommendations.append({
                'category': 'routing',
                'title': 'Minimize transfers',
                'description': 'We\'ll prioritize direct routes and provide extra guidance for necessary transfers.'
            })
        
        # High noise sensitivity
        if sensory_profile.get('noise_sensitivity', 0) >= 4.0:
            recommendations.append({
                'category': 'comfort',
                'title': 'Quieter transport options',
                'description': 'We\'ll suggest quieter buses over trains where possible, and indicate noise levels.'
            })
        
        # High familiarity preference
        if sensory_profile.get('preference_for_familiarity', 0) >= 4.0:
            recommendations.append({
                'category': 'learning',
                'title': 'Build route familiarity',
                'description': 'We\'ll help you learn new routes gradually, showing detailed previews before you start.'
            })
        
        return recommendations
    
    def generate_quiz_for_api(self) -> List[Dict[str, Any]]:
        """
        Generate quiz in API-friendly format
        """
        
        return [
            {
                'id': q.id,
                'question': q.question,
                'type': q.type,
                'category': q.category,
                'options': q.options,
                'scenario': q.scenario,
                'min_value': q.min_value,
                'max_value': q.max_value
            }
            for q in self.questions
        ]


# Example usage
if __name__ == "__main__":
    quiz = OnboardingQuiz()
    
    print("="*60)
    print("ONBOARDING QUIZ SYSTEM")
    print("="*60)
    print(f"\nTotal questions: {len(quiz.questions)}")
    
    # Sample answers (simulating a highly sensitive user)
    sample_answers = {
        'crowd_1': 5,  # Very uncomfortable in crowds
        'crowd_2': 4,  # Avoid crowded platforms
        'noise_1': 5,  # Very sensitive to noise
        'visual_1': 4,  # Visually overwhelming
        'touch_1': 5,  # Distressed by close proximity
        'transfer_1': 5,  # High transfer anxiety
        'time_1': 4,  # Stressed by time pressure
        'familiar_1': 5,  # Strong preference for familiar routes
        'priority_2': 1  # Comfort most important
    }
    
    # Calculate profile
    profile = quiz.calculate_sensory_profile(sample_answers)
    weights = quiz.calculate_initial_ml_weights(profile, sample_answers)
    recommendations = quiz.generate_recommendations(profile, sample_answers)
    
    print("\nCalculated Sensory Profile:")
    for dimension, score in profile.items():
        print(f"  {dimension}: {score:.1f}/5.0")
    
    print("\nInitial ML Weights:")
    for component, weight in weights.items():
        print(f"  {component}: {weight:.1%}")
    
    print(f"\nPersonalized Recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"  - {rec['title']}")
