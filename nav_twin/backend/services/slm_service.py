"""
SMALL LANGUAGE MODEL SERVICE
Generates natural, literal, personalized instructions for neurodivergent users
Uses user's communication preferences to adapt instruction style
"""

from typing import Dict, List, Any, Optional
import os
from openai import OpenAI  # We'll use OpenAI API initially, then migrate to on-device


class SLMService:
    """
    SLM for generating personalized navigation instructions
    
    Features:
    - Literal, step-by-step guidance
    - Adapted to user's communication style
    - Explains "why" decisions were made
    - Provides stress-aware suggestions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SLM Service
        
        For production:
        - Start with API (OpenAI/Anthropic)
        - Migrate to on-device Phi-3 or Gemma later
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Fallback to rule-based if no API
        self.use_api = self.client is not None
    
    def generate_milestone_instruction(
        self,
        milestone: Dict[str, Any],
        user_profile: Dict[str, Any],
        current_conditions: Dict[str, Any]
    ) -> str:
        """
        Generate personalized instruction for this milestone
        
        Takes into account:
        - User's communication preferences (literal vs detailed)
        - Current sensory conditions (crowd, noise)
        - User's specific sensitivities
        """
        
        if self.use_api:
            return self._generate_with_api(milestone, user_profile, current_conditions)
        else:
            return self._generate_rule_based(milestone, user_profile, current_conditions)
    
    def _generate_with_api(
        self,
        milestone: Dict[str, Any],
        user_profile: Dict[str, Any],
        current_conditions: Dict[str, Any]
    ) -> str:
        """Generate instruction using LLM API"""
        
        # Build context-aware prompt
        instruction_level = user_profile.get('instruction_detail_level', 'moderate')
        crowd_sensitivity = user_profile.get('crowd_sensitivity', 3.0)
        noise_sensitivity = user_profile.get('noise_sensitivity', 3.0)
        transfer_anxiety = user_profile.get('transfer_anxiety', 3.0)
        
        # Current conditions
        crowd_level = current_conditions.get('crowd_level', 3)
        noise_level = current_conditions.get('noise_level', 3)
        
        # Build prompt
        prompt = f"""You are a navigation assistant for a neurodivergent user. Generate a clear, literal instruction for this milestone.

User Profile:
- Instruction detail level: {instruction_level}
- Crowd sensitivity: {crowd_sensitivity}/5 (5 = very sensitive)
- Noise sensitivity: {noise_sensitivity}/5
- Transfer anxiety: {transfer_anxiety}/5

Current Conditions:
- Crowd level: {crowd_level}/5
- Noise level: {noise_level}/5

Milestone:
- Type: {milestone['type']}
- Base instruction: {milestone['instruction']}
- Transit info: {milestone.get('transit_info', {})}

Requirements:
1. Be LITERAL and CONCRETE (no metaphors)
2. Use exact numbers and distances
3. If crowd/noise is high and user is sensitive, acknowledge it and provide coping strategies
4. If this is a transfer, provide extra reassurance for high transfer anxiety
5. Keep it brief for 'brief' level, detailed for 'detailed' level

Generate the instruction:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap
                messages=[
                    {"role": "system", "content": "You are a helpful navigation assistant for neurodivergent users. Be literal, concrete, and supportive."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ SLM API error: {e}")
            return self._generate_rule_based(milestone, user_profile, current_conditions)
    
    def _generate_rule_based(
        self,
        milestone: Dict[str, Any],
        user_profile: Dict[str, Any],
        current_conditions: Dict[str, Any]
    ) -> str:
        """Generate instruction using rule-based system (fallback)"""
        
        milestone_type = milestone['type']
        transit_info = milestone.get('transit_info', {})
        distance = milestone.get('distance_meters', 0)
        
        instruction_level = user_profile.get('instruction_detail_level', 'moderate')
        crowd_sensitivity = user_profile.get('crowd_sensitivity', 3.0)
        noise_sensitivity = user_profile.get('noise_sensitivity', 3.0)
        transfer_anxiety = user_profile.get('transfer_anxiety', 3.0)
        
        crowd_level = current_conditions.get('crowd_level', 3)
        noise_level = current_conditions.get('noise_level', 3)
        
        instruction = ""
        
        # Base instruction by type
        if milestone_type == 'walk':
            instruction = f"Walk {distance} meters straight ahead."
            if instruction_level == 'detailed':
                instruction += f" This will take about {milestone.get('duration_seconds', 0) // 60} minutes."
        
        elif milestone_type == 'board_transit':
            vehicle_type = transit_info.get('vehicle_type', 'bus').lower()
            line_name = transit_info.get('line_short_name') or transit_info.get('line_name', 'Unknown')
            stop_name = transit_info.get('departure_stop', 'this stop')
            
            instruction = f"Board {vehicle_type} {line_name} at {stop_name}."
            
            if instruction_level == 'detailed':
                instruction += f" Look for the {vehicle_type} number {line_name}. "
                instruction += f"Board through any open door. Find a seat if available."
        
        elif milestone_type == 'transfer':
            instruction = "Transfer to next vehicle."
            
            if transfer_anxiety >= 4.0:
                instruction = "🌟 Transfer time. Take a breath - you've got this. " + instruction
            
            if instruction_level == 'detailed':
                instruction += " Exit current vehicle. Follow signs to next platform. "
                instruction += f"You have {milestone.get('duration_seconds', 180) // 60} minutes to transfer."
        
        elif milestone_type == 'exit_transit':
            instruction = "Exit the vehicle at the next stop."
            
            if instruction_level == 'detailed':
                stop_name = transit_info.get('arrival_stop', 'your stop')
                instruction = f"The next stop is {stop_name}. Press the stop button. Exit through the nearest door."
        
        elif milestone_type == 'arrival':
            instruction = "🎉 You have arrived at your destination!"
        
        else:
            instruction = milestone.get('instruction', 'Continue on your route.')
        
        # Add sensory warnings if needed
        warnings = []
        
        if crowd_level >= 4 and crowd_sensitivity >= 4.0:
            warnings.append("⚠️ Crowded area ahead. Consider wearing headphones.")
        
        if noise_level >= 4 and noise_sensitivity >= 4.0:
            warnings.append("⚠️ Noisy area. Noise-canceling headphones recommended.")
        
        if warnings:
            instruction = " ".join(warnings) + " " + instruction
        
        return instruction
    
    def explain_route_choice(
        self,
        chosen_route: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> str:
        """
        Explain why this route was chosen
        "Why this route?" feature
        """
        
        if self.use_api:
            return self._explain_with_api(chosen_route, alternative_routes, user_profile)
        else:
            return self._explain_rule_based(chosen_route, alternative_routes, user_profile)
    
    def _explain_with_api(
        self,
        chosen_route: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> str:
        """Explain route choice using LLM"""
        
        prompt = f"""Explain why this route was chosen over alternatives in simple, literal language.

User Sensitivities:
- Crowd sensitivity: {user_profile.get('crowd_sensitivity', 3)}/5
- Noise sensitivity: {user_profile.get('noise_sensitivity', 3)}/5
- Transfer anxiety: {user_profile.get('transfer_anxiety', 3)}/5

Chosen Route:
- Duration: {chosen_route.get('duration_seconds', 0) // 60} minutes
- Transfers: {chosen_route.get('transfer_count', 0)}
- Personal score: {chosen_route.get('personal_score', 0):.2f}

Alternative had:
- Duration: {alternative_routes[0].get('duration_seconds', 0) // 60 if alternative_routes else 0} minutes
- Transfers: {alternative_routes[0].get('transfer_count', 0) if alternative_routes else 0}

Explain in 2-3 short sentences why the chosen route is better FOR THIS USER based on their sensitivities."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You explain route choices clearly and literally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ SLM API error: {e}")
            return self._explain_rule_based(chosen_route, alternative_routes, user_profile)
    
    def _explain_rule_based(
        self,
        chosen_route: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> str:
        """Explain route choice using rules"""
        
        transfers = chosen_route.get('transfer_count', 0)
        duration = chosen_route.get('duration_seconds', 0) // 60
        
        explanation = f"I chose this route because: "
        
        reasons = []
        
        # Transfer reasoning
        transfer_anxiety = user_profile.get('transfer_anxiety', 3.0)
        if transfers == 0 and transfer_anxiety >= 4.0:
            reasons.append("it's a direct route with no transfers (important for your comfort)")
        elif transfers > 0 and transfer_anxiety >= 4.0:
            reasons.append(f"it has only {transfers} transfer(s), which is the minimum available")
        
        # Crowd reasoning
        crowd_sensitivity = user_profile.get('crowd_sensitivity', 3.0)
        if crowd_sensitivity >= 4.0:
            reasons.append("it avoids the busiest routes")
        
        # Time
        if alternative_routes:
            alt_duration = alternative_routes[0].get('duration_seconds', 0) // 60
            time_diff = abs(duration - alt_duration)
            if time_diff < 5:
                reasons.append("the travel time is similar to faster routes")
        
        if not reasons:
            reasons.append("it balances comfort and efficiency for you")
        
        explanation += ", ".join(reasons) + "."
        
        return explanation
    
    def generate_stress_response(
        self,
        stress_level: float,
        stress_factors: List[str],
        current_location: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> str:
        """
        Generate supportive response when stress is detected
        """
        
        if stress_level < 0.3:
            return "✨ You're doing great! Keep going!"
        
        elif stress_level < 0.6:
            return "💪 I notice your stress is rising a bit. Take a deep breath. You're doing well!"
        
        else:
            # High stress - provide actionable help
            suggestions = []
            
            if 'crowd' in stress_factors:
                suggestions.append("Find a quieter corner if possible")
            
            if 'noise' in stress_factors:
                suggestions.append("Put on your headphones")
            
            if 'transfer' in stress_factors:
                suggestions.append("Take your time with this transfer - there's no rush")
            
            suggestion_text = ". ".join(suggestions) if suggestions else "Take a moment to pause if you need"
            
            return f"🌟 I notice you're stressed. {suggestion_text}. You're doing amazing - this will pass."


# ============================================================================
# ON-DEVICE SLM (Future Implementation)
# ============================================================================

class OnDeviceSLM:
    """
    On-device SLM using Phi-3 Mini or Gemma-2B
    For privacy-first, offline navigation
    
    To implement:
    1. Convert model to ONNX format
    2. Use ONNX Runtime for inference
    3. Run on mobile device
    """
    
    def __init__(self, model_path: str):
        """
        Initialize on-device model
        
        Example:
        model = OnDeviceSLM("models/phi-3-mini.onnx")
        """
        self.model_path = model_path
        # TODO: Load ONNX model
        # self.session = onnxruntime.InferenceSession(model_path)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using on-device model"""
        # TODO: Implement
        pass