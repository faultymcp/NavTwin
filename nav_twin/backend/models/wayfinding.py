"""
COMPONENT 3: AI-Powered Wayfinding Assistant
Provides turn-by-turn navigation with NLP and computer vision support
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class WayfindingAssistant:
    """
    AI-powered wayfinding with:
    - Adaptive instruction detail levels
    - Natural language simplification
    - Visual landmark guidance
    - Conversational Q&A support
    """
    
    def __init__(self, user_profile: Dict[str, Any]):
        self.profile = user_profile
        self.detail_level = user_profile.get('instruction_detail_level', 'moderate')
        self.prefers_visual = user_profile.get('prefers_visual_over_text', True)
        self.use_voice = user_profile.get('wants_voice_guidance', False)
        
        # OpenAI for advanced NLP (optional)
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.use_ai_simplification = bool(self.openai_key)
    
    def generate_instructions(
        self,
        route: Dict[str, Any],
        current_location: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate turn-by-turn instructions adapted to user's needs
        """
        
        instructions = []
        step_number = 1
        
        for leg in route.get('legs', []):
            mode = leg.get('mode', 'walk')
            distance = leg.get('distance', 0)
            duration = leg.get('duration', '0 mins')
            raw_instructions = leg.get('instructions', '')
            
            # Generate instruction at appropriate detail level
            instruction = self._create_instruction(
                step_number=step_number,
                mode=mode,
                distance=distance,
                duration=duration,
                raw_instructions=raw_instructions
            )
            
            instructions.append(instruction)
            step_number += 1
        
        return instructions
    
    def _create_instruction(
        self,
        step_number: int,
        mode: str,
        distance: int,
        duration: str,
        raw_instructions: str
    ) -> Dict[str, Any]:
        """Create instruction at appropriate detail level"""
        
        # Simplify HTML instructions
        clean_instructions = self._clean_html(raw_instructions)
        
        # Generate three levels of detail
        simple = self._generate_simple_instruction(mode, distance, clean_instructions)
        moderate = self._generate_moderate_instruction(mode, distance, duration, clean_instructions)
        detailed = self._generate_detailed_instruction(mode, distance, duration, clean_instructions)
        
        # Select based on user preference
        if self.detail_level == 'minimal':
            primary_instruction = simple
        elif self.detail_level == 'detailed':
            primary_instruction = detailed
        else:
            primary_instruction = moderate
        
        return {
            'step_number': step_number,
            'mode': mode,
            'instruction_simple': simple,
            'instruction_moderate': moderate,
            'instruction_detailed': detailed,
            'primary_instruction': primary_instruction,
            'distance_meters': distance,
            'duration': duration,
            'icon': self._get_mode_icon(mode),
            'requires_attention': self._requires_attention(mode),
            'landmarks': self._extract_landmarks(clean_instructions)
        }
    
    def _generate_simple_instruction(self, mode: str, distance: int, instructions: str) -> str:
        """Minimal instruction - just the essential action"""
        
        if mode == 'transit':
            # Extract line/route number
            route_match = re.search(r'(Bus|Train|Metro|Tube)\s+(\w+)', instructions, re.IGNORECASE)
            if route_match:
                return f"Take {route_match.group(1)} {route_match.group(2)}"
            return "Take public transport"
        
        elif mode == 'walking':
            direction = self._extract_direction(instructions)
            if distance < 100:
                return f"Walk {direction}"
            else:
                return f"Walk {distance}m {direction}"
        
        elif mode == 'transfer':
            return "Change here"
        
        return instructions[:50]
    
    def _generate_moderate_instruction(self, mode: str, distance: int, duration: str, instructions: str) -> str:
        """Moderate detail - action + context"""
        
        if mode == 'transit':
            return f"{self._generate_simple_instruction(mode, distance, instructions)}. Travel for {duration}."
        
        elif mode == 'walking':
            direction = self._extract_direction(instructions)
            return f"Walk {distance}m {direction}. Takes about {duration}."
        
        return instructions
    
    def _generate_detailed_instruction(self, mode: str, distance: int, duration: str, instructions: str) -> str:
        """Detailed instruction - full context + tips"""
        
        base = self._generate_moderate_instruction(mode, distance, duration, instructions)
        
        # Add contextual tips
        tips = []
        
        if mode == 'transit':
            tips.append("Check the display for the correct platform.")
            tips.append("Listen for announcements.")
        
        elif mode == 'walking' and distance > 200:
            tips.append("Take your time, no need to rush.")
        
        elif mode == 'transfer':
            tips.append("Follow the signs for your next line.")
            tips.append("Ask staff if you need help.")
        
        if tips:
            return f"{base} Tips: {' '.join(tips)}"
        
        return base
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags and clean up text"""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', html_text)
        # Clean up extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def _extract_direction(self, instructions: str) -> str:
        """Extract direction from instructions"""
        directions = ['north', 'south', 'east', 'west', 'left', 'right', 'straight', 'ahead']
        
        for direction in directions:
            if direction in instructions.lower():
                return direction
        
        return 'forward'
    
    def _extract_landmarks(self, instructions: str) -> List[str]:
        """Extract landmark mentions from instructions"""
        # Common landmark patterns
        patterns = [
            r'(Station|Building|Park|Square|Bridge|Street|Road|Avenue)',
            r'([A-Z][a-z]+ (Station|Park|Square|Bridge))'
        ]
        
        landmarks = []
        for pattern in patterns:
            matches = re.findall(pattern, instructions)
            landmarks.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        return list(set(landmarks))[:3]  # Top 3 unique landmarks
    
    def _get_mode_icon(self, mode: str) -> str:
        """Get emoji icon for transport mode"""
        icons = {
            'walking': '🚶',
            'walk': '🚶',
            'transit': '🚇',
            'bus': '🚌',
            'train': '🚂',
            'metro': '🚇',
            'subway': '🚇',
            'transfer': '🔄',
            'bicycle': '🚴'
        }
        return icons.get(mode.lower(), '➡️')
    
    def _requires_attention(self, mode: str) -> bool:
        """Does this step require special attention?"""
        return mode.lower() in ['transfer', 'transit']
    
    def simplify_signage(self, detected_text: str) -> str:
        """
        Simplify complex signage text using NLP
        """
        
        # Remove clutter
        text = detected_text.strip()
        
        # Common simplifications
        simplifications = {
            'Please proceed to': 'Go to',
            'Platform Number': 'Platform',
            'Departing from': 'From',
            'Arriving at': 'To',
            'Exit via': 'Exit',
            'Entrance at': 'Enter at'
        }
        
        for old, new in simplifications.items():
            text = text.replace(old, new)
        
        # If OpenAI available, use AI simplification
        if self.use_ai_simplification:
            text = self._ai_simplify(text)
        
        return text
    
    def _ai_simplify(self, text: str) -> str:
        """Use OpenAI to simplify complex text"""
        
        try:
            import openai
            openai.api_key = self.openai_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Simplify this transit instruction to be clear and concise. Keep only essential information."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"AI simplification error: {e}")
            return text
    
    def answer_question(self, question: str, journey_context: Dict[str, Any]) -> str:
        """
        Answer user questions about their journey using conversational AI
        """
        
        question_lower = question.lower()
        
        # Pattern matching for common questions
        if any(word in question_lower for word in ['how long', 'duration', 'time']):
            total_duration = journey_context.get('total_duration_minutes', 0)
            return f"Your journey will take approximately {total_duration} minutes."
        
        elif any(word in question_lower for word in ['transfer', 'change']):
            transfer_count = journey_context.get('transfer_count', 0)
            if transfer_count == 0:
                return "This is a direct route with no transfers."
            elif transfer_count == 1:
                return "You'll need to make 1 transfer during this journey."
            else:
                return f"You'll make {transfer_count} transfers during this journey."
        
        elif any(word in question_lower for word in ['where am i', 'location', 'lost']):
            current_step = journey_context.get('current_step', 1)
            total_steps = journey_context.get('total_steps', 0)
            return f"You're on step {current_step} of {total_steps}. You're doing great!"
        
        elif any(word in question_lower for word in ['help', 'confused', 'dont understand']):
            return "No problem! I can repeat the instruction in simpler terms. Would you like me to break it down step by step?"
        
        elif any(word in question_lower for word in ['toilet', 'bathroom', 'restroom']):
            return "Look for blue bathroom signs. Most stations have facilities near the ticket office."
        
        elif any(word in question_lower for word in ['wheelchair', 'accessible', 'elevator', 'lift']):
            return "I'll check for accessible routes. Most major stations have elevators, but let me find the best accessible path for you."
        
        else:
            # Use AI for complex questions if available
            if self.use_ai_simplification:
                return self._ai_answer(question, journey_context)
            else:
                return "I can help with that! Could you rephrase your question? For example: 'How long will this take?' or 'Where do I transfer?'"
    
    def _ai_answer(self, question: str, context: Dict) -> str:
        """Use OpenAI to answer complex questions"""
        
        try:
            import openai
            openai.api_key = self.openai_key
            
            context_str = f"""
            Journey context:
            - Origin: {context.get('origin', 'Unknown')}
            - Destination: {context.get('destination', 'Unknown')}
            - Duration: {context.get('total_duration_minutes', 0)} minutes
            - Transfers: {context.get('transfer_count', 0)}
            - Current step: {context.get('current_step', 1)} of {context.get('total_steps', 0)}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly transit assistant. Give clear, concise answers. Be reassuring and supportive."},
                    {"role": "user", "content": f"{context_str}\n\nUser question: {question}"}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"AI answer error: {e}")
            return "I can help with that! Could you rephrase your question?"
    
    def get_visual_guidance(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visual guidance for a step
        Includes icons, colors, and layout suggestions for UI
        """
        
        mode = step.get('mode', 'walk')
        
        visual = {
            'icon': self._get_mode_icon(mode),
            'color': self._get_mode_color(mode),
            'size': 'large' if step.get('requires_attention') else 'medium',
            'animation': 'pulse' if step.get('requires_attention') else 'none',
            'landmarks': step.get('landmarks', []),
            'suggested_photo': self._suggest_photo_type(mode)
        }
        
        return visual
    
    def _get_mode_color(self, mode: str) -> str:
        """Get color for transport mode"""
        colors = {
            'walking': '#4CAF50',  # Green
            'transit': '#2196F3',  # Blue
            'bus': '#FF9800',      # Orange
            'train': '#9C27B0',    # Purple
            'metro': '#F44336',    # Red
            'transfer': '#FFC107'  # Amber
        }
        return colors.get(mode.lower(), '#607D8B')  # Grey default
    
    def _suggest_photo_type(self, mode: str) -> str:
        """Suggest what type of photo would be helpful"""
        if mode == 'transfer':
            return 'transfer_point'
        elif mode == 'transit':
            return 'platform_sign'
        elif mode == 'walking':
            return 'landmark'
        return 'none'
    
    def get_progress_update(
        self,
        current_step: int,
        total_steps: int,
        time_elapsed: int,
        time_remaining: int
    ) -> str:
        """
        Generate encouraging progress updates
        """
        
        progress_percent = (current_step / total_steps) * 100 if total_steps > 0 else 0
        
        if progress_percent < 25:
            message = f"You've started your journey! {time_remaining} minutes to go."
        elif progress_percent < 50:
            message = f"Almost halfway there! You're doing great. {time_remaining} minutes left."
        elif progress_percent < 75:
            message = f"More than halfway! Keep going. {time_remaining} minutes remaining."
        else:
            message = f"Almost there! Just {time_remaining} minutes left. You've got this!"
        
        return message


# Example usage
if __name__ == "__main__":
    # Test with a sample user profile
    user_profile = {
        'instruction_detail_level': 'detailed',
        'prefers_visual_over_text': True,
        'wants_voice_guidance': False
    }
    
    assistant = WayfindingAssistant(user_profile)
    
    # Sample route leg
    sample_leg = {
        'mode': 'transit',
        'distance': 5000,
        'duration': '15 mins',
        'instructions': 'Take <b>Bus 73</b> from <b>King\'s Cross Station</b> towards <b>Victoria</b>'
    }
    
    # Generate instructions
    instruction = assistant._create_instruction(
        step_number=1,
        mode=sample_leg['mode'],
        distance=sample_leg['distance'],
        duration=sample_leg['duration'],
        raw_instructions=sample_leg['instructions']
    )
    
    print("Simple:", instruction['instruction_simple'])
    print("Moderate:", instruction['instruction_moderate'])
    print("Detailed:", instruction['instruction_detailed'])
    print("Landmarks:", instruction['landmarks'])
