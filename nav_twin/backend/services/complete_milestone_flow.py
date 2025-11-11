"""
COMPLETE MILESTONE FLOW
Shows how SLM + CV work together during journey
"""

from typing import Dict, Any
from services.slm_service import SLMService
from services.cv_service import CVService, MockCVService


class CompleteMilestoneFlow:
    """
    Complete flow: User approaches milestone → CV analyzes → SLM guides → User completes → Rewards
    """
    
    def __init__(self, user_profile: Dict[str, Any]):
        self.profile = user_profile
        self.slm = SLMService()
        self.cv = CVService()
    
    def process_milestone_approach(
        self,
        milestone: Dict[str, Any],
        camera_image: str = None
    ) -> Dict[str, Any]:
        """
        User is approaching a milestone
        
        Flow:
        1. CV analyzes scene (crowd, brightness)
        2. SLM generates personalized instruction
        3. Return everything to mobile app
        """
        
        # Step 1: Analyze current scene with CV
        if camera_image:
            scene_analysis = self.cv.analyze_scene(camera_image)
        else:
            # Use mock for testing
            mock_cv = MockCVService()
            scene_analysis = mock_cv.analyze_scene()
        
        # Step 2: Extract current conditions
        current_conditions = {
            'crowd_level': self._crowd_count_to_level(scene_analysis['crowd_count']),
            'noise_level': self._brightness_to_noise_estimate(scene_analysis['brightness_level']),
            'visual_complexity': scene_analysis['visual_complexity']
        }
        
        # Step 3: Generate personalized instruction with SLM
        instruction = self.slm.generate_milestone_instruction(
            milestone=milestone,
            user_profile=self.profile,
            current_conditions=current_conditions
        )
        
        # Step 4: Combine everything
        return {
            'milestone': milestone,
            'ai_instruction': instruction,
            'scene_analysis': scene_analysis,
            'current_conditions': current_conditions,
            'ready_to_complete': True
        }
    
    def validate_milestone_completion(
        self,
        milestone: Dict[str, Any],
        completion_image: str = None
    ) -> Dict[str, Any]:
        """
        User claims they completed milestone
        Validate with CV if possible
        """
        
        # If milestone has expected features (bus number, platform, etc.)
        expected_features = milestone.get('transit_info', {})
        
        if completion_image and expected_features:
            validation = self.cv.validate_location(completion_image, expected_features)
            
            if validation['confirmed']:
                return {
                    'validated': True,
                    'confidence': validation['confidence'],
                    'message': '✅ Location confirmed! Well done!'
                }
            else:
                return {
                    'validated': False,
                    'confidence': validation['confidence'],
                    'message': '⚠️ Double-check your location'
                }
        else:
            # Trust user without validation
            return {
                'validated': True,
                'confidence': 1.0,
                'message': '✅ Milestone complete!'
            }
    
    def _crowd_count_to_level(self, count: int) -> int:
        """Convert person count to 1-5 scale"""
        if count < 2:
            return 1
        elif count < 5:
            return 2
        elif count < 8:
            return 3
        elif count < 12:
            return 4
        else:
            return 5
    
    def _brightness_to_noise_estimate(self, brightness: float) -> int:
        """
        Estimate noise from brightness
        (Bright public spaces are often noisy)
        """
        if brightness < 200:
            return 2  # Dark = quiet
        elif brightness < 400:
            return 3  # Normal
        else:
            return 4  # Bright = probably loud
```

---

## 🚀 **HOW IT ALL WORKS TOGETHER:**
```
USER APPROACHES MILESTONE
         ↓
📷 CAMERA captures scene
         ↓
🤖 CV ANALYSIS
   - Count people (YOLOv8)
   - Measure brightness
   - Check visual complexity
   → "8 people detected, 450 lux brightness"
         ↓
🧠 SLM GENERATES INSTRUCTION
   Input: milestone + CV data + user profile
   Output: "The platform has 8 people waiting (more crowded than you prefer).
            Put on your headphones. Board bus 243 through the back door.
            Find a seat near the middle - quieter than the back."
         ↓
📱 SHOW TO USER
         ↓
✅ USER COMPLETES MILESTONE
         ↓
📷 VALIDATION (Optional)
   - CV confirms location
   - "Yes, this is bus 243"
         ↓
🎮 AWARD POINTS + BADGES
   - Base: +15 points
   - Bonus: +10 (handled crowd despite sensitivity)
   - Badge: "💪 Crowd Champion"
         ↓
🧠 SLM ENCOURAGEMENT
   "Amazing! You handled a crowded situation. That took real courage!"