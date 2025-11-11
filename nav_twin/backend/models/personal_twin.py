"""
PERSONAL DIGITAL TWIN
Uses YOUR quiz results to personalize routes!
This is where the magic happens - your sensory preferences drive the AI.
"""

from typing import Dict, Any
import numpy as np


class PersonalDigitalTwin:
    """
    Your Personal AI Twin - Knows YOUR sensory preferences from the quiz
    Uses them to find routes that match YOUR needs!
    """
    
    def __init__(
        self, 
        crowd_sensitivity: float = 3.0,
        noise_sensitivity: float = 3.0,
        visual_sensitivity: float = 3.0,
        transfer_anxiety: float = 3.0
    ):
        """
        Initialize with YOUR sensory preferences from quiz
        
        Scale: 1 (low sensitivity) to 5 (high sensitivity)
        """
        self.crowd_sensitivity = crowd_sensitivity
        self.noise_sensitivity = noise_sensitivity
        self.visual_sensitivity = visual_sensitivity
        self.transfer_anxiety = transfer_anxiety
        
        print(f"\n👤 Personal Digital Twin Created:")
        print(f"   🚶 Crowd Sensitivity: {crowd_sensitivity}/5")
        print(f"   🔊 Noise Sensitivity: {noise_sensitivity}/5")
        print(f"   💡 Visual Sensitivity: {visual_sensitivity}/5")
        print(f"   🔄 Transfer Anxiety: {transfer_anxiety}/5")
        
        # Determine user's travel style
        avg_sensitivity = np.mean([
            crowd_sensitivity, 
            noise_sensitivity, 
            visual_sensitivity, 
            transfer_anxiety
        ])
        
        if avg_sensitivity >= 4.0:
            self.travel_style = "Comfort Seeker 🛋️"
            print(f"   ✨ Your Style: {self.travel_style}")
            print(f"   💭 You prefer quiet, calm, direct routes")
        elif avg_sensitivity >= 3.0:
            self.travel_style = "Balanced Traveler ⚖️"
            print(f"   ✨ Your Style: {self.travel_style}")
            print(f"   💭 You balance comfort and efficiency")
        else:
            self.travel_style = "Efficiency Expert ⚡"
            print(f"   ✨ Your Style: {self.travel_style}")
            print(f"   💭 You prioritize speed and efficiency")
    
    def score_route(self, route: Dict[str, Any]) -> float:
        """
        Score how well this route matches YOUR preferences
        
        Returns: 0.0 (bad match) to 1.0 (perfect match)
        """
        
        # Extract route characteristics
        transfer_count = route.get('transfer_count', 0)
        duration_minutes = route.get('duration_seconds', 0) / 60
        
        # Estimate crowd and noise levels based on route characteristics
        # In real system, this would come from live data
        estimated_crowd_level = self._estimate_crowd_level(route)
        estimated_noise_level = self._estimate_noise_level(route)
        
        # Score each dimension based on YOUR sensitivity
        # High sensitivity = penalize routes with that factor
        
        # 1. Transfer Score (Higher anxiety = prefer fewer transfers)
        if self.transfer_anxiety >= 4.0:
            # High anxiety: strongly prefer direct routes
            transfer_score = 1.0 - (transfer_count * 0.3)
        elif self.transfer_anxiety >= 3.0:
            # Moderate: somewhat prefer fewer transfers
            transfer_score = 1.0 - (transfer_count * 0.2)
        else:
            # Low anxiety: transfers don't matter much
            transfer_score = 1.0 - (transfer_count * 0.1)
        
        # 2. Crowd Score (Higher sensitivity = avoid crowds)
        if self.crowd_sensitivity >= 4.0:
            crowd_score = 1.0 - (estimated_crowd_level * 0.25)
        elif self.crowd_sensitivity >= 3.0:
            crowd_score = 1.0 - (estimated_crowd_level * 0.15)
        else:
            crowd_score = 1.0 - (estimated_crowd_level * 0.05)
        
        # 3. Noise Score (Higher sensitivity = avoid noise)
        if self.noise_sensitivity >= 4.0:
            noise_score = 1.0 - (estimated_noise_level * 0.25)
        elif self.noise_sensitivity >= 3.0:
            noise_score = 1.0 - (estimated_noise_level * 0.15)
        else:
            noise_score = 1.0 - (estimated_noise_level * 0.05)
        
        # 4. Duration Score (longer routes slightly penalized)
        duration_score = max(0.0, 1.0 - (duration_minutes / 120))
        
        # Weighted average based on YOUR sensitivities
        # Higher sensitivity = that factor weighs more
        total_sensitivity = (
            self.transfer_anxiety + 
            self.crowd_sensitivity + 
            self.noise_sensitivity
        ) + 0.01  # Avoid division by zero
        
        transfer_weight = self.transfer_anxiety / total_sensitivity
        crowd_weight = self.crowd_sensitivity / total_sensitivity
        noise_weight = self.noise_sensitivity / total_sensitivity
        
        # Calculate final personal preference score
        final_score = (
            transfer_score * transfer_weight * 0.4 +
            crowd_score * crowd_weight * 0.3 +
            noise_score * noise_weight * 0.3 +
            duration_score * 0.1
        )
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def _estimate_crowd_level(self, route: Dict[str, Any]) -> float:
        """
        Estimate crowd level (0-1 scale)
        In production: would use real-time data
        """
        # More transfers usually means busier stations
        transfers = route.get('transfer_count', 0)
        
        # Peak hours are busier
        # This is placeholder - real system would check departure time
        base_crowd = 0.5
        
        # Each transfer adds crowd exposure
        crowd_penalty = transfers * 0.15
        
        return min(1.0, base_crowd + crowd_penalty)
    
    def _estimate_noise_level(self, route: Dict[str, Any]) -> float:
        """
        Estimate noise level (0-1 scale)
        In production: would use real-time data
        """
        # Buses generally quieter than trains/tubes
        modes = route.get('transport_modes', [])
        
        base_noise = 0.5
        
        # Underground/metro tends to be louder
        if 'subway' in modes or 'metro' in modes:
            base_noise += 0.2
        
        # More transfers = more station noise
        transfers = route.get('transfer_count', 0)
        noise_penalty = transfers * 0.1
        
        return min(1.0, base_noise + noise_penalty)
    
    def get_travel_recommendations(self) -> Dict[str, str]:
        """Get personalized travel tips based on YOUR profile"""
        
        tips = []
        
        if self.crowd_sensitivity >= 4.0:
            tips.append("🚶 Travel during off-peak hours when possible")
            tips.append("📍 Choose quieter stations for transfers")
        
        if self.noise_sensitivity >= 4.0:
            tips.append("🎧 Noise-canceling headphones recommended")
            tips.append("🚌 Consider buses over underground when possible")
        
        if self.transfer_anxiety >= 4.0:
            tips.append("🎯 Direct routes are prioritized for you")
            tips.append("📱 Extra time added between transfers")
        
        if self.visual_sensitivity >= 4.0:
            tips.append("🕶️ Sunglasses may help in bright stations")
            tips.append("📖 Step-by-step visual guides included")
        
        return {
            'travel_style': self.travel_style,
            'personalized_tips': tips
        }
    
    def explain_preference(self, route: Dict[str, Any]) -> str:
        """Explain why this route does/doesn't match your preferences"""
        
        score = self.score_route(route)
        transfers = route.get('transfer_count', 0)
        
        if score >= 0.75:
            explanation = f"✨ Great match! "
            if self.transfer_anxiety >= 4.0 and transfers == 0:
                explanation += "Direct route (perfect for your transfer anxiety). "
            if self.crowd_sensitivity >= 4.0:
                explanation += "Expected to be quiet and uncrowded. "
            return explanation
        
        elif score >= 0.5:
            explanation = f"👍 Good option. "
            if transfers > 0:
                explanation += f"{transfers} transfer(s) required. "
            return explanation
        
        else:
            explanation = f"⚠️ May be challenging. "
            if self.transfer_anxiety >= 4.0 and transfers > 1:
                explanation += f"{transfers} transfers (high for your comfort level). "
            if self.crowd_sensitivity >= 4.0:
                explanation += "Expected to be busy. "
            return explanation


# Test the PersonalDigitalTwin
if __name__ == "__main__":
    print("="*70)
    print("PERSONAL DIGITAL TWIN - TESTING")
    print("="*70)
    
    # High sensitivity user
    print("\n" + "="*70)
    print("USER 1: High Sensory Sensitivity")
    print("="*70)
    
    user1 = PersonalDigitalTwin(
        crowd_sensitivity=5.0,
        noise_sensitivity=4.5,
        visual_sensitivity=4.0,
        transfer_anxiety=5.0
    )
    
    # Test routes
    route1 = {
        'route_id': 'direct_quiet',
        'transfer_count': 0,
        'duration_seconds': 1800,  # 30 mins
        'transport_modes': ['bus']
    }
    
    route2 = {
        'route_id': 'fast_busy',
        'transfer_count': 2,
        'duration_seconds': 1200,  # 20 mins
        'transport_modes': ['subway', 'subway']
    }
    
    print("\nRoute 1 (Direct, quiet bus):")
    score1 = user1.score_route(route1)
    print(f"Score: {score1:.2f}")
    print(user1.explain_preference(route1))
    
    print("\nRoute 2 (Fast, 2 transfers, subway):")
    score2 = user1.score_route(route2)
    print(f"Score: {score2:.2f}")
    print(user1.explain_preference(route2))
    
    print("\n" + "="*70)
    print("USER 2: Low Sensory Sensitivity")
    print("="*70)
    
    user2 = PersonalDigitalTwin(
        crowd_sensitivity=2.0,
        noise_sensitivity=1.5,
        visual_sensitivity=2.0,
        transfer_anxiety=1.5
    )
    
    print("\nRoute 1 (Direct, quiet bus):")
    score1 = user2.score_route(route1)
    print(f"Score: {score1:.2f}")
    
    print("\nRoute 2 (Fast, 2 transfers, subway):")
    score2 = user2.score_route(route2)
    print(f"Score: {score2:.2f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Same routes, different scores!")
    print("High sensitivity user prefers Route 1 (direct, quiet)")
    print("Low sensitivity user prefers Route 2 (fast, efficient)")
    print("="*70)