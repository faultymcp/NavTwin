"""
GAMIFICATION SERVICE
Tracks milestone completion and awards points/badges/achievements
Makes travel feel like a game, not a challenge
"""

from typing import Dict, List, Any
from datetime import datetime


class GamificationService:
    """
    Handles all gamification logic:
    - Milestone tracking
    - Points calculation
    - Badge awards
    - Achievement unlocks
    - Level progression
    """

    def __init__(self, user_profile: Dict[str, Any], user_stats: Dict[str, Any]):
        self.profile = user_profile
        self.stats = user_stats
        self.current_journey_points = 0
        self.completed_milestones: List[Dict[str, Any]] = []

    def start_journey(self, milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Start journey and show gamification preview (resilient to missing fields)."""

        total_points = sum((m.get("gamification") or {}).get("points", 0) for m in (milestones or []))

        # Add difficulty bonuses
        transfer_milestones = [m for m in (milestones or []) if m.get("type") == "transfer"]
        transfer_bonus = len(transfer_milestones) * 30

        # Personal challenge bonuses
        anxiety_bonus = self._calculate_anxiety_bonus(milestones or [])

        total_available = total_points + transfer_bonus + anxiety_bonus

        first_milestone = milestones[0] if milestones else None
        encouragement = self._get_start_message(milestones or [])

        return {
            "journey_started": True,
            "total_milestones": len(milestones or []),
            "total_points_available": total_available,
            "current_level": self.stats.get("level", 1),
            "points_to_next_level": self._points_to_next_level(),
            "first_milestone": first_milestone,
            "encouragement": encouragement,
        }

    def complete_milestone(
        self,
        milestone: Dict[str, Any],
        all_milestones: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        User completed a milestone!
        Calculate rewards and check for achievements
        """
        g = (milestone.get("gamification") or {})
        base_points = g.get("points", 10)

        # Bonus points for personal challenges
        bonus_points = 0
        bonus_reasons: List[str] = []

        # Transfer anxiety bonus
        if milestone.get("type") == "transfer":
            transfer_anxiety = self.profile.get("transfer_anxiety", 3.0)
            if transfer_anxiety >= 4.0:
                bonus_points += 20
                bonus_reasons.append("🌟 Overcame high transfer anxiety!")

        # Crowd sensitivity bonus
        if milestone.get("type") == "board_transit":
            crowd_sens = self.profile.get("crowd_sensitivity", 3.0)
            if crowd_sens >= 4.0:
                bonus_points += 15
                bonus_reasons.append("💪 Handled crowded situation!")

        total_points = base_points + bonus_points
        self.current_journey_points += total_points

        # Mark as completed
        milestone["completed"] = True
        milestone["completion_time"] = datetime.now().isoformat()
        self.completed_milestones.append(milestone)

        # Calculate progress
        completed_count = len(self.completed_milestones)
        total_count = max(len(all_milestones or []), completed_count)
        progress_pct = (completed_count / total_count) * 100 if total_count > 0 else 0.0

        # Check for achievements
        achievements = self._check_achievements(milestone, progress_pct, completed_count)

        # Check for level up
        new_total_points = self.stats.get("total_points", 0) + total_points
        current_level = self.stats.get("level", 1)
        new_level = self._calculate_level(new_total_points)
        level_up = new_level > current_level

        # Next milestone (guarded)
        next_milestone = None
        if completed_count < len(all_milestones or []):
            next_milestone = all_milestones[completed_count]

        # Journey complete?
        journey_complete = completed_count >= len(all_milestones or [])

        return {
            "milestone_completed": True,
            "milestone": milestone,
            "points_earned": total_points,
            "base_points": base_points,
            "bonus_points": bonus_points,
            "bonus_reasons": bonus_reasons,
            "total_journey_points": self.current_journey_points,
            "progress_percentage": round(progress_pct, 1),
            "completed_count": completed_count,
            "total_count": total_count,
            "encouragement": self._get_encouragement(progress_pct, milestone),
            "badges": g.get("badges", []),
            "achievements": achievements,
            "level_up": level_up,
            "new_level": new_level if level_up else current_level,
            "next_milestone": next_milestone,
            "journey_complete": journey_complete,
        }

    def _calculate_anxiety_bonus(self, milestones: List[Dict[str, Any]]) -> int:
        """Calculate bonus points based on user's specific anxieties."""
        bonus = 0
        transfer_anxiety = self.profile.get("transfer_anxiety", 3.0)
        transfer_count = sum(1 for m in milestones if m.get("type") == "transfer")
        if transfer_anxiety >= 4.0 and transfer_count > 0:
            bonus += transfer_count * 25
        return bonus

    def _check_achievements(
        self,
        milestone: Dict[str, Any],
        progress_pct: float,
        completed_count: int,
    ) -> List[Dict[str, Any]]:
        """Check if any achievements were unlocked."""
        achievements: List[Dict[str, Any]] = []

        # First milestone
        if completed_count == 1:
            achievements.append({
                "id": "first_step",
                "title": "🎯 First Step",
                "description": "Completed your first milestone!",
                "points_bonus": 25,
                "rarity": "common",
            })

        # Halfway point
        if 48 <= progress_pct <= 52:
            achievements.append({
                "id": "halfway_hero",
                "title": "⚡ Halfway Hero",
                "description": "You're halfway through!",
                "points_bonus": 30,
                "rarity": "uncommon",
            })

        # Transfer achievement
        if milestone.get("type") == "transfer":
            transfer_anxiety = self.profile.get("transfer_anxiety", 3.0)
            if transfer_anxiety >= 4.0:
                achievements.append({
                    "id": "anxiety_conqueror",
                    "title": "💪 Anxiety Conqueror",
                    "description": "Completed transfer despite high anxiety!",
                    "points_bonus": 50,
                    "rarity": "rare",
                })

        # Add achievement points to journey
        for a in achievements:
            self.current_journey_points += a["points_bonus"]

        return achievements

    def _calculate_level(self, total_points: int) -> int:
        """Calculate level from total points."""
        return (total_points // 200) + 1

    def _points_to_next_level(self) -> int:
        """How many points until next level."""
        current_points = self.stats.get("total_points", 0)
        current_level = self._calculate_level(current_points)
        next_level_points = current_level * 200
        return next_level_points - current_points

    def _get_start_message(self, milestones: List[Dict[str, Any]]) -> str:
        """Get encouraging start message."""
        transfer_count = sum(1 for m in milestones if m.get("type") == "transfer")
        transfer_anxiety = self.profile.get("transfer_anxiety", 3.0)
        if transfer_count > 0 and transfer_anxiety >= 4.0:
            return f"🌟 I know {transfer_count} transfer(s) can feel daunting, but I'll guide you through each step. You've got this!"
        elif transfer_count > 1:
            return f"💪 {transfer_count} transfers ahead - but we'll take them one at a time!"
        else:
            return "✨ Let's do this! One milestone at a time. You're ready!"

    def _get_encouragement(self, progress_pct: float, milestone: Dict[str, Any]) -> str:
        """Get encouraging message based on progress."""
        if progress_pct >= 90:
            return "🎉 SO CLOSE! You're almost there!"
        elif progress_pct >= 75:
            return "🔥 Incredible progress! Keep it up!"
        elif progress_pct >= 50:
            return "💪 Halfway done! You're crushing it!"
        elif progress_pct >= 25:
            return "✨ Great start! Every step counts!"
        else:
            return "🚀 You're doing amazing! Keep going!"

    def get_journey_summary(self) -> Dict[str, Any]:
        """Get complete journey summary with stats."""
        total_milestones = len(self.completed_milestones)
        transfer_count = sum(1 for m in self.completed_milestones if m.get("type") == "transfer")
        if transfer_count >= 2:
            difficulty = "🔥 EXPERT LEVEL"
        elif transfer_count == 1:
            difficulty = "⭐ INTERMEDIATE"
        else:
            difficulty = "✨ BEGINNER"

        if self.current_journey_points >= 150:
            final_message = "🏆 LEGENDARY! You absolutely dominated this journey!"
        elif self.current_journey_points >= 100:
            final_message = "🌟 AMAZING! You're a navigation master!"
        else:
            final_message = "✨ GREAT JOB! You completed the journey!"

        new_total = self.stats.get("total_points", 0) + self.current_journey_points
        return {
            "journey_complete": True,
            "total_milestones": total_milestones,
            "total_points_earned": self.current_journey_points,
            "difficulty_rating": difficulty,
            "final_message": final_message,
            "new_total_points": new_total,
            "new_level": self._calculate_level(new_total),
        }
