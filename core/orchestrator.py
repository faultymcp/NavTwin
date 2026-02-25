"""
Adaptive ML Orchestrator — Coordinates specialised agents through a
publish-subscribe messaging architecture.

The orchestrator fuses multi-agent insights using learned weight
configurations from the RL Agent, dynamically adjusting the influence
of different agents based on context.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from digital_twins.environment_dt import EnvironmentDigitalTwin, EnvironmentState
from digital_twins.personal_dt import PersonalDigitalTwin, JourneyRecord
from agents.route_scoring import RouteScorer, RouteFeatureExtractor
from agents.rl_agent import RLWeightAgent
from agents.supporting_agents import (
    StressDetectionAgent, StressSignals,
    GamificationEngine, PrivacyGuardian
)
import config as cfg


@dataclass
class NavigationQuery:
    """User's navigation request."""
    user_id: str
    origin: int
    destination: int
    time_of_day: float
    day_of_week: int
    expressed_concern: str = ""     # e.g., "worried about crowds"
    time_pressure: float = 30.0    # minutes until needed
    stress_signals: Optional[StressSignals] = None


@dataclass
class NavigationRecommendation:
    """System's recommendation to the user."""
    selected_route: List[int]
    route_locations: List[str]
    composite_score: float
    score_breakdown: Dict[str, float]
    all_candidates: List[Dict]
    weight_config: np.ndarray
    weight_config_id: int
    stress_assessment: Dict
    gamification_update: Dict
    privacy_check: Dict
    explanation: str
    estimated_time_min: float
    distance_m: float


class MessageBus:
    """
    Publish-subscribe message bus for agent coordination.

    Agents publish observations and insights to topic channels.
    Other agents subscribe to topics based on functional dependencies.
    """

    def __init__(self):
        self.channels: Dict[str, List] = {
            "user_query": [],
            "route_generated": [],
            "env_prediction": [],
            "stress_detected": [],
            "rl_weights_selected": [],
            "route_scored": [],
            "privacy_checked": [],
            "gamification_event": [],
            "orchestrator_decision": [],
        }
        self.message_log: List[Dict] = []

    def publish(self, channel: str, message: Dict):
        """Publish a message to a channel."""
        self.message_log.append({"channel": channel, "message": message})
        if channel in self.channels:
            self.channels[channel].append(message)

    def get_latest(self, channel: str) -> Optional[Dict]:
        """Get the most recent message on a channel."""
        if channel in self.channels and self.channels[channel]:
            return self.channels[channel][-1]
        return None

    def clear(self):
        """Clear all channels for next query cycle."""
        for channel in self.channels:
            self.channels[channel] = []


class AdaptiveMLOrchestrator:
    """
    Central coordinator that manages the complete navigation pipeline.

    Pipeline:
    1. Receive query → activate agents in parallel
    2. EDT generates environmental predictions
    3. PDT retrieves user profile
    4. Stress Detection assesses current state
    5. Route Planning generates candidates
    6. RL Agent selects weight configuration
    7. Route Scorer evaluates all candidates
    8. Orchestrator selects best route
    9. Gamification processes outcome
    10. Privacy Guardian audits data flow
    """

    def __init__(self, edt: EnvironmentDigitalTwin,
                 pdt: PersonalDigitalTwin,
                 seed: int = cfg.RANDOM_SEED):
        self.edt = edt
        self.pdt = pdt
        self.route_scorer = RouteScorer(edt, pdt)
        self.rl_agent = RLWeightAgent(seed=seed)
        self.stress_agent = StressDetectionAgent(seed=seed)
        self.gamification = GamificationEngine()
        self.privacy_guardian = PrivacyGuardian()
        self.message_bus = MessageBus()
        self.rng = np.random.RandomState(seed)

        self.query_count = 0
        self.decision_log: List[Dict] = []

    def process_query(self, query: NavigationQuery) -> NavigationRecommendation:
        """Process a navigation query through the full multi-agent pipeline."""
        self.query_count += 1
        self.message_bus.clear()

        # ── 1. Publish query ───────────────────────────────────────
        self.message_bus.publish("user_query", {
            "user_id": query.user_id,
            "origin": query.origin,
            "destination": query.destination,
            "concern": query.expressed_concern
        })

        # ── 2. EDT prediction (parallel in production) ─────────────
        env_state = self.edt.predict(query.time_of_day, query.day_of_week)
        self.message_bus.publish("env_prediction", {
            "timestamp": env_state.timestamp,
            "weather": env_state.weather,
            "high_crowd_areas": [
                nid for nid, preds in env_state.crowd_predictions.items()
                if preds[0] > 6.0
            ]
        })

        # ── 3. Stress assessment (parallel) ────────────────────────
        stress_signals = query.stress_signals or StressSignals(
            query_sentiment=self._infer_sentiment(query.expressed_concern),
            query_urgency=min(1.0, 30.0 / max(query.time_pressure, 1))
        )

        user_sens = self.pdt.sensory_profile
        avg_sensitivity = np.mean([user_sens.crowd_sensitivity,
                                    user_sens.noise_sensitivity,
                                    user_sens.visual_sensitivity]) / 10.0

        stress_context = {
            "crowd_level": np.mean([p[0] for p in env_state.crowd_predictions.values()]) / 10.0,
            "noise_level": np.mean([p[0] for p in env_state.noise_predictions.values()]) / 10.0,
            "user_sensitivity_mean": avg_sensitivity
        }
        stress_result = self.stress_agent.assess_stress(stress_signals, stress_context)
        self.message_bus.publish("stress_detected", stress_result)

        # ── 4. Route generation ────────────────────────────────────
        candidate_routes = self.edt.find_routes(
            query.origin, query.destination,
            max_routes=cfg.NUM_CANDIDATE_ROUTES
        )
        if not candidate_routes:
            # Fallback: direct path if no routes found
            candidate_routes = [[query.origin, query.destination]]

        self.message_bus.publish("route_generated", {
            "count": len(candidate_routes),
            "routes": candidate_routes
        })

        # ── 5. RL weight selection ─────────────────────────────────
        recent_journeys = self.pdt.journey_history[-7:]
        recent_success = sum(1 for j in recent_journeys if j.completed) / max(len(recent_journeys), 1)

        rl_context = {
            "stress_level": stress_result["stress_level"],
            "time_pressure": query.time_pressure,
            "temperature": env_state.weather.get("temperature", 15),
            "precipitation": env_state.weather.get("precipitation", 0),
            "wind": env_state.weather.get("wind", 5),
            "recent_success_rate": recent_success,
            "time_of_day": query.time_of_day,
            "day_of_week": query.day_of_week,
            "user_sensitivity_mean": avg_sensitivity,
            "journey_count": len(self.pdt.journey_history),
            "crowd_level": stress_context["crowd_level"],
            "noise_level": stress_context["noise_level"],
        }

        action_id, weights = self.rl_agent.select_action(rl_context)
        self.message_bus.publish("rl_weights_selected", {
            "action_id": action_id,
            "weights": {"w_PPS": weights[0], "w_ECS": weights[1],
                       "w_SPS": weights[2], "w_ES": weights[3]}
        })

        # ── 6. Score all routes ────────────────────────────────────
        scored_routes = self.route_scorer.score_all_routes(
            candidate_routes, env_state, weights
        )
        self.message_bus.publish("route_scored", {
            "best_score": scored_routes[0]["composite"] if scored_routes else 0,
            "candidates_scored": len(scored_routes)
        })

        # ── 7. Privacy audit ──────────────────────────────────────
        privacy_result = self.privacy_guardian.check_data_request({
            "type": "route_recommendation",
            "purpose": "route_recommendation",
            "fields_requested": ["sensory_profile", "journey_history", "location"],
            "necessary_fields": ["sensory_profile", "journey_history", "location"],
            "requires_pii": False
        })
        self.message_bus.publish("privacy_checked", privacy_result)

        # ── 8. Select best route ──────────────────────────────────
        best = scored_routes[0] if scored_routes else None
        if best is None:
            raise ValueError("No routes could be scored")

        # Generate explanation
        explanation = self._generate_explanation(best, stress_result, weights)

        # ── 9. Build recommendation ───────────────────────────────
        recommendation = NavigationRecommendation(
            selected_route=best["route"],
            route_locations=best["route_locations"],
            composite_score=best["composite"],
            score_breakdown={
                "PPS": best["pps"],
                "ECS": best["ecs"],
                "SPS": best["sps"],
                "ES": best["es"]
            },
            all_candidates=[{
                "route": s["route_locations"],
                "composite": round(s["composite"], 3),
                "pps": round(s["pps"], 3),
                "ecs": round(s["ecs"], 3),
                "sps": round(s["sps"], 3),
                "es": round(s["es"], 3),
            } for s in scored_routes],
            weight_config=weights,
            weight_config_id=action_id,
            stress_assessment=stress_result,
            gamification_update={},  # filled after journey completion
            privacy_check=privacy_result,
            explanation=explanation,
            estimated_time_min=best["estimated_time_min"],
            distance_m=best["distance_m"],
        )

        self.message_bus.publish("orchestrator_decision", {
            "selected_route": best["route_locations"],
            "score": best["composite"],
            "weight_config": action_id
        })

        # Log decision
        self.decision_log.append({
            "query_id": self.query_count,
            "origin": query.origin,
            "destination": query.destination,
            "action_id": action_id,
            "weights": weights.tolist(),
            "best_score": best["composite"],
            "stress_level": stress_result["stress_level"],
            "candidates": len(scored_routes),
        })

        return recommendation

    def process_journey_outcome(self, query: NavigationQuery,
                                 recommendation: NavigationRecommendation,
                                 outcome: Dict) -> Dict:
        """
        Process the outcome of a journey (acceptance, completion, stress).

        Updates all agents:
        - RL Agent: reward signal
        - PDT: journey record
        - Gamification: progress tracking
        """
        # Compute RL reward
        reward = self.rl_agent.compute_reward(outcome)

        # Build RL context for update
        avg_sensitivity = np.mean([
            self.pdt.sensory_profile.crowd_sensitivity,
            self.pdt.sensory_profile.noise_sensitivity,
            self.pdt.sensory_profile.visual_sensitivity
        ]) / 10.0

        rl_context = {
            "stress_level": recommendation.stress_assessment["stress_level"],
            "time_pressure": query.time_pressure,
            "temperature": 15, "precipitation": 0, "wind": 5,
            "recent_success_rate": 0.8,
            "time_of_day": query.time_of_day,
            "day_of_week": query.day_of_week,
            "user_sensitivity_mean": avg_sensitivity,
            "journey_count": len(self.pdt.journey_history),
            "crowd_level": 0.5, "noise_level": 0.5,
        }

        # Update RL agent
        self.rl_agent.update(rl_context, recommendation.weight_config_id, reward)

        # Record in PDT
        features = self.route_scorer.feature_extractor.extract(
            recommendation.selected_route,
            self.edt.predict(query.time_of_day, query.day_of_week)
        )
        journey_record = JourneyRecord(
            route_features=features,
            outcome=1 if outcome.get("accepted", True) else -1,
            completed=outcome.get("completed", False),
            stress_level=outcome.get("stress_level", 0.5),
            time_of_day=query.time_of_day,
            weather_conditions={},
            weight_config_used=recommendation.weight_config_id,
            route_score=recommendation.composite_score,
            timestamp=self.query_count
        )
        self.pdt.record_journey(journey_record)

        # Gamification
        gam_result = self.gamification.process_journey(
            query.user_id, outcome
        )

        return {
            "reward": reward,
            "rl_stats": self.rl_agent.get_stats(),
            "gamification": gam_result,
            "privacy_compliance": self.privacy_guardian.get_compliance_report(),
        }

    def _infer_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment for stress detection."""
        if not text:
            return 0.0
        negative = ["worried", "anxious", "scared", "nervous", "crowded",
                    "loud", "overwhelming", "stressed", "panic", "afraid"]
        positive = ["fine", "okay", "good", "calm", "relaxed", "happy"]
        text_lower = text.lower()
        neg_count = sum(1 for w in negative if w in text_lower)
        pos_count = sum(1 for w in positive if w in text_lower)
        if neg_count + pos_count == 0:
            return 0.0
        return (pos_count - neg_count) / (neg_count + pos_count)

    def _generate_explanation(self, route: Dict, stress: Dict,
                              weights: np.ndarray) -> str:
        """Generate human-readable explanation of route selection."""
        w_names = ["Personal Preference", "Environmental Comfort",
                   "Success Probability", "Efficiency"]
        dominant = w_names[np.argmax(weights)]

        parts = [f"Selected route via {' → '.join(route['route_locations'])}"]
        parts.append(f"(Score: {route['composite']:.3f}, {route['estimated_time_min']:.0f} min)")

        if stress["stress_level"] > 0.5:
            parts.append(f"Prioritising comfort due to elevated stress ({stress['stress_level']:.2f}).")
        else:
            parts.append(f"Optimising for {dominant} (weight: {max(weights):.2f}).")

        if route["ecs"] > 0.7:
            parts.append("This route avoids high crowd and noise areas.")
        if route["es"] > 0.8:
            parts.append("This is also one of the most efficient options.")

        return " ".join(parts)
