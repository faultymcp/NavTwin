"""
Simulation Engine — Generates diverse user scenarios and models
realistic journey outcomes for evaluation.

Creates multiple user profiles with different sensory profiles and
simulates navigation interactions to evaluate adaptation and
personalisation effectiveness.
"""
import numpy as np
from typing import Dict, List, Tuple
from digital_twins.personal_dt import PersonalDigitalTwin, SensoryProfile
from digital_twins.environment_dt import EnvironmentDigitalTwin
from core.orchestrator import AdaptiveMLOrchestrator, NavigationQuery
from agents.supporting_agents import StressSignals
import config as cfg


# ── User Scenario Definitions ─────────────────────────────────────

SCENARIOS = {
    "sarah": {
        "description": "Sarah — University student, high crowd/noise sensitivity, "
                       "prefers quiet park routes. Morning anxiety pattern.",
        "profile": SensoryProfile(
            crowd_sensitivity=8.2,
            noise_sensitivity=7.5,
            visual_sensitivity=4.0,
            social_interaction_avoidance=7.0,
            transition_difficulty=6.0,
            time_pressure_tolerance=3.0,
            preferred_route_type="quiet",
            communication_style="minimal"
        ),
        "typical_queries": [
            {"origin": 0, "destination": 2, "concern": "worried about crowds at the station",
             "time_pressure": 40, "time_of_day": 8.5, "day_of_week": 1},
            {"origin": 5, "destination": 9, "concern": "",
             "time_pressure": 30, "time_of_day": 12.0, "day_of_week": 1},
            {"origin": 12, "destination": 0, "concern": "running late, a bit stressed",
             "time_pressure": 15, "time_of_day": 17.5, "day_of_week": 2},
            {"origin": 7, "destination": 3, "concern": "",
             "time_pressure": 45, "time_of_day": 10.0, "day_of_week": 3},
            {"origin": 2, "destination": 5, "concern": "the mall area is usually loud",
             "time_pressure": 35, "time_of_day": 14.0, "day_of_week": 4},
        ],
        "stress_baseline": 0.45,
        "acceptance_model": {
            "comfort_weight": 0.7,
            "efficiency_weight": 0.3,
            "morning_anxiety_boost": 0.15,  # higher stress in mornings
        }
    },
    "alex": {
        "description": "Alex — Software developer, moderate sensitivities, values "
                       "efficiency but needs visual simplicity. Consistent patterns.",
        "profile": SensoryProfile(
            crowd_sensitivity=4.5,
            noise_sensitivity=5.0,
            visual_sensitivity=7.5,
            social_interaction_avoidance=4.0,
            transition_difficulty=3.0,
            time_pressure_tolerance=7.0,
            preferred_route_type="direct",
            communication_style="detailed"
        ),
        "typical_queries": [
            {"origin": 12, "destination": 10, "concern": "",
             "time_pressure": 20, "time_of_day": 9.0, "day_of_week": 0},
            {"origin": 10, "destination": 3, "concern": "need to get there quickly",
             "time_pressure": 10, "time_of_day": 12.5, "day_of_week": 0},
            {"origin": 3, "destination": 12, "concern": "",
             "time_pressure": 45, "time_of_day": 18.0, "day_of_week": 1},
            {"origin": 5, "destination": 14, "concern": "",
             "time_pressure": 30, "time_of_day": 8.0, "day_of_week": 2},
            {"origin": 14, "destination": 7, "concern": "don't mind taking the main road",
             "time_pressure": 25, "time_of_day": 16.0, "day_of_week": 3},
        ],
        "stress_baseline": 0.25,
        "acceptance_model": {
            "comfort_weight": 0.3,
            "efficiency_weight": 0.7,
            "morning_anxiety_boost": 0.0,
        }
    },
    "maya": {
        "description": "Maya — Artist, extreme noise sensitivity but loves exploring "
                       "new areas. Variable stress depending on time of day.",
        "profile": SensoryProfile(
            crowd_sensitivity=5.5,
            noise_sensitivity=9.0,
            visual_sensitivity=2.0,
            social_interaction_avoidance=3.0,
            transition_difficulty=4.0,
            time_pressure_tolerance=5.0,
            preferred_route_type="scenic",
            communication_style="visual"
        ),
        "typical_queries": [
            {"origin": 9, "destination": 16, "concern": "",
             "time_pressure": 50, "time_of_day": 10.0, "day_of_week": 5},
            {"origin": 2, "destination": 8, "concern": "avoid noisy areas please",
             "time_pressure": 30, "time_of_day": 13.0, "day_of_week": 5},
            {"origin": 11, "destination": 15, "concern": "",
             "time_pressure": 40, "time_of_day": 15.0, "day_of_week": 6},
            {"origin": 7, "destination": 0, "concern": "the station will be packed",
             "time_pressure": 20, "time_of_day": 17.0, "day_of_week": 0},
            {"origin": 0, "destination": 9, "concern": "",
             "time_pressure": 35, "time_of_day": 11.0, "day_of_week": 1},
        ],
        "stress_baseline": 0.35,
        "acceptance_model": {
            "comfort_weight": 0.5,
            "efficiency_weight": 0.3,
            "morning_anxiety_boost": 0.05,
            "noise_critical": True,  # will reject routes through noisy areas
        }
    },
    "james": {
        "description": "James — Retired teacher, moderate all-round sensitivities, "
                       "strong preference for familiar routes. Low tech comfort.",
        "profile": SensoryProfile(
            crowd_sensitivity=6.0,
            noise_sensitivity=6.0,
            visual_sensitivity=6.0,
            social_interaction_avoidance=5.0,
            transition_difficulty=8.0,
            time_pressure_tolerance=8.0,
            preferred_route_type="familiar",
            communication_style="detailed"
        ),
        "typical_queries": [
            {"origin": 7, "destination": 3, "concern": "",
             "time_pressure": 60, "time_of_day": 10.0, "day_of_week": 2},
            {"origin": 3, "destination": 2, "concern": "prefer my usual route",
             "time_pressure": 45, "time_of_day": 14.0, "day_of_week": 2},
            {"origin": 2, "destination": 7, "concern": "",
             "time_pressure": 40, "time_of_day": 16.0, "day_of_week": 3},
            {"origin": 7, "destination": 11, "concern": "",
             "time_pressure": 50, "time_of_day": 11.0, "day_of_week": 4},
            {"origin": 0, "destination": 7, "concern": "the station is always overwhelming",
             "time_pressure": 30, "time_of_day": 9.0, "day_of_week": 5},
        ],
        "stress_baseline": 0.30,
        "acceptance_model": {
            "comfort_weight": 0.5,
            "efficiency_weight": 0.2,
            "familiarity_weight": 0.3,
            "morning_anxiety_boost": 0.0,
        }
    },
}


class JourneyOutcomeModel:
    """
    Simulates realistic journey outcomes based on route characteristics,
    user profile, and environmental conditions.

    This replaces real user feedback for evaluation purposes.
    """

    def __init__(self, seed: int = cfg.RANDOM_SEED):
        self.rng = np.random.RandomState(seed)

    def simulate_outcome(self, scenario_key: str, scenario: Dict,
                         recommendation, query: NavigationQuery) -> Dict:
        """Simulate whether the user accepts, completes, and their stress."""
        model = scenario["acceptance_model"]
        profile = scenario["profile"]

        scores = recommendation.score_breakdown

        # ── Acceptance decision ────────────────────────────────────
        # User evaluates based on their personal priorities
        comfort_score = model["comfort_weight"] * (
            scores["PPS"] * 0.4 + scores["ECS"] * 0.6
        )
        efficiency_score = model["efficiency_weight"] * scores["ES"]
        familiarity_bonus = model.get("familiarity_weight", 0) * 0.5

        # Success probability also influences acceptance (user can sense route quality)
        sps_influence = 0.15 * scores["SPS"]

        acceptance_prob = comfort_score + efficiency_score + familiarity_bonus + sps_influence

        # Morning anxiety effect
        if query.time_of_day < 10:
            acceptance_prob -= model.get("morning_anxiety_boost", 0)

        # Noise-critical users reject noisy routes
        if model.get("noise_critical", False) and scores["ECS"] < 0.4:
            acceptance_prob -= 0.3

        # High composite scores should generally be accepted more
        acceptance_prob += 0.1 * (scores["PPS"] + scores["ECS"]) / 2

        accepted = self.rng.random() < np.clip(acceptance_prob + 0.2, 0.1, 0.95)

        if not accepted:
            return {
                "accepted": False,
                "completed": False,
                "abandoned": False,
                "stress_level": scenario["stress_baseline"] + 0.1,
            }

        # ── Completion simulation ──────────────────────────────────
        completion_prob = scores["SPS"]
        # Route comfort strongly affects completion for sensitive users
        if model["comfort_weight"] > 0.4:
            completion_prob += 0.15 * scores["ECS"]
        # Efficiency matters for time-pressured users
        if model["efficiency_weight"] > 0.4:
            completion_prob += 0.1 * scores["ES"]
        # Stress affects completion
        stress = self._simulate_stress(scenario, scores, query)
        if stress > 0.7:
            completion_prob -= 0.25
        if stress > 0.9:
            completion_prob -= 0.3

        completed = self.rng.random() < np.clip(completion_prob + 0.1, 0.1, 0.95)

        return {
            "accepted": True,
            "completed": completed,
            "abandoned": not completed,
            "stress_level": stress if completed else min(1.0, stress + 0.2),
            "new_route": self.rng.random() < 0.3,
        }

    def _simulate_stress(self, scenario: Dict, scores: Dict,
                         query: NavigationQuery) -> float:
        """Simulate post-journey stress level."""
        baseline = scenario["stress_baseline"]

        # Comfort score reduces stress
        comfort_effect = -0.3 * scores["ECS"]

        # Time pressure increases stress
        pressure_effect = 0.2 * max(0, 1 - query.time_pressure / 30)

        # Morning anxiety
        morning_effect = 0.1 if query.time_of_day < 10 else 0

        # Concern text suggests elevated starting stress
        concern_effect = 0.1 if query.expressed_concern else 0

        stress = baseline + comfort_effect + pressure_effect + morning_effect + concern_effect
        stress += self.rng.normal(0, 0.05)  # noise

        return np.clip(stress, 0, 1)


class SimulationEngine:
    """
    Runs full simulation across all user scenarios.

    For each user, simulates N journeys and tracks:
    - RL adaptation over time
    - Route quality improvement
    - Stress levels
    - Acceptance and completion rates
    """

    def __init__(self, n_journeys: int = cfg.NUM_SIMULATION_JOURNEYS,
                 seed: int = cfg.RANDOM_SEED):
        self.n_journeys = n_journeys
        self.seed = seed
        self.outcome_model = JourneyOutcomeModel(seed=seed)
        self.results: Dict[str, Dict] = {}

    def run_scenario(self, scenario_key: str, verbose: bool = False) -> Dict:
        """Run full simulation for a single user scenario."""
        scenario = SCENARIOS[scenario_key]
        seed = self.seed + hash(scenario_key) % 10000

        # Initialize components
        edt = EnvironmentDigitalTwin(seed=seed)
        pdt = PersonalDigitalTwin(
            user_id=scenario_key,
            sensory_profile=scenario["profile"],
            seed=seed
        )
        orchestrator = AdaptiveMLOrchestrator(edt, pdt, seed=seed)

        # Tracking
        journey_log = []
        reward_over_time = []
        weight_over_time = []
        score_over_time = []
        stress_over_time = []
        acceptance_over_time = []
        completion_over_time = []
        epsilon_over_time = []

        queries = scenario["typical_queries"]

        for j in range(self.n_journeys):
            # Cycle through typical queries with variation
            base_query = queries[j % len(queries)].copy()

            # Add variation
            rng = np.random.RandomState(seed + j)
            base_query["time_of_day"] += rng.normal(0, 0.5)
            base_query["time_of_day"] = np.clip(base_query["time_of_day"], 6, 22)
            base_query["time_pressure"] += rng.normal(0, 5)
            base_query["time_pressure"] = max(5, base_query["time_pressure"])

            query = NavigationQuery(
                user_id=scenario_key,
                origin=base_query["origin"],
                destination=base_query["destination"],
                time_of_day=base_query["time_of_day"],
                day_of_week=base_query.get("day_of_week", j % 7),
                expressed_concern=base_query.get("concern", ""),
                time_pressure=base_query["time_pressure"],
            )

            try:
                # Get recommendation
                rec = orchestrator.process_query(query)

                # Simulate outcome
                outcome = self.outcome_model.simulate_outcome(
                    scenario_key, scenario, rec, query
                )

                # Process outcome (updates RL, PDT, gamification)
                result = orchestrator.process_journey_outcome(query, rec, outcome)

                # Log
                journey_log.append({
                    "journey": j + 1,
                    "origin": edt.locations[query.origin].name,
                    "destination": edt.locations[query.destination].name,
                    "route": rec.route_locations,
                    "composite_score": rec.composite_score,
                    "pps": rec.score_breakdown["PPS"],
                    "ecs": rec.score_breakdown["ECS"],
                    "sps": rec.score_breakdown["SPS"],
                    "es": rec.score_breakdown["ES"],
                    "weight_config": int(rec.weight_config_id),
                    "weights": rec.weight_config.tolist(),
                    "accepted": outcome["accepted"],
                    "completed": outcome.get("completed", False),
                    "stress": outcome.get("stress_level", 0.5),
                    "reward": result["reward"],
                })

                reward_over_time.append(result["reward"])
                weight_over_time.append(rec.weight_config.tolist())
                score_over_time.append(rec.composite_score)
                stress_over_time.append(outcome.get("stress_level", 0.5))
                acceptance_over_time.append(1 if outcome["accepted"] else 0)
                completion_over_time.append(1 if outcome.get("completed", False) else 0)
                epsilon_over_time.append(orchestrator.rl_agent.epsilon)

                if verbose and (j + 1) % 25 == 0:
                    recent_acc = np.mean(acceptance_over_time[-25:])
                    recent_comp = np.mean(completion_over_time[-25:])
                    recent_stress = np.mean(stress_over_time[-25:])
                    recent_reward = np.mean(reward_over_time[-25:])
                    print(f"  Journey {j+1:3d}/{self.n_journeys}: "
                          f"Acc={recent_acc:.2f} Comp={recent_comp:.2f} "
                          f"Stress={recent_stress:.2f} Reward={recent_reward:.2f} "
                          f"ε={orchestrator.rl_agent.epsilon:.3f}")

            except Exception as e:
                if verbose:
                    print(f"  Journey {j+1}: Error - {e}")
                continue

        # Compute summary metrics
        result = {
            "scenario": scenario_key,
            "description": scenario["description"],
            "n_journeys": len(journey_log),
            "journey_log": journey_log,
            "metrics": self._compute_metrics(journey_log, reward_over_time,
                                              stress_over_time, acceptance_over_time,
                                              completion_over_time),
            "time_series": {
                "rewards": reward_over_time,
                "weights": weight_over_time,
                "scores": score_over_time,
                "stress": stress_over_time,
                "acceptance": acceptance_over_time,
                "completion": completion_over_time,
                "epsilon": epsilon_over_time,
            },
            "rl_stats": orchestrator.rl_agent.get_stats(),
            "gamification": orchestrator.gamification.get_or_create_state(scenario_key).__dict__,
            "privacy": orchestrator.privacy_guardian.get_compliance_report(),
        }

        self.results[scenario_key] = result
        return result

    def run_all(self, verbose: bool = True) -> Dict:
        """Run all scenarios and return combined results."""
        if verbose:
            print("=" * 70)
            print("NavTwin Multi-Agent Digital Twin — Simulation Suite")
            print("=" * 70)

        for key in SCENARIOS:
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"Scenario: {SCENARIOS[key]['description'][:60]}...")
                print(f"{'─' * 60}")
            self.run_scenario(key, verbose=verbose)

        return self.results

    def run_baseline_comparison(self, scenario_key: str,
                                 static_weights: np.ndarray = None) -> Dict:
        """
        Run the same scenario with static (non-adaptive) weights
        for comparison with RL-adaptive weights.
        """
        if static_weights is None:
            static_weights = np.array([0.25, 0.25, 0.25, 0.25])

        scenario = SCENARIOS[scenario_key]
        seed = self.seed + hash(scenario_key + "_baseline") % 10000

        edt = EnvironmentDigitalTwin(seed=seed)
        pdt = PersonalDigitalTwin(
            user_id=scenario_key + "_baseline",
            sensory_profile=scenario["profile"],
            seed=seed
        )
        scorer = __import__('agents.route_scoring', fromlist=['RouteScorer']).RouteScorer(edt, pdt)
        outcome_model = JourneyOutcomeModel(seed=seed)

        queries = scenario["typical_queries"]
        scores, stresses, accepted, completed = [], [], [], []

        for j in range(self.n_journeys):
            base_query = queries[j % len(queries)].copy()
            rng = np.random.RandomState(seed + j)
            base_query["time_of_day"] += rng.normal(0, 0.5)
            base_query["time_of_day"] = np.clip(base_query["time_of_day"], 6, 22)
            base_query["time_pressure"] += rng.normal(0, 5)
            base_query["time_pressure"] = max(5, base_query["time_pressure"])

            query = NavigationQuery(
                user_id=scenario_key + "_baseline",
                origin=base_query["origin"],
                destination=base_query["destination"],
                time_of_day=base_query["time_of_day"],
                day_of_week=base_query.get("day_of_week", j % 7),
                expressed_concern=base_query.get("concern", ""),
                time_pressure=base_query["time_pressure"],
            )

            try:
                env_state = edt.predict(query.time_of_day, query.day_of_week)
                routes = edt.find_routes(query.origin, query.destination, cfg.NUM_CANDIDATE_ROUTES)
                if not routes:
                    routes = [[query.origin, query.destination]]
                scored = scorer.score_all_routes(routes, env_state, static_weights)

                if scored:
                    best = scored[0]

                    # Create a minimal recommendation-like object
                    class MinRec:
                        pass
                    rec = MinRec()
                    rec.score_breakdown = {
                        "PPS": best["pps"], "ECS": best["ecs"],
                        "SPS": best["sps"], "ES": best["es"]
                    }
                    rec.composite_score = best["composite"]
                    rec.selected_route = best["route"]
                    rec.route_locations = best["route_locations"]

                    outcome = outcome_model.simulate_outcome(scenario_key, scenario, rec, query)

                    scores.append(best["composite"])
                    stresses.append(outcome.get("stress_level", 0.5))
                    accepted.append(1 if outcome["accepted"] else 0)
                    completed.append(1 if outcome.get("completed", False) else 0)
            except:
                continue

        return {
            "method": "static_weights",
            "weights": static_weights.tolist(),
            "n_journeys": len(scores),
            "avg_score": np.mean(scores) if scores else 0,
            "avg_stress": np.mean(stresses) if stresses else 0,
            "acceptance_rate": np.mean(accepted) if accepted else 0,
            "completion_rate": np.mean(completed) if completed else 0,
            "scores": scores,
            "stresses": stresses,
        }

    def _compute_metrics(self, log, rewards, stresses, accepted, completed) -> Dict:
        if not log:
            return {}

        n = len(log)
        first_half = n // 2
        second_half = n - first_half
        # "Converged" period: last third of journeys
        converged_start = n * 2 // 3

        return {
            "overall": {
                "acceptance_rate": np.mean(accepted),
                "completion_rate": np.mean(completed),
                "avg_stress": np.mean(stresses),
                "avg_reward": np.mean(rewards),
                "avg_score": np.mean([j["composite_score"] for j in log]),
            },
            "first_half": {
                "acceptance_rate": np.mean(accepted[:first_half]),
                "completion_rate": np.mean(completed[:first_half]),
                "avg_stress": np.mean(stresses[:first_half]),
                "avg_reward": np.mean(rewards[:first_half]),
            },
            "second_half": {
                "acceptance_rate": np.mean(accepted[first_half:]),
                "completion_rate": np.mean(completed[first_half:]),
                "avg_stress": np.mean(stresses[first_half:]),
                "avg_reward": np.mean(rewards[first_half:]),
            },
            "converged": {
                "acceptance_rate": np.mean(accepted[converged_start:]),
                "completion_rate": np.mean(completed[converged_start:]),
                "avg_stress": np.mean(stresses[converged_start:]),
                "avg_reward": np.mean(rewards[converged_start:]),
            },
            "improvement": {
                "acceptance_rate": np.mean(accepted[first_half:]) - np.mean(accepted[:first_half]),
                "completion_rate": np.mean(completed[first_half:]) - np.mean(completed[:first_half]),
                "stress_reduction": np.mean(stresses[:first_half]) - np.mean(stresses[first_half:]),
                "reward_increase": np.mean(rewards[first_half:]) - np.mean(rewards[:first_half]),
            }
        }
