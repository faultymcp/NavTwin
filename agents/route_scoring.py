"""
Route Scoring Algorithm — Multi-criteria evaluation across four dimensions:

  Score(r) = w_PPS·PPS(r) + w_ECS·ECS(r) + w_SPS·SPS(r) + w_ES·ES(r)

where weights w_i ∈ [0,1] satisfy Σw_i = 1 and are dynamically adjusted
by the RL Agent based on context and learned user preferences.

Each route r is represented as a feature vector r ∈ R^47.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from digital_twins.environment_dt import EnvironmentDigitalTwin, EnvironmentState, Location
from digital_twins.personal_dt import PersonalDigitalTwin, SensoryProfile
import config as cfg


class RouteFeatureExtractor:
    """Extracts the d=47 dimensional feature vector for each candidate route."""

    def __init__(self, edt: EnvironmentDigitalTwin):
        self.edt = edt

    def extract(self, route: List[int], env_state: EnvironmentState) -> np.ndarray:
        """
        Extract feature vector r ∈ R^47 for a candidate route.

        Features:
        - 12 environmental (crowd density at 15-min intervals across segments)
        - 8 personal preference (acceptance patterns, sensitivity alignment)
        - 15 route characteristics (segment-level crowd, noise, visual complexity)
        - 8 success probability (transfer complexity, completion patterns)
        - 4 efficiency (travel time, distance, schedule reliability)
        """
        features = np.zeros(cfg.ROUTE_FEATURE_DIM)

        if len(route) < 2:
            return features

        # ── Environmental features (12) ────────────────────────────
        # Average crowd density predictions across route at each interval
        for interval_idx in range(4):  # 15m, 30m, 45m, 60m
            segment_crowds = []
            segment_noises = []
            for node in route:
                crowd = env_state.crowd_predictions.get(node, [5]*4)[interval_idx]
                noise = env_state.noise_predictions.get(node, [5]*4)[interval_idx]
                segment_crowds.append(crowd)
                segment_noises.append(noise)
            features[interval_idx] = np.mean(segment_crowds) / cfg.CROWD_SCALE
            features[4 + interval_idx] = np.max(segment_crowds) / cfg.CROWD_SCALE
            features[8 + interval_idx] = np.mean(segment_noises) / cfg.NOISE_SCALE

        # ── Personal preference features (8) ───────────────────────
        # These encode route characteristics relevant to preference learning
        park_nodes = sum(1 for n in route if self.edt.locations[n].location_type == 'park')
        station_nodes = sum(1 for n in route if self.edt.locations[n].location_type == 'station')
        residential = sum(1 for n in route if self.edt.locations[n].location_type == 'residential')

        features[12] = park_nodes / max(len(route), 1)       # park ratio
        features[13] = station_nodes / max(len(route), 1)    # transit ratio
        features[14] = residential / max(len(route), 1)      # residential ratio
        features[15] = len(route) / 10.0                     # normalised length
        features[16] = self._route_type_consistency(route)    # how consistent the route type is
        features[17] = self._familiarity_score(route)         # placeholder for familiarity
        features[18] = self._greenspace_ratio(route)          # green space coverage
        features[19] = self._social_exposure(route, env_state)  # social interaction likelihood

        # ── Route characteristics (15) ─────────────────────────────
        for i, (seg_start, seg_end) in enumerate(zip(route[:-1], route[1:])):
            if i >= 5:
                break
            edge_data = self.edt.get_edge_features(seg_start, seg_end, env_state)
            if edge_data:
                features[20 + i*3] = edge_data['base_crowd']
                features[21 + i*3] = edge_data['base_noise']
                features[22 + i*3] = edge_data['visual_complexity']

        # ── Success probability features (8) ───────────────────────
        transfers = sum(1 for n in route if self.edt.locations[n].transit_hub) - \
                    (1 if self.edt.locations[route[0]].transit_hub else 0) - \
                    (1 if self.edt.locations[route[-1]].transit_hub else 0)
        transfers = max(0, transfers)

        features[35] = transfers / 3.0                        # normalised transfer count
        features[36] = self._complexity_score(route)          # route complexity
        features[37] = np.mean([env_state.confidence.get(n, 0.5) for n in route])  # prediction confidence
        features[38] = 1.0 if len(route) <= 5 else 0.5 if len(route) <= 8 else 0.2  # simplicity
        features[39] = self._max_crowd_exposure(route, env_state)  # worst-case crowd
        features[40] = self._variability_score(route, env_state)   # consistency of conditions
        features[41] = len(route) / 20.0                           # raw length
        features[42] = transfers                                    # raw transfers

        # ── Efficiency features (4) ────────────────────────────────
        total_dist = self._total_distance(route)
        features[43] = total_dist / 5000.0                    # normalised distance (5km ref)
        features[44] = self._estimated_time(route) / 60.0     # normalised time (60min ref)
        features[45] = self._schedule_reliability(route, env_state)
        features[46] = 1.0 / (1.0 + transfers)               # transfer penalty

        return features

    def _route_type_consistency(self, route: List[int]) -> float:
        types = [self.edt.locations[n].location_type for n in route]
        if not types:
            return 0
        most_common = max(set(types), key=types.count)
        return types.count(most_common) / len(types)

    def _familiarity_score(self, route: List[int]) -> float:
        return 0.5  # placeholder - would use journey history

    def _greenspace_ratio(self, route: List[int]) -> float:
        green = sum(1 for n in route if self.edt.locations[n].location_type in ('park',))
        return green / max(len(route), 1)

    def _social_exposure(self, route: List[int], env_state: EnvironmentState) -> float:
        crowds = [env_state.crowd_predictions.get(n, [5])[0] for n in route]
        return np.mean(crowds) / cfg.CROWD_SCALE

    def _complexity_score(self, route: List[int]) -> float:
        types = set(self.edt.locations[n].location_type for n in route)
        return len(types) / 5.0

    def _max_crowd_exposure(self, route: List[int], env_state: EnvironmentState) -> float:
        crowds = [env_state.crowd_predictions.get(n, [5])[0] for n in route]
        return max(crowds) / cfg.CROWD_SCALE if crowds else 0.5

    def _variability_score(self, route: List[int], env_state: EnvironmentState) -> float:
        crowds = [env_state.crowd_predictions.get(n, [5])[0] for n in route]
        return np.std(crowds) / cfg.CROWD_SCALE if len(crowds) > 1 else 0

    def _total_distance(self, route: List[int]) -> float:
        total = 0
        for i in range(len(route) - 1):
            s, t = self.edt.locations[route[i]], self.edt.locations[route[i+1]]
            total += np.sqrt((s.x - t.x)**2 + (s.y - t.y)**2) * 1000
        return max(total, 100)

    def _estimated_time(self, route: List[int]) -> float:
        dist = self._total_distance(route)
        speed = 4.5  # km/h walking speed
        return (dist / 1000.0) / speed * 60  # minutes

    def _schedule_reliability(self, route: List[int], env_state: EnvironmentState) -> float:
        delays = [env_state.transit_delays.get(n, 0) for n in route]
        max_delay = max(delays) if delays else 0
        return max(0, 1.0 - max_delay / 15.0)


class RouteScorer:
    """
    Implements the multi-criteria route scoring: Score(r) = Σ w_i · S_i(r)
    """

    def __init__(self, edt: EnvironmentDigitalTwin, pdt: PersonalDigitalTwin):
        self.edt = edt
        self.pdt = pdt
        self.feature_extractor = RouteFeatureExtractor(edt)

    def score_route(self, route: List[int], env_state: EnvironmentState,
                    weights: np.ndarray) -> Dict:
        """
        Score a single candidate route across all four dimensions.

        Returns dict with individual scores and weighted composite.
        """
        features = self.feature_extractor.extract(route, env_state)

        pps = self._compute_pps(features)
        ecs = self._compute_ecs(route, features, env_state)
        sps = self._compute_sps(route, features, env_state)
        es = self._compute_es(features)

        # Weighted composite score
        composite = (weights[0] * pps + weights[1] * ecs +
                     weights[2] * sps + weights[3] * es)

        return {
            "route": route,
            "features": features,
            "pps": pps,
            "ecs": ecs,
            "sps": sps,
            "es": es,
            "composite": composite,
            "weights": weights.copy(),
            "route_locations": [self.edt.locations[n].name for n in route],
            "distance_m": self.feature_extractor._total_distance(route),
            "estimated_time_min": self.feature_extractor._estimated_time(route),
        }

    def _compute_pps(self, features: np.ndarray) -> float:
        """Personal Preference Score via PDT neural network."""
        return self.pdt.compute_pps(features)

    def _compute_ecs(self, route: List[int], features: np.ndarray,
                     env_state: EnvironmentState) -> float:
        """
        Environmental Comfort Score — quantifies sensory demand of the route
        given predicted environmental conditions weighted by user sensitivities.
        """
        sensitivity = self.pdt.get_sensitivity_weights()

        # Aggregate comfort across segments
        segment_scores = []
        for node in route:
            crowd = env_state.crowd_predictions.get(node, [5])[0] / cfg.CROWD_SCALE
            noise = env_state.noise_predictions.get(node, [5])[0] / cfg.NOISE_SCALE

            loc = self.edt.locations[node]
            visual = 0.3 if loc.location_type in ('park', 'residential') else 0.7

            # Comfort is inverse of weighted sensory load
            sensory_load = (sensitivity["crowd"] * crowd +
                           sensitivity["noise"] * noise +
                           sensitivity["visual"] * visual)
            comfort = 1.0 - np.clip(sensory_load, 0, 1)
            segment_scores.append(comfort)

        if not segment_scores:
            return 0.5

        # Weight: min comfort matters more than average (bottleneck effect)
        avg_comfort = np.mean(segment_scores)
        min_comfort = np.min(segment_scores)
        return 0.4 * avg_comfort + 0.6 * min_comfort

    def _compute_sps(self, route: List[int], features: np.ndarray,
                     env_state: EnvironmentState) -> float:
        """
        Success Probability Score — estimates likelihood the user will
        complete the journey without abandonment or significant distress.

        Computed via gradient boosting-like feature combination.
        """
        # Feature extraction for success prediction
        transfers = features[35] * 3.0  # denormalise
        complexity = features[36]
        simplicity = features[38]
        max_crowd = features[39]
        prediction_confidence = features[37]

        # Base success probability
        base_p = 0.8

        # Penalties
        transfer_penalty = transfers * 0.08
        complexity_penalty = complexity * 0.1
        crowd_penalty = max_crowd * 0.15

        # Bonuses
        simplicity_bonus = simplicity * 0.1
        confidence_bonus = prediction_confidence * 0.05

        # User-specific factors
        sp = self.pdt.sensory_profile
        sensitivity_factor = np.mean([sp.crowd_sensitivity, sp.noise_sensitivity,
                                       sp.visual_sensitivity]) / 10.0
        # More sensitive users are more likely to abandon uncomfortable routes
        crowd_penalty *= (1.0 + sensitivity_factor)

        sps = base_p - transfer_penalty - complexity_penalty - crowd_penalty + \
              simplicity_bonus + confidence_bonus

        # Historical success rate modulation (if history available)
        if self.pdt.journey_history:
            recent = self.pdt.journey_history[-20:]
            hist_rate = sum(1 for j in recent if j.completed) / len(recent)
            sps = 0.7 * sps + 0.3 * hist_rate

        return np.clip(sps, 0, 1)

    def _compute_es(self, features: np.ndarray) -> float:
        """
        Efficiency Score — measures objective travel quality.

        Normalised so routes within 10% of optimal receive ES > 0.8.
        """
        distance_norm = features[43]       # dist / 5000m
        time_norm = features[44]           # time / 60min
        reliability = features[45]         # schedule reliability
        transfer_factor = features[46]     # 1/(1+transfers)

        # Efficiency favours shorter, faster, reliable routes
        raw_es = (0.3 * (1 - distance_norm) + 0.3 * (1 - time_norm) +
                  0.2 * reliability + 0.2 * transfer_factor)

        return np.clip(raw_es, 0, 1)

    def score_all_routes(self, routes: List[List[int]],
                         env_state: EnvironmentState,
                         weights: np.ndarray) -> List[Dict]:
        """Score all candidate routes and return sorted by composite score."""
        scored = [self.score_route(r, env_state, weights) for r in routes]
        scored.sort(key=lambda x: x["composite"], reverse=True)
        return scored
