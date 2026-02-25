"""
Environmental Digital Twin (EDT) — Models external conditions and predicts
future states using a Temporal Graph Neural Network (TGNN) approach.

The EDT operates on a spatial graph where nodes represent locations and
edges capture spatial relationships. It integrates real-time data with
historical patterns to generate probabilistic predictions of crowd density,
noise levels, and transit delays.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import config as cfg


@dataclass
class Location:
    """A node in the city graph."""
    node_id: int
    name: str
    x: float
    y: float
    base_crowd: float      # baseline crowd density [0, 10]
    base_noise: float      # baseline noise level [0, 10]
    location_type: str     # 'station', 'park', 'street', 'mall', 'residential'
    transit_hub: bool = False


@dataclass
class Edge:
    """A connection between two locations."""
    source: int
    target: int
    distance: float        # metres
    road_type: str         # 'main_road', 'side_street', 'footpath', 'park_path'
    base_noise: float
    base_crowd: float
    visual_complexity: float  # 0-1, how visually busy the segment is


@dataclass
class EnvironmentState:
    """Current environmental predictions from the EDT."""
    timestamp: float       # simulation time
    crowd_predictions: Dict[int, List[float]]   # node_id -> [15m, 30m, 45m, 60m]
    noise_predictions: Dict[int, List[float]]    # node_id -> [15m, 30m, 45m, 60m]
    confidence: Dict[int, float]                 # node_id -> prediction confidence
    weather: Dict[str, float]                    # temperature, precipitation, wind
    transit_delays: Dict[int, float]             # node_id -> delay minutes


class EnvironmentDigitalTwin:
    """
    EDT with TGNN-inspired predictions.

    For the prototype, we simulate the TGNN using historical pattern models
    with temporal variation and stochastic perturbation. In production, this
    would be a trained Temporal Graph Neural Network processing real municipal
    API data.
    """

    def __init__(self, seed: int = cfg.RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.locations: Dict[int, Location] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[int, List[int]] = {}
        self.time_step = 0
        self._build_city_graph()

    def _build_city_graph(self):
        """Build a synthetic but realistic city graph."""
        # Define location templates for a realistic urban environment
        location_templates = [
            ("Central Station",    "station",     0.5, 0.5, 8.0, 7.5, True),
            ("Park Street",        "street",      0.3, 0.6, 4.0, 5.0, False),
            ("Riverside Park",     "park",        0.1, 0.4, 1.5, 1.0, False),
            ("Shopping Mall",      "mall",        0.6, 0.7, 7.0, 6.5, False),
            ("Main Street",        "street",      0.5, 0.3, 6.0, 7.0, False),
            ("University Campus",  "residential", 0.7, 0.5, 3.5, 3.0, False),
            ("Bus Terminal",       "station",     0.4, 0.2, 6.5, 7.0, True),
            ("Library Quarter",    "residential", 0.3, 0.3, 2.0, 1.5, False),
            ("Market Square",      "street",      0.6, 0.4, 7.5, 6.0, False),
            ("Quiet Gardens",      "park",        0.2, 0.7, 1.0, 0.5, False),
            ("Tech Park",          "residential", 0.8, 0.6, 3.0, 2.5, False),
            ("Cafe District",      "street",      0.4, 0.5, 5.0, 4.5, False),
            ("Residential North",  "residential", 0.3, 0.8, 2.0, 1.5, False),
            ("Bridge Crossing",    "street",      0.2, 0.5, 4.5, 5.5, False),
            ("South Station",      "station",     0.6, 0.1, 7.0, 6.5, True),
            ("Waterfront Walk",    "park",        0.1, 0.2, 2.0, 2.5, False),
            ("Cinema Complex",     "mall",        0.7, 0.3, 6.0, 5.5, False),
            ("Hospital Road",      "street",      0.8, 0.4, 4.0, 5.0, False),
            ("Community Centre",   "residential", 0.5, 0.6, 3.5, 2.5, False),
            ("Industrial Area",    "street",      0.9, 0.2, 3.0, 6.0, False),
        ]

        for i, (name, loc_type, x, y, crowd, noise, transit) in enumerate(location_templates):
            self.locations[i] = Location(
                node_id=i, name=name, x=x, y=y,
                base_crowd=crowd, base_noise=noise,
                location_type=loc_type, transit_hub=transit
            )
            self.adjacency[i] = []

        # Define realistic connections
        edge_definitions = [
            (0, 1, "main_road"),   (0, 4, "main_road"),   (0, 6, "main_road"),
            (1, 2, "side_street"), (1, 3, "main_road"),   (1, 7, "side_street"),
            (2, 9, "park_path"),   (2, 13, "footpath"),   (3, 4, "main_road"),
            (3, 8, "side_street"), (4, 5, "main_road"),   (4, 14, "main_road"),
            (5, 10, "side_street"),(5, 11, "side_street"),(6, 7, "side_street"),
            (6, 14, "main_road"), (7, 11, "footpath"),    (8, 3, "side_street"),
            (8, 16, "main_road"), (9, 12, "park_path"),   (9, 2, "park_path"),
            (10, 17, "side_street"),(10, 5, "side_street"),(11, 18, "side_street"),
            (12, 13, "side_street"),(13, 2, "footpath"),  (14, 15, "side_street"),
            (14, 19, "main_road"),(15, 16, "footpath"),   (16, 17, "side_street"),
            (17, 18, "side_street"),(18, 19, "side_street"),(0, 8, "main_road"),
            (1, 11, "side_street"),(3, 16, "main_road"),  (5, 18, "side_street"),
            (7, 12, "footpath"),  (9, 13, "park_path"),
        ]

        road_noise = {"main_road": 0.8, "side_street": 0.5, "footpath": 0.2, "park_path": 0.1}
        road_crowd = {"main_road": 0.7, "side_street": 0.4, "footpath": 0.2, "park_path": 0.15}
        road_visual = {"main_road": 0.8, "side_street": 0.5, "footpath": 0.3, "park_path": 0.2}

        for src, tgt, road_type in edge_definitions:
            loc_s, loc_t = self.locations[src], self.locations[tgt]
            dist = np.sqrt((loc_s.x - loc_t.x)**2 + (loc_s.y - loc_t.y)**2) * 1000
            dist = max(dist, 100)  # minimum 100m

            edge = Edge(
                source=src, target=tgt, distance=dist,
                road_type=road_type,
                base_noise=road_noise[road_type],
                base_crowd=road_crowd[road_type],
                visual_complexity=road_visual[road_type]
            )
            self.edges.append(edge)
            self.adjacency[src].append(tgt)
            self.adjacency[tgt].append(src)

    def predict(self, time_of_day: float, day_of_week: int,
                weather: Optional[Dict] = None) -> EnvironmentState:
        """
        Generate environmental predictions for all locations.

        Simulates TGNN output: crowd density and noise predictions at
        15-minute intervals up to 60 minutes ahead, with confidence scores.

        Args:
            time_of_day: Hour (0-24)
            day_of_week: 0=Monday ... 6=Sunday
            weather: Optional weather conditions
        """
        self.time_step += 1

        if weather is None:
            weather = {
                "temperature": 15.0 + self.rng.normal(0, 3),
                "precipitation": max(0, self.rng.normal(0.1, 0.3)),
                "wind": max(0, self.rng.normal(5, 3))
            }

        crowd_preds = {}
        noise_preds = {}
        confidence = {}
        transit_delays = {}

        for nid, loc in self.locations.items():
            # Temporal modulation: morning rush, lunch, evening rush
            time_factor = self._temporal_pattern(time_of_day, loc.location_type)
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0

            # Weather effect: rain reduces outdoor crowds
            rain_factor = max(0.5, 1.0 - weather["precipitation"] * 0.5)
            if loc.location_type == "park":
                rain_factor *= 0.6  # parks much emptier in rain

            base_crowd = loc.base_crowd * time_factor * weekend_factor * rain_factor
            base_noise = loc.base_noise * time_factor * weekend_factor * 0.8 + loc.base_noise * 0.2

            # Generate predictions at each interval with increasing uncertainty
            c_preds, n_preds = [], []
            for i, interval in enumerate(cfg.TGNN_PREDICTION_INTERVALS):
                uncertainty = 0.05 * (i + 1)  # increasing uncertainty
                c_pred = np.clip(base_crowd + self.rng.normal(0, uncertainty * base_crowd), 0, 10)
                n_pred = np.clip(base_noise + self.rng.normal(0, uncertainty * base_noise), 0, 10)
                c_preds.append(round(c_pred, 1))
                n_preds.append(round(n_pred, 1))

            crowd_preds[nid] = c_preds
            noise_preds[nid] = n_preds

            # Confidence decreases with prediction horizon and location variability
            variability = (loc.base_crowd + loc.base_noise) / 20.0
            confidence[nid] = round(max(0.5, 0.95 - variability * 0.3), 2)

            # Transit delays for hub stations
            if loc.transit_hub:
                delay = max(0, self.rng.normal(2, 3) * time_factor)
                transit_delays[nid] = round(delay, 1)

        return EnvironmentState(
            timestamp=self.time_step,
            crowd_predictions=crowd_preds,
            noise_predictions=noise_preds,
            confidence=confidence,
            weather=weather,
            transit_delays=transit_delays
        )

    def _temporal_pattern(self, hour: float, loc_type: str) -> float:
        """Model time-of-day patterns for different location types."""
        patterns = {
            "station": lambda h: 0.3 + 0.7 * (np.exp(-((h-8.5)**2)/2) + np.exp(-((h-17.5)**2)/2)),
            "park":    lambda h: 0.2 + 0.8 * np.exp(-((h-14)**2)/8),
            "street":  lambda h: 0.3 + 0.7 * (0.5 * np.exp(-((h-12)**2)/4) + 0.5 * np.exp(-((h-18)**2)/4)),
            "mall":    lambda h: 0.2 + 0.8 * np.exp(-((h-15)**2)/8),
            "residential": lambda h: 0.4 + 0.3 * np.exp(-((h-9)**2)/4) + 0.3 * np.exp(-((h-19)**2)/4),
        }
        return np.clip(patterns.get(loc_type, patterns["street"])(hour), 0.1, 1.0)

    def get_edge_features(self, source: int, target: int,
                          env_state: EnvironmentState) -> Dict:
        """Get environmental features for a specific edge."""
        for edge in self.edges:
            if (edge.source == source and edge.target == target) or \
               (edge.target == source and edge.source == target):
                # Blend edge baseline with endpoint predictions
                src_crowd = env_state.crowd_predictions.get(source, [5]*4)
                tgt_crowd = env_state.crowd_predictions.get(target, [5]*4)
                src_noise = env_state.noise_predictions.get(source, [5]*4)
                tgt_noise = env_state.noise_predictions.get(target, [5]*4)

                return {
                    "distance": edge.distance,
                    "road_type": edge.road_type,
                    "crowd_density": [(s+t)/2 for s, t in zip(src_crowd, tgt_crowd)],
                    "noise_level": [(s+t)/2 for s, t in zip(src_noise, tgt_noise)],
                    "visual_complexity": edge.visual_complexity,
                    "base_noise": edge.base_noise,
                    "base_crowd": edge.base_crowd,
                }
        return None

    def find_routes(self, origin: int, destination: int,
                    max_routes: int = 5) -> List[List[int]]:
        """Find diverse candidate routes using modified DFS with variation."""
        routes = []
        self._find_paths_dfs(origin, destination, [origin], set([origin]),
                             routes, max_length=10)

        # Sort by length (hops) and take diverse subset
        routes.sort(key=len)
        if len(routes) > max_routes:
            # Select diverse routes: shortest, longest, and spread in between
            indices = np.linspace(0, len(routes)-1, max_routes, dtype=int)
            routes = [routes[i] for i in indices]

        return routes[:max_routes]

    def _find_paths_dfs(self, current: int, destination: int,
                        path: List[int], visited: set,
                        routes: List, max_length: int = 10):
        """DFS path finding with max length constraint."""
        if current == destination:
            routes.append(list(path))
            return
        if len(path) >= max_length:
            return
        if len(routes) >= 20:  # cap total routes found
            return

        neighbors = self.adjacency.get(current, [])
        # Shuffle for route diversity
        self.rng.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                self._find_paths_dfs(neighbor, destination, path, visited,
                                     routes, max_length)
                path.pop()
                visited.remove(neighbor)
