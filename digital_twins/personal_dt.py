"""
Personal Digital Twin (PDT) — Maintains a comprehensive model of the user's
sensory profile, preferences, and behavioral patterns.

Architecture:
  - Static profile vector (p_s ∈ R^32): baseline sensory sensitivities
  - Dynamic preference embedding (p_d ∈ R^64): evolves via online gradient-based learning
  - Journey history memory: sliding window of 100 most recent journeys
  - Neural preference scoring: GRU + cross-attention + MLP
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import config as cfg


@dataclass
class SensoryProfile:
    """User's sensory sensitivity profile."""
    crowd_sensitivity: float      # 0-10 (10 = extremely sensitive)
    noise_sensitivity: float
    visual_sensitivity: float
    social_interaction_avoidance: float
    transition_difficulty: float  # difficulty with route changes
    time_pressure_tolerance: float  # ability to handle time pressure
    preferred_route_type: str     # 'quiet', 'direct', 'scenic', 'familiar'
    communication_style: str      # 'minimal', 'detailed', 'visual'


@dataclass
class JourneyRecord:
    """Record of a completed journey."""
    route_features: np.ndarray    # feature vector of the chosen route
    outcome: int                  # +1 accepted, 0 neutral, -1 rejected
    completed: bool
    stress_level: float           # 0-1 post-journey stress
    time_of_day: float
    weather_conditions: Dict
    weight_config_used: int       # which K config was active
    route_score: float            # final composite score
    timestamp: int


class NeuralPreferenceNetwork:
    """
    Simplified neural network for Personal Preference Score (PPS).

    Combines:
    - GRU for temporal patterns from journey history
    - Cross-attention between route features and user representation
    - MLP prediction head for preference score

    Implementation uses numpy for portability (no PyTorch dependency needed
    for the prototype evaluation).
    """

    def __init__(self, route_dim: int = cfg.ROUTE_FEATURE_DIM,
                 static_dim: int = cfg.PDT_STATIC_DIM,
                 dynamic_dim: int = cfg.PDT_DYNAMIC_DIM,
                 hidden_dim: int = cfg.GRU_HIDDEN_DIM,
                 seed: int = cfg.RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.route_dim = route_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim

        # GRU parameters (simplified)
        self._init_gru()
        # Attention parameters
        self._init_attention()
        # MLP prediction head
        self._init_mlp()

        self.gru_hidden = np.zeros(hidden_dim)

    def _init_gru(self):
        """Initialize GRU gate parameters."""
        scale = 0.1
        d = self.route_dim + 1  # route features + outcome
        h = self.hidden_dim
        # Update gate
        self.W_z = self.rng.randn(d, h) * scale
        self.U_z = self.rng.randn(h, h) * scale
        self.b_z = np.zeros(h)
        # Reset gate
        self.W_r = self.rng.randn(d, h) * scale
        self.U_r = self.rng.randn(h, h) * scale
        self.b_r = np.zeros(h)
        # Candidate
        self.W_h = self.rng.randn(d, h) * scale
        self.U_h = self.rng.randn(h, h) * scale
        self.b_h = np.zeros(h)

    def _init_attention(self):
        """Initialize cross-attention parameters."""
        scale = 0.1
        self.W_q = self.rng.randn(self.route_dim, self.hidden_dim) * scale
        self.W_k = self.rng.randn(self.hidden_dim + self.static_dim, self.hidden_dim) * scale
        self.W_v = self.rng.randn(self.hidden_dim + self.static_dim, self.hidden_dim) * scale

    def _init_mlp(self):
        """Initialize MLP prediction head."""
        scale = 0.1
        input_dim = self.hidden_dim + self.dynamic_dim
        self.W1 = self.rng.randn(input_dim, 32) * scale
        self.b1 = np.zeros(32)
        self.W2 = self.rng.randn(32, 16) * scale
        self.b2 = np.zeros(16)
        self.W3 = self.rng.randn(16, 1) * scale
        self.b3 = np.zeros(1)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-8)

    def update_history(self, journey: JourneyRecord):
        """Process a journey through the GRU to update temporal state."""
        # Prepare input: route features + outcome
        feat_dim = min(len(journey.route_features), self.route_dim)
        x = np.zeros(self.route_dim + 1)
        x[:feat_dim] = journey.route_features[:feat_dim]
        x[-1] = journey.outcome

        h_prev = self.gru_hidden

        # GRU forward pass
        z = self._sigmoid(x @ self.W_z + h_prev @ self.U_z + self.b_z)
        r = self._sigmoid(x @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_candidate = np.tanh(x @ self.W_h + (r * h_prev) @ self.U_h + self.b_h)
        self.gru_hidden = (1 - z) * h_prev + z * h_candidate

    def compute_preference_score(self, route_features: np.ndarray,
                                  static_profile: np.ndarray,
                                  dynamic_embedding: np.ndarray) -> float:
        """
        Compute Personal Preference Score for a candidate route.

        f_PPS: (r, p_s, p_d, H) → [0, 1]
        """
        # Cross-attention: query from route, key/value from user state
        feat_dim = min(len(route_features), self.route_dim)
        r = np.zeros(self.route_dim)
        r[:feat_dim] = route_features[:feat_dim]

        query = r @ self.W_q  # route queries user

        # Combine GRU hidden state with static profile for keys/values
        s_dim = min(len(static_profile), self.static_dim)
        sp = np.zeros(self.static_dim)
        sp[:s_dim] = static_profile[:s_dim]

        user_state = np.concatenate([self.gru_hidden, sp])
        key = user_state @ self.W_k
        value = user_state @ self.W_v

        # Scaled dot-product attention
        attn_score = np.dot(query, key) / np.sqrt(self.hidden_dim)
        attn_weight = self._sigmoid(attn_score)
        attended = attn_weight * value

        # MLP prediction head
        d_dim = min(len(dynamic_embedding), self.dynamic_dim)
        de = np.zeros(self.dynamic_dim)
        de[:d_dim] = dynamic_embedding[:d_dim]

        mlp_input = np.concatenate([attended, de])
        h1 = self._relu(mlp_input @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        output = self._sigmoid(h2 @ self.W3 + self.b3)

        return float(output[0])


class PersonalDigitalTwin:
    """
    Complete PDT implementation.

    Maintains user profile, preference network, journey history,
    and online learning of dynamic embeddings.
    """

    def __init__(self, user_id: str, sensory_profile: SensoryProfile,
                 seed: int = cfg.RANDOM_SEED):
        self.user_id = user_id
        self.sensory_profile = sensory_profile
        self.rng = np.random.RandomState(seed)

        # Static profile vector (encodes sensory sensitivities)
        self.static_vector = self._encode_static_profile()

        # Dynamic preference embedding (learned online)
        self.dynamic_embedding = self.rng.randn(cfg.PDT_DYNAMIC_DIM) * 0.01

        # Journey history
        self.journey_history: List[JourneyRecord] = []

        # Neural preference network
        self.preference_net = NeuralPreferenceNetwork(seed=seed)

        # Learning rate for dynamic embedding updates
        self.lr = cfg.LEARNING_RATE_PDT

    def _encode_static_profile(self) -> np.ndarray:
        """Encode sensory profile into static vector p_s ∈ R^32."""
        sp = self.sensory_profile
        raw = np.array([
            sp.crowd_sensitivity / 10.0,
            sp.noise_sensitivity / 10.0,
            sp.visual_sensitivity / 10.0,
            sp.social_interaction_avoidance / 10.0,
            sp.transition_difficulty / 10.0,
            sp.time_pressure_tolerance / 10.0,
            1.0 if sp.preferred_route_type == 'quiet' else 0.0,
            1.0 if sp.preferred_route_type == 'direct' else 0.0,
            1.0 if sp.preferred_route_type == 'scenic' else 0.0,
            1.0 if sp.preferred_route_type == 'familiar' else 0.0,
        ])
        # Pad to 32 dims with interaction features
        padded = np.zeros(cfg.PDT_STATIC_DIM)
        padded[:len(raw)] = raw
        # Add cross-sensitivity interactions
        padded[10] = raw[0] * raw[1]  # crowd × noise
        padded[11] = raw[0] * raw[2]  # crowd × visual
        padded[12] = raw[1] * raw[2]  # noise × visual
        padded[13] = raw[3] * raw[4]  # social × transition
        padded[14] = np.mean(raw[:3])  # mean sensitivity
        padded[15] = np.std(raw[:3])   # sensitivity variation
        return padded

    def compute_pps(self, route_features: np.ndarray) -> float:
        """Compute Personal Preference Score for a route."""
        return self.preference_net.compute_preference_score(
            route_features, self.static_vector, self.dynamic_embedding
        )

    def record_journey(self, record: JourneyRecord):
        """Record a journey and update the PDT."""
        self.journey_history.append(record)
        if len(self.journey_history) > cfg.JOURNEY_HISTORY_WINDOW:
            self.journey_history = self.journey_history[-cfg.JOURNEY_HISTORY_WINDOW:]

        # Update GRU with new journey
        self.preference_net.update_history(record)

        # Update dynamic embedding via gradient signal
        feedback = record.outcome  # {-1, 0, +1}
        if record.completed:
            feedback += 1.0 - record.stress_level
        else:
            feedback -= 0.5

        # Gradient-like update to dynamic embedding
        gradient = feedback * self.rng.randn(cfg.PDT_DYNAMIC_DIM) * 0.01
        self.dynamic_embedding += self.lr * gradient
        # Normalize to prevent drift
        norm = np.linalg.norm(self.dynamic_embedding)
        if norm > 1.0:
            self.dynamic_embedding /= norm

    def get_sensitivity_weights(self) -> Dict[str, float]:
        """Get normalised sensitivity weights for route scoring."""
        sp = self.sensory_profile
        total = sp.crowd_sensitivity + sp.noise_sensitivity + sp.visual_sensitivity + 1e-8
        return {
            "crowd": sp.crowd_sensitivity / total,
            "noise": sp.noise_sensitivity / total,
            "visual": sp.visual_sensitivity / total,
        }

    def get_profile_summary(self) -> Dict:
        """Return a human-readable profile summary."""
        sp = self.sensory_profile
        return {
            "user_id": self.user_id,
            "crowd_sensitivity": f"{sp.crowd_sensitivity}/10",
            "noise_sensitivity": f"{sp.noise_sensitivity}/10",
            "visual_sensitivity": f"{sp.visual_sensitivity}/10",
            "preferred_route_type": sp.preferred_route_type,
            "journeys_completed": sum(1 for j in self.journey_history if j.completed),
            "total_journeys": len(self.journey_history),
            "avg_stress": f"{np.mean([j.stress_level for j in self.journey_history]) if self.journey_history else 0:.2f}",
        }
