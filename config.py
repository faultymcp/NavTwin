"""
NavTwin Configuration — All hyperparameters from the paper.
"""
import numpy as np

# ── Personal Digital Twin ──────────────────────────────────────────
PDT_STATIC_DIM = 32          # p_s ∈ R^32
PDT_DYNAMIC_DIM = 64         # p_d ∈ R^64
JOURNEY_HISTORY_WINDOW = 100 # sliding window of recent journeys
GRU_HIDDEN_DIM = 64
ATTENTION_HEADS = 4
LEARNING_RATE_PDT = 1e-3

# ── Environmental Digital Twin ─────────────────────────────────────
TGNN_PREDICTION_INTERVALS = [15, 30, 45, 60]  # minutes ahead
CROWD_SCALE = 10             # crowd density 0-10
NOISE_SCALE = 10             # noise level 0-10
NUM_GRAPH_NODES = 50         # synthetic city graph size
EDGE_DENSITY = 0.15          # connectivity

# ── Route Scoring ──────────────────────────────────────────────────
ROUTE_FEATURE_DIM = 47       # d = 47 per route
ENV_FEATURES = 12            # predicted crowd density at 15-min intervals
PREF_FEATURES = 8            # personal preference features
ROUTE_CHAR_FEATURES = 15     # segment-level characteristics
SUCCESS_FEATURES = 8         # completion probability features
EFFICIENCY_FEATURES = 4      # travel time, distance, reliability

# ── RL Agent (Contextual Bandit) ───────────────────────────────────
K_ACTIONS = 10               # discrete weight configurations
RL_STATE_DIM = 12            # state space dimension
RL_HIDDEN_DIM = 64
RL_LEARNING_RATE = 3e-3
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.05
RL_EPSILON_DECAY = 0.98
RL_GAMMA = 0.99

# The K=10 weight configurations [w_PPS, w_ECS, w_SPS, w_ES]
# Ranging from efficiency-prioritising to comfort-prioritising
WEIGHT_CONFIGS = np.array([
    [0.10, 0.20, 0.20, 0.50],  # 0: Strong efficiency
    [0.15, 0.25, 0.20, 0.40],  # 1: Moderate efficiency
    [0.20, 0.30, 0.20, 0.30],  # 2: Slight efficiency bias
    [0.25, 0.30, 0.20, 0.25],  # 3: Balanced-efficiency
    [0.25, 0.25, 0.25, 0.25],  # 4: Fully balanced
    [0.30, 0.30, 0.25, 0.15],  # 5: Balanced-comfort
    [0.30, 0.35, 0.20, 0.15],  # 6: Slight comfort bias
    [0.35, 0.40, 0.15, 0.10],  # 7: Moderate comfort
    [0.35, 0.50, 0.10, 0.05],  # 8: Strong comfort
    [0.40, 0.45, 0.10, 0.05],  # 9: Maximum comfort
])

# ── Reward Signal ──────────────────────────────────────────────────
REWARD_ACCEPT = 1.0
REWARD_REJECT = -0.5
REWARD_COMPLETE = 2.0
REWARD_ABANDON = -2.0
REWARD_LOW_STRESS = 0.5
REWARD_HIGH_STRESS = -1.0

# ── Gamification ───────────────────────────────────────────────────
MILESTONE_THRESHOLDS = [5, 10, 25, 50, 100]  # journey milestones
BADGE_TYPES = ["explorer", "comfort_seeker", "time_master", "adaptable", "streak"]

# ── Simulation ─────────────────────────────────────────────────────
NUM_SIMULATION_JOURNEYS = 200
NUM_CANDIDATE_ROUTES = 5     # routes per query
RANDOM_SEED = 42
