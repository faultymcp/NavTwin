"""
RL Agent — Contextual Multi-Armed Bandit for Dynamic Weight Adaptation.

Implements a contextual bandit with neural network function approximation
to learn when to prioritise different scoring dimensions.

State space S includes:
- User's current stress level
- Time constraints (minutes until required arrival)
- Weather conditions
- Recent journey history (success/failure over past 7 days)
- Time of day

Action space A = K=10 discrete weight configurations ranging from
efficiency-prioritising (w_ES = 0.5) to comfort-prioritising
(w_ECS = 0.5, w_PPS = 0.3).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import config as cfg


class ContextualBanditNetwork:
    """
    Neural network for Q-value estimation in the contextual bandit.

    Architecture: State(12) → Dense(64) → ReLU → Dense(32) → ReLU → Q-values(10)
    """

    def __init__(self, state_dim: int = cfg.RL_STATE_DIM,
                 n_actions: int = cfg.K_ACTIONS,
                 hidden_dim: int = cfg.RL_HIDDEN_DIM,
                 lr: float = cfg.RL_LEARNING_RATE,
                 seed: int = cfg.RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.lr = lr
        self.n_actions = n_actions

        # Xavier initialization
        self.W1 = self.rng.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.randn(hidden_dim, 32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(32)
        self.W3 = self.rng.randn(32, n_actions) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: returns Q-values for all actions."""
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)     # ReLU
        q_values = h2 @ self.W3 + self.b3
        return q_values

    def update(self, state: np.ndarray, action: int,
               target: float) -> float:
        """
        Single-step gradient update for the chosen action.
        Returns the loss.
        """
        # Forward pass with intermediate activations
        z1 = state @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        q_values = h2 @ self.W3 + self.b3

        # Loss for the selected action
        predicted = q_values[action]
        loss = (predicted - target) ** 2

        # Backpropagation
        dq = np.zeros(self.n_actions)
        dq[action] = 2 * (predicted - target)

        # Layer 3
        dW3 = np.outer(h2, dq)
        db3 = dq
        dh2 = dq @ self.W3.T

        # Layer 2
        dz2 = dh2 * (z2 > 0)
        dW2 = np.outer(h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T

        # Layer 1
        dz1 = dh1 * (z1 > 0)
        dW1 = np.outer(state, dz1)
        db1 = dz1

        # Gradient descent with clipping
        clip = 1.0
        for grad in [dW1, dW2, dW3, db1, db2, db3]:
            np.clip(grad, -clip, clip, out=grad)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        return loss


class RLWeightAgent:
    """
    RL Agent that selects weight configurations for route scoring.

    Uses ε-greedy exploration with decaying ε and experience replay.
    """

    def __init__(self, seed: int = cfg.RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.network = ContextualBanditNetwork(seed=seed)
        self.epsilon = cfg.RL_EPSILON_START
        self.weight_configs = cfg.WEIGHT_CONFIGS.copy()

        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=500)

        # Tracking
        self.action_counts = np.zeros(cfg.K_ACTIONS)
        self.action_rewards = np.zeros(cfg.K_ACTIONS)
        self.total_steps = 0
        self.training_losses: List[float] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        self.weight_history: List[np.ndarray] = []

    def encode_state(self, context: Dict) -> np.ndarray:
        """
        Encode the current context into the state vector s ∈ R^12.

        Context includes:
        - stress_level: 0-1
        - time_pressure: minutes until needed arrival
        - weather: temperature, precipitation, wind
        - recent_success_rate: success over past 7 days
        - time_of_day: hour
        - day_of_week: 0-6
        - user_sensitivity_mean: average sensory sensitivity
        - journey_count: total journeys taken
        """
        state = np.zeros(cfg.RL_STATE_DIM)
        state[0] = context.get("stress_level", 0.3)
        state[1] = min(context.get("time_pressure", 30), 60) / 60.0
        state[2] = context.get("temperature", 15) / 30.0
        state[3] = min(context.get("precipitation", 0), 1.0)
        state[4] = context.get("wind", 5) / 20.0
        state[5] = context.get("recent_success_rate", 0.8)
        state[6] = context.get("time_of_day", 12) / 24.0
        state[7] = context.get("day_of_week", 2) / 6.0
        state[8] = context.get("user_sensitivity_mean", 0.5)
        state[9] = min(context.get("journey_count", 0), 100) / 100.0
        state[10] = context.get("crowd_level", 0.5)
        state[11] = context.get("noise_level", 0.5)
        return state

    def select_action(self, context: Dict) -> Tuple[int, np.ndarray]:
        """
        Select a weight configuration using ε-greedy policy.

        Returns: (action_index, weight_vector)
        """
        state = self.encode_state(context)

        if self.rng.random() < self.epsilon:
            action = self.rng.randint(cfg.K_ACTIONS)
        else:
            q_values = self.network.forward(state)
            action = int(np.argmax(q_values))

        weights = self.weight_configs[action]
        self.action_history.append(action)
        self.weight_history.append(weights.copy())
        self.action_counts[action] += 1
        self.total_steps += 1

        return action, weights

    def compute_reward(self, journey_outcome: Dict) -> float:
        """
        Compute multi-component reward signal R.

        Components:
        - Route acceptance/rejection: +1.0 / -0.5
        - Journey completion/abandonment: +2.0 / -2.0
        - Stress level: +0.5 (low) / -1.0 (high)
        """
        reward = 0.0

        # Acceptance signal
        if journey_outcome.get("accepted", True):
            reward += cfg.REWARD_ACCEPT
        else:
            reward += cfg.REWARD_REJECT

        # Completion signal
        if journey_outcome.get("completed", False):
            reward += cfg.REWARD_COMPLETE
        elif journey_outcome.get("abandoned", False):
            reward += cfg.REWARD_ABANDON

        # Stress signal
        stress = journey_outcome.get("stress_level", 0.5)
        if stress < 0.3:
            reward += cfg.REWARD_LOW_STRESS
        elif stress > 0.7:
            reward += cfg.REWARD_HIGH_STRESS

        return reward

    def update(self, context: Dict, action: int, reward: float):
        """Update the network with the observed reward."""
        state = self.encode_state(context)

        # Store in replay buffer
        self.replay_buffer.append((state, action, reward))
        self.action_rewards[action] += reward
        self.reward_history.append(reward)

        # Update network
        loss = self.network.update(state, action, reward)
        self.training_losses.append(loss)

        # Mini-batch replay (sample 8 from buffer)
        if len(self.replay_buffer) > 16:
            indices = self.rng.choice(len(self.replay_buffer), min(8, len(self.replay_buffer)),
                                       replace=False)
            for idx in indices:
                s, a, r = self.replay_buffer[idx]
                self.network.update(s, a, r)

        # Decay epsilon
        self.epsilon = max(cfg.RL_EPSILON_END,
                          self.epsilon * cfg.RL_EPSILON_DECAY)

    def get_stats(self) -> Dict:
        """Return agent statistics."""
        return {
            "total_steps": self.total_steps,
            "epsilon": round(self.epsilon, 4),
            "action_distribution": {
                f"config_{i}": int(self.action_counts[i])
                for i in range(cfg.K_ACTIONS)
            },
            "avg_reward_per_action": {
                f"config_{i}": round(self.action_rewards[i] / max(self.action_counts[i], 1), 3)
                for i in range(cfg.K_ACTIONS)
            },
            "recent_avg_reward": round(np.mean(self.reward_history[-20:]), 3) if self.reward_history else 0,
            "avg_training_loss": round(np.mean(self.training_losses[-20:]), 4) if self.training_losses else 0,
        }
