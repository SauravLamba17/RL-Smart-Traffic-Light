"""
Tabular Q-Learning Reinforcement Learning Agent for Traffic Signal Control.
Implements Bellman update, epsilon-greedy action selection, and portable serialization.
"""
import os
import pickle
import random
import time
import numpy as np
from typing import Tuple, Dict, Any
from environment.intersection_env import IntersectionEnv


class QLearningAgent:
    """
    Q-Learning Agent using tabular Q-learning over discretized intersection states.
    """

    NUM_STATES = 13824  # 4 * 4 * 4 * 4 * 2 * 9 * 3
    NUM_ACTIONS = 2     # 0: Keep Phase, 1: Switch Phase

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.998,
        seed: int | None = None,
    ):
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.name = "Q-Learning RL Agent"

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.NUM_STATES, self.NUM_ACTIONS), dtype=np.float32)

        # Training statistics
        self.total_updates = 0
        self.episodes_trained = 0

    def get_state_index(self, state: Tuple[int, ...]) -> int:
        """Converts state tuple into flat index [0, 13823]."""
        return IntersectionEnv.state_to_index(state)

    def get_action(
        self,
        state: Tuple[int, ...],
        env: Any = None,
        training: bool = True
    ) -> int:
        """
        Selects action using epsilon-greedy policy.
        """
        state_idx = self.get_state_index(state)

        # Epsilon-greedy exploration
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1])

        # Greedy exploitation with tie-breaking
        q_vals = self.q_table[state_idx]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool = False
    ) -> float:
        """
        Applies standard Q-learning Bellman update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        Returns TD error.
        """
        s_idx = self.get_state_index(state)
        next_s_idx = self.get_state_index(next_state)

        current_q = self.q_table[s_idx, action]
        best_next_q = np.max(self.q_table[next_s_idx]) if not done else 0.0

        target = reward + (self.gamma * best_next_q)
        td_error = target - current_q

        # Q-table update
        self.q_table[s_idx, action] += self.learning_rate * td_error
        self.total_updates += 1

        return float(td_error)

    def decay_epsilon(self) -> None:
        """Decays exploration rate epsilon towards minimum bound."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str) -> None:
        """Serializes trained Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episodes_trained': self.episodes_trained,
            'total_updates': self.total_updates,
            'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'version': "2.0.0",
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)

    def load_model(self, filepath: str) -> bool:
        """Loads serialized Q-table from disk."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        loaded_table = payload.get('q_table', None)
        if loaded_table is None or loaded_table.shape != self.q_table.shape:
            return False

        self.q_table = loaded_table
        self.learning_rate = payload.get('learning_rate', self.learning_rate)
        self.gamma = payload.get('gamma', self.gamma)
        self.epsilon = payload.get('epsilon', self.epsilon_min)
        self.episodes_trained = payload.get('episodes_trained', 0)
        self.total_updates = payload.get('total_updates', 0)
        return True
