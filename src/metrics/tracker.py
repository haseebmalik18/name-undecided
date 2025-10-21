import numpy as np
from typing import Dict, List
import json


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_hit_rates = []
        self.episode_latencies = []
        self.episode_bandwidth = []
        self.losses = []

        self.step_rewards = []
        self.step_hit_rates = []

    def add_episode(self, total_reward: float, hit_rate: float, avg_latency: float, bandwidth: float):
        self.episode_rewards.append(total_reward)
        self.episode_hit_rates.append(hit_rate)
        self.episode_latencies.append(avg_latency)
        self.episode_bandwidth.append(bandwidth)

    def add_step(self, reward: float, hit_rate: float):
        self.step_rewards.append(reward)
        self.step_hit_rates.append(hit_rate)

    def add_loss(self, loss: float):
        if loss is not None:
            self.losses.append(loss)

    def get_stats(self, window: int = 100) -> Dict:
        recent_rewards = self.episode_rewards[-window:] if self.episode_rewards else [0]
        recent_hit_rates = self.episode_hit_rates[-window:] if self.episode_hit_rates else [0]

        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_hit_rate': np.mean(recent_hit_rates),
            'std_hit_rate': np.std(recent_hit_rates),
            'total_episodes': len(self.episode_rewards),
            'mean_loss': np.mean(self.losses[-window:]) if self.losses else 0
        }

    def save(self, filepath: str):
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_hit_rates': self.episode_hit_rates,
            'episode_latencies': self.episode_latencies,
            'episode_bandwidth': self.episode_bandwidth,
            'losses': self.losses
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.episode_rewards = data['episode_rewards']
        self.episode_hit_rates = data['episode_hit_rates']
        self.episode_latencies = data['episode_latencies']
        self.episode_bandwidth = data['episode_bandwidth']
        self.losses = data['losses']
