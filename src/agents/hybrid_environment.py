import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
sys.path.append('..')
from cache.hybrid import HybridCache


class HybridCacheEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        cache_capacity: int,
        num_items: int,
        workload_generator,
        episode_length: int = 1000,
        state_size: int = 20,
        base_policy: str = 'lru',
        rl_weight: float = 0.5,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05
    ):
        super().__init__()

        self.cache = HybridCache(
            capacity=cache_capacity,
            base_policy=base_policy,
            rl_weight=rl_weight,
            miss_penalty=10.0,
            hit_latency=1.0
        )
        self.num_items = num_items
        self.workload = workload_generator
        self.episode_length = episode_length
        self.state_size = state_size
        self.base_policy = base_policy

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.current_step = 0
        self.current_request = None

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(cache_capacity,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(state_size,),
            dtype=np.float32
        )

        self.request_history = []
        self.item_to_cache_idx = {}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.cache.reset()
        self.current_step = 0
        self.request_history = []
        self.item_to_cache_idx = {}

        self.current_request = next(iter(self.workload))

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        is_hit, latency = self.cache.access(self.current_request)

        rl_eviction_scores = self._action_to_eviction_scores(action)

        evicted = None
        if not is_hit:
            evicted = self.cache.admit(self.current_request, rl_eviction_scores)

        hit_reward = self.alpha if is_hit else -self.alpha
        latency_penalty = -self.beta * latency / self.cache.miss_penalty
        bandwidth_penalty = 0 if is_hit else -self.gamma

        reward = hit_reward + latency_penalty + bandwidth_penalty

        if evicted is not None:
            future_access_penalty = 0.0
            if evicted in self.request_history[-10:]:
                future_access_penalty = -0.2
            reward += future_access_penalty

        self.request_history.append(self.current_request)
        if len(self.request_history) > 100:
            self.request_history.pop(0)

        self.current_step += 1
        done = self.current_step >= self.episode_length
        truncated = False

        self.current_request = next(iter(self.workload))

        obs = self._get_observation()
        info = {
            'hit_rate': self.cache.get_metrics()['hit_rate'],
            'cache_size': len(self.cache),
            'evicted': evicted,
            'rl_influenced': self.cache.rl_influenced_evictions
        }

        return obs, reward, done, truncated, info

    def _action_to_eviction_scores(self, action: np.ndarray) -> Dict[Any, float]:
        scores = {}
        cache_items = list(self.cache.cache.keys())

        self.item_to_cache_idx = {}
        for idx, item in enumerate(cache_items):
            self.item_to_cache_idx[item] = idx

        for item in cache_items:
            idx = self.item_to_cache_idx.get(item, 0)
            if idx < len(action):
                scores[item] = float(action[idx])
            else:
                scores[item] = 0.5

        return scores

    def _get_observation(self) -> np.ndarray:
        state = np.zeros(self.state_size, dtype=np.float32)

        state[0] = len(self.cache) / self.cache.capacity

        state[1] = self.current_request / self.num_items

        freq = self.cache.access_frequency.get(self.current_request, 0)
        state[2] = min(freq / 10.0, 1.0)

        if self.current_request in self.cache:
            state[3] = 1.0

        recent_unique = len(set(self.request_history[-20:]))
        state[4] = recent_unique / 20.0 if self.request_history else 0.0

        metrics = self.cache.get_metrics()
        state[5] = metrics['hit_rate']

        if len(self.request_history) >= 5:
            recent_requests = self.request_history[-5:]
            state[6] = 1.0 if self.current_request in recent_requests else 0.0

        cache_items = list(self.cache.cache.keys())
        for i, item in enumerate(cache_items[:10]):
            freq = self.cache.access_frequency.get(item, 0)
            state[7 + i] = min(freq / 10.0, 1.0)

        state[17] = metrics.get('rl_influence_rate', 0.0)

        if len(self.request_history) >= 3:
            pattern = [self.request_history[-3], self.request_history[-2], self.request_history[-1]]
            state[18] = 1.0 if self.current_request in pattern else 0.0

        return state

    def render(self):
        pass
