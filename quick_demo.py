import sys
sys.path.append('src')

from agents.environment import CacheEnv
from agents.dqn_agent import DQNAgent
from workloads.generators import ZipfWorkload
from cache.policies import LRUCache
import numpy as np


def quick_demo():
    CACHE_SIZE = 50
    NUM_ITEMS = 500
    EPISODES = 50
    EPISODE_LENGTH = 500

    workload = ZipfWorkload(num_items=NUM_ITEMS, alpha=1.5, seed=42)

    env = CacheEnv(
        cache_capacity=CACHE_SIZE,
        num_items=NUM_ITEMS,
        workload_generator=workload,
        episode_length=EPISODE_LENGTH
    )

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=0.001
    )

    dqn_hit_rates = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(EPISODE_LENGTH):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state

            if done or truncated:
                break

        metrics = env.cache.get_metrics()
        dqn_hit_rates.append(metrics['hit_rate'])

    dqn_final_hit_rate = np.mean(dqn_hit_rates[-10:])

    lru = LRUCache(capacity=CACHE_SIZE)
    lru_hit_rates = []

    workload_lru = ZipfWorkload(num_items=NUM_ITEMS, alpha=1.5, seed=42)

    for episode in range(EPISODES):
        lru.reset()
        requests = workload_lru.generate(EPISODE_LENGTH)

        for req in requests:
            lru.access(req)

        lru_hit_rates.append(lru.get_hit_rate())

    lru_final_hit_rate = np.mean(lru_hit_rates[-10:])

    print(f"\nDQN: {dqn_final_hit_rate:.3f}")
    print(f"LRU: {lru_final_hit_rate:.3f}")

    if dqn_final_hit_rate > lru_final_hit_rate:
        improvement = ((dqn_final_hit_rate - lru_final_hit_rate) / lru_final_hit_rate) * 100
        print(f"Improvement: +{improvement:.2f}%")


if __name__ == '__main__':
    quick_demo()
