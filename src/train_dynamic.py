import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys
sys.path.append('..')

from agents.priority_agent import PriorityAgent
from agents.hybrid_environment import HybridCacheEnv
from workloads.generators import (
    TemporalShiftWorkload,
    PopularitySpikeWorkload,
    WorkloadDriftWorkload,
    AdversarialLRUWorkload,
    TimeOfDayWorkload
)
from cache.policies import LRUCache, LFUCache


def train_with_dynamic_workload(
    workload_type: str,
    cache_size: int = 100,
    num_items: int = 1000,
    episodes: int = 1000,
    steps_per_episode: int = 1000,
    rl_weight: float = 0.5,
    base_policy: str = 'lru',
    lr: float = 0.001,
    save_path: str = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training with {workload_type} workload")

    if workload_type == 'temporal_shift':
        workload = TemporalShiftWorkload(
            num_items=num_items,
            phase_length=200,
            num_popular_sets=3,
            popular_set_size=20,
            alpha=1.5,
            seed=42
        )
    elif workload_type == 'popularity_spike':
        workload = PopularitySpikeWorkload(
            num_items=num_items,
            alpha=1.5,
            spike_probability=0.01,
            spike_duration=50,
            spike_intensity=0.9,
            seed=42
        )
    elif workload_type == 'workload_drift':
        workload = WorkloadDriftWorkload(
            num_items=num_items,
            drift_rate=0.01,
            base_alpha=1.5,
            seed=42
        )
    elif workload_type == 'adversarial_scan':
        workload = AdversarialLRUWorkload(
            num_items=num_items,
            cache_size=cache_size,
            pattern='scan',
            seed=42
        )
    elif workload_type == 'adversarial_loop':
        workload = AdversarialLRUWorkload(
            num_items=num_items,
            cache_size=cache_size,
            pattern='loop',
            seed=42
        )
    elif workload_type == 'time_of_day':
        workload = TimeOfDayWorkload(
            num_items=num_items,
            cycle_length=500,
            num_cycles=4,
            phase_overlap=0.1,
            seed=42
        )
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

    env = HybridCacheEnv(
        cache_capacity=cache_size,
        num_items=num_items,
        workload_generator=workload,
        episode_length=steps_per_episode,
        state_size=30,
        base_policy=base_policy,
        rl_weight=rl_weight,
        alpha=1.0,
        beta=0.1,
        gamma=0.05
    )

    agent = PriorityAgent(
        state_size=30,
        action_size=cache_size,
        lr=lr,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device=device
    )

    metrics = {
        'episode_rewards': [],
        'episode_hit_rates': [],
        'episode_rl_influence': [],
        'episode_avg_latency': [],
        'best_hit_rate': 0.0
    }

    print(f"\nStarting training for {episodes} episodes...")
    print(f"Config: cache_size={cache_size}, base_policy={base_policy}, rl_weight={rl_weight}")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_hit_rates = []

        for step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.train()

            total_reward += reward
            episode_hit_rates.append(info['hit_rate'])

            state = next_state

            if done or truncated:
                break

        cache_metrics = env.cache.get_metrics()
        avg_hit_rate = np.mean(episode_hit_rates)

        metrics['episode_rewards'].append(total_reward)
        metrics['episode_hit_rates'].append(avg_hit_rate)
        metrics['episode_rl_influence'].append(cache_metrics.get('rl_influence_rate', 0.0))
        metrics['episode_avg_latency'].append(cache_metrics.get('avg_latency', 0.0))

        if avg_hit_rate > metrics['best_hit_rate']:
            metrics['best_hit_rate'] = avg_hit_rate

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Hit Rate: {avg_hit_rate:.4f} | "
                  f"RL Influence: {cache_metrics.get('rl_influence_rate', 0.0):.4f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    print(f"\nTraining completed!")
    print(f"Best Hit Rate: {metrics['best_hit_rate']:.4f}")

    if save_path:
        model_path = f"{save_path}_{workload_type}_model.pth"
        metrics_path = f"{save_path}_{workload_type}_metrics.json"

        agent.save(model_path)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")

    baseline_hit_rate = evaluate_baseline(workload_type, cache_size, num_items, steps_per_episode, base_policy)
    improvement = ((metrics['best_hit_rate'] - baseline_hit_rate) / baseline_hit_rate) * 100

    print(f"\nBaseline ({base_policy.upper()}) Hit Rate: {baseline_hit_rate:.4f}")
    print(f"RL-Enhanced Hit Rate: {metrics['best_hit_rate']:.4f}")
    print(f"Improvement: {improvement:.2f}%")

    return metrics, agent


def evaluate_baseline(workload_type: str, cache_size: int, num_items: int, steps: int, policy: str):
    if workload_type == 'temporal_shift':
        workload = TemporalShiftWorkload(num_items=num_items, seed=42)
    elif workload_type == 'popularity_spike':
        workload = PopularitySpikeWorkload(num_items=num_items, seed=42)
    elif workload_type == 'workload_drift':
        workload = WorkloadDriftWorkload(num_items=num_items, seed=42)
    elif workload_type == 'adversarial_scan':
        workload = AdversarialLRUWorkload(num_items=num_items, cache_size=cache_size, pattern='scan', seed=42)
    elif workload_type == 'adversarial_loop':
        workload = AdversarialLRUWorkload(num_items=num_items, cache_size=cache_size, pattern='loop', seed=42)
    elif workload_type == 'time_of_day':
        workload = TimeOfDayWorkload(num_items=num_items, seed=42)
    else:
        return 0.0

    if policy == 'lru':
        cache = LRUCache(cache_size)
    elif policy == 'lfu':
        cache = LFUCache(cache_size)
    else:
        cache = LRUCache(cache_size)

    requests = workload.generate(steps)
    for req in requests:
        cache.access(req)

    return cache.get_metrics()['hit_rate']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hybrid RL cache with dynamic workloads')
    parser.add_argument('--workload', type=str, default='temporal_shift',
                        choices=['temporal_shift', 'popularity_spike', 'workload_drift',
                                'adversarial_scan', 'adversarial_loop', 'time_of_day'],
                        help='Type of dynamic workload')
    parser.add_argument('--cache-size', type=int, default=100, help='Cache capacity')
    parser.add_argument('--num-items', type=int, default=1000, help='Total number of items')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--rl-weight', type=float, default=0.5, help='Weight for RL influence (0-1)')
    parser.add_argument('--base-policy', type=str, default='lru', choices=['lru', 'lfu'],
                        help='Base eviction policy')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='./dynamic_agent',
                        help='Path prefix for saving models and metrics')

    args = parser.parse_args()

    train_with_dynamic_workload(
        workload_type=args.workload,
        cache_size=args.cache_size,
        num_items=args.num_items,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        rl_weight=args.rl_weight,
        base_policy=args.base_policy,
        lr=args.lr,
        save_path=args.save_path
    )
