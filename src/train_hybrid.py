import argparse
import numpy as np
from tqdm import tqdm
import os

from agents.hybrid_environment import HybridCacheEnv
from agents.priority_agent import PriorityAgent
from workloads.generators import ZipfWorkload
from metrics.tracker import MetricsTracker
from visualization.plotter import plot_training_metrics, plot_comparison
from cache.policies import LRUCache


def train_hybrid(
    cache_capacity=100,
    num_items=1000,
    episodes=1000,
    episode_length=1000,
    alpha=1.5,
    base_policy='lru',
    rl_weight=0.5,
    learning_rate=0.001,
    save_dir='checkpoints_hybrid'
):
    os.makedirs(save_dir, exist_ok=True)

    workload = ZipfWorkload(num_items=num_items, alpha=alpha, seed=42)

    env = HybridCacheEnv(
        cache_capacity=cache_capacity,
        num_items=num_items,
        workload_generator=workload,
        episode_length=episode_length,
        base_policy=base_policy,
        rl_weight=rl_weight
    )

    state_size = env.observation_space.shape[0]
    output_size = cache_capacity

    agent = PriorityAgent(
        state_size=state_size,
        output_size=output_size,
        learning_rate=learning_rate
    )

    metrics = MetricsTracker()

    for episode in tqdm(range(episodes), desc="Training Hybrid"):
        state, _ = env.reset()
        total_reward = 0

        for step in range(episode_length):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()

            if loss is not None:
                metrics.add_loss(loss)

            total_reward += reward
            state = next_state

            if done or truncated:
                break

        cache_metrics = env.cache.get_metrics()
        metrics.add_episode(
            total_reward,
            cache_metrics['hit_rate'],
            cache_metrics['avg_latency'],
            cache_metrics['bandwidth_used']
        )

        if (episode + 1) % 100 == 0:
            stats = metrics.get_stats(window=100)
            rl_influence = cache_metrics.get('rl_influence_rate', 0.0)
            print(f"\nEp {episode + 1}: Hit Rate {stats['mean_hit_rate']:.3f}, "
                  f"Reward {stats['mean_reward']:.2f}, RL Influence {rl_influence:.2%}, ε {agent.epsilon:.3f}")

        if (episode + 1) % 500 == 0:
            agent.save(os.path.join(save_dir, f'hybrid_agent_ep{episode + 1}.pth'))
            metrics.save(os.path.join(save_dir, f'metrics_ep{episode + 1}.json'))

    agent.save(os.path.join(save_dir, 'hybrid_agent_final.pth'))
    metrics.save(os.path.join(save_dir, 'metrics_final.json'))

    return agent, metrics


def evaluate_baseline(
    cache_capacity=100,
    num_items=1000,
    episodes=100,
    episode_length=1000,
    alpha=1.5,
    policy='lru'
):
    if policy == 'lru':
        cache = LRUCache(capacity=cache_capacity)
    else:
        cache = LRUCache(capacity=cache_capacity)

    workload = ZipfWorkload(num_items=num_items, alpha=alpha, seed=42)

    hit_rates = []

    for episode in tqdm(range(episodes), desc=f"{policy.upper()} Eval"):
        cache.reset()
        requests = workload.generate(episode_length)

        for req in requests:
            cache.access(req)

        hit_rates.append(cache.get_hit_rate())

    mean_hit_rate = np.mean(hit_rates)
    std_hit_rate = np.std(hit_rates)
    print(f"{policy.upper()}: {mean_hit_rate:.3f} ± {std_hit_rate:.3f}")

    return {'hit_rates': hit_rates, 'rewards': [0] * episodes}


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid RL Cache Management')
    parser.add_argument('--cache-size', type=int, default=100, help='Cache capacity')
    parser.add_argument('--num-items', type=int, default=1000, help='Number of unique items')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--alpha', type=float, default=1.5, help='Zipf distribution parameter')
    parser.add_argument('--base-policy', type=str, default='lru', choices=['lru', 'lfu'], help='Base cache policy')
    parser.add_argument('--rl-weight', type=float, default=0.5, help='Weight for RL influence (0-1)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints_hybrid', help='Save directory')
    parser.add_argument('--plot', action='store_true', help='Plot training metrics')

    args = parser.parse_args()

    agent, metrics = train_hybrid(
        cache_capacity=args.cache_size,
        num_items=args.num_items,
        episodes=args.episodes,
        episode_length=args.episode_length,
        alpha=args.alpha,
        base_policy=args.base_policy,
        rl_weight=args.rl_weight,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )

    baseline_results = evaluate_baseline(
        cache_capacity=args.cache_size,
        num_items=args.num_items,
        episodes=100,
        episode_length=args.episode_length,
        alpha=args.alpha,
        policy=args.base_policy
    )

    if args.plot:
        plot_training_metrics(metrics, save_path=os.path.join(args.save_dir, 'training_plot.png'))

        results = {
            f'Hybrid ({args.base_policy.upper()} + RL)': {
                'hit_rates': metrics.episode_hit_rates,
                'rewards': metrics.episode_rewards
            },
            f'{args.base_policy.upper()} Only': baseline_results
        }
        plot_comparison(results, save_path=os.path.join(args.save_dir, 'comparison_plot.png'))


if __name__ == '__main__':
    main()
