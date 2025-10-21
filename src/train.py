import argparse
import numpy as np
from tqdm import tqdm
import os

from agents.environment import CacheEnv
from agents.dqn_agent import DQNAgent
from workloads.generators import ZipfWorkload
from metrics.tracker import MetricsTracker
from visualization.plotter import plot_training_metrics
from cache.policies import LRUCache


def train_dqn(
    cache_capacity=100,
    num_items=1000,
    episodes=1000,
    episode_length=1000,
    alpha=1.5,
    learning_rate=0.001,
    save_dir='checkpoints'
):
    os.makedirs(save_dir, exist_ok=True)

    workload = ZipfWorkload(num_items=num_items, alpha=alpha, seed=42)

    env = CacheEnv(
        cache_capacity=cache_capacity,
        num_items=num_items,
        workload_generator=workload,
        episode_length=episode_length
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate
    )

    metrics = MetricsTracker()

    for episode in tqdm(range(episodes), desc="Training"):
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
            print(f"\nEp {episode + 1}: Hit Rate {stats['mean_hit_rate']:.3f}, Reward {stats['mean_reward']:.2f}, ε {agent.epsilon:.3f}")

        if (episode + 1) % 500 == 0:
            agent.save(os.path.join(save_dir, f'dqn_agent_ep{episode + 1}.pth'))
            metrics.save(os.path.join(save_dir, f'metrics_ep{episode + 1}.json'))

    agent.save(os.path.join(save_dir, 'dqn_agent_final.pth'))
    metrics.save(os.path.join(save_dir, 'metrics_final.json'))

    return agent, metrics


def evaluate_lru(
    cache_capacity=100,
    num_items=1000,
    episodes=100,
    episode_length=1000,
    alpha=1.5
):
    workload = ZipfWorkload(num_items=num_items, alpha=alpha, seed=42)
    lru = LRUCache(capacity=cache_capacity)

    hit_rates = []

    for episode in tqdm(range(episodes), desc="LRU Eval"):
        lru.reset()
        requests = workload.generate(episode_length)

        for req in requests:
            lru.access(req)

        hit_rates.append(lru.get_hit_rate())

    mean_hit_rate = np.mean(hit_rates)
    std_hit_rate = np.std(hit_rates)
    print(f"LRU: {mean_hit_rate:.3f} ± {std_hit_rate:.3f}")

    return {'hit_rates': hit_rates, 'rewards': [0] * episodes}


def main():
    parser = argparse.ArgumentParser(description='Train RL Cache Management Agent')
    parser.add_argument('--cache-size', type=int, default=100, help='Cache capacity')
    parser.add_argument('--num-items', type=int, default=1000, help='Number of unique items')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--alpha', type=float, default=1.5, help='Zipf distribution parameter')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--plot', action='store_true', help='Plot training metrics')

    args = parser.parse_args()

    agent, metrics = train_dqn(
        cache_capacity=args.cache_size,
        num_items=args.num_items,
        episodes=args.episodes,
        episode_length=args.episode_length,
        alpha=args.alpha,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )

    lru_results = evaluate_lru(
        cache_capacity=args.cache_size,
        num_items=args.num_items,
        episodes=100,
        episode_length=args.episode_length,
        alpha=args.alpha
    )

    if args.plot:
        plot_training_metrics(metrics, save_path=os.path.join(args.save_dir, 'training_plot.png'))


if __name__ == '__main__':
    main()
