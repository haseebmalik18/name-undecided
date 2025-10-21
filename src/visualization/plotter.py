import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import pandas as pd

sns.set_style('darkgrid')


def plot_training_metrics(metrics_tracker, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    episodes = range(len(metrics_tracker.episode_rewards))

    axes[0, 0].plot(episodes, metrics_tracker.episode_rewards, alpha=0.6, label='Reward')
    if len(metrics_tracker.episode_rewards) > 10:
        window = min(50, len(metrics_tracker.episode_rewards) // 10)
        smoothed = pd.Series(metrics_tracker.episode_rewards).rolling(window=window).mean()
        axes[0, 0].plot(episodes, smoothed, linewidth=2, label=f'Smoothed (w={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(episodes, metrics_tracker.episode_hit_rates, alpha=0.6, label='Hit Rate')
    if len(metrics_tracker.episode_hit_rates) > 10:
        window = min(50, len(metrics_tracker.episode_hit_rates) // 10)
        smoothed = pd.Series(metrics_tracker.episode_hit_rates).rolling(window=window).mean()
        axes[0, 1].plot(episodes, smoothed, linewidth=2, label=f'Smoothed (w={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Hit Rate')
    axes[0, 1].set_title('Cache Hit Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(episodes, metrics_tracker.episode_latencies, alpha=0.6)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Latency')
    axes[1, 0].set_title('Average Latency per Episode')
    axes[1, 0].grid(True)

    if metrics_tracker.losses:
        loss_steps = range(len(metrics_tracker.losses))
        axes[1, 1].plot(loss_steps, metrics_tracker.losses, alpha=0.6)
        if len(metrics_tracker.losses) > 100:
            smoothed = pd.Series(metrics_tracker.losses).rolling(window=100).mean()
            axes[1, 1].plot(loss_steps, smoothed, linewidth=2, label='Smoothed (w=100)')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comparison(results: Dict[str, Dict[str, List]], save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, metrics in results.items():
        episodes = range(len(metrics['hit_rates']))
        axes[0].plot(episodes, metrics['hit_rates'], label=name, alpha=0.7)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Hit Rate')
    axes[0].set_title('Hit Rate Comparison')
    axes[0].legend()
    axes[0].grid(True)

    for name, metrics in results.items():
        episodes = range(len(metrics['rewards']))
        axes[1].plot(episodes, metrics['rewards'], label=name, alpha=0.7)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].set_title('Reward Comparison')
    axes[1].legend()
    axes[1].grid(True)

    names = list(results.keys())
    final_hit_rates = [np.mean(results[name]['hit_rates'][-100:]) for name in names]

    axes[2].bar(names, final_hit_rates, alpha=0.7)
    axes[2].set_ylabel('Hit Rate')
    axes[2].set_title('Average Hit Rate (Last 100 Episodes)')
    axes[2].grid(True, axis='y')

    for i, v in enumerate(final_hit_rates):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_cache_heatmap(access_frequency: Dict, cache_items: List, num_items: int, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))

    items = list(range(num_items))
    frequencies = [access_frequency.get(i, 0) for i in items]
    in_cache = [1 if i in cache_items else 0 for i in items]

    data = np.array([frequencies, in_cache])

    sns.heatmap(data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frequency / In Cache'})
    ax.set_yticklabels(['Access Freq', 'In Cache'])
    ax.set_xlabel('Item ID')
    ax.set_title('Cache Occupancy vs Access Frequency')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
