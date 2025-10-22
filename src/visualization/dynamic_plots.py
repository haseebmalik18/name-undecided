import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from typing import List, Dict, Any
from collections import Counter


def plot_workload_pattern(requests: List[int], window_size: int = 100, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    unique_counts = []
    for i in range(0, len(requests) - window_size, window_size):
        window = requests[i:i + window_size]
        unique_counts.append(len(set(window)))

    axes[0, 0].plot(unique_counts)
    axes[0, 0].set_title('Unique Items per Window')
    axes[0, 0].set_xlabel('Window Index')
    axes[0, 0].set_ylabel('Unique Items')
    axes[0, 0].grid(True, alpha=0.3)

    heatmap_data = []
    num_bins = 50
    items_per_bin = max(requests) // num_bins + 1
    for i in range(0, len(requests), window_size):
        window = requests[i:i + window_size]
        bins = [0] * num_bins
        for item in window:
            bin_idx = min(item // items_per_bin, num_bins - 1)
            bins[bin_idx] += 1
        heatmap_data.append(bins)

    if heatmap_data:
        heatmap_data = np.array(heatmap_data).T
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Frequency'})
        axes[0, 1].set_title('Access Pattern Heatmap')
        axes[0, 1].set_xlabel('Time Window')
        axes[0, 1].set_ylabel('Item Range')

    item_freq = Counter(requests[:1000])
    top_items = item_freq.most_common(20)
    items, freqs = zip(*top_items) if top_items else ([], [])

    axes[1, 0].bar(range(len(items)), freqs)
    axes[1, 0].set_title('Top 20 Most Accessed Items (First 1000 Requests)')
    axes[1, 0].set_xlabel('Item Rank')
    axes[1, 0].set_ylabel('Access Count')
    axes[1, 0].grid(True, alpha=0.3)

    access_gaps = {}
    for idx, item in enumerate(requests):
        if item in access_gaps:
            access_gaps[item].append(idx - access_gaps[item][-1])
        else:
            access_gaps[item] = [idx]

    avg_gaps = []
    for item, gaps in list(access_gaps.items())[:100]:
        if len(gaps) > 1:
            avg_gaps.append(np.mean(gaps[1:]))

    if avg_gaps:
        axes[1, 1].hist(avg_gaps, bins=30, edgecolor='black')
        axes[1, 1].set_title('Distribution of Average Access Gaps')
        axes[1, 1].set_xlabel('Average Gap Between Accesses')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_rl_influence_heatmap(
    metrics_path: str,
    save_path: str = None,
    episode_window: int = 10
):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    rl_influence = metrics.get('episode_rl_influence', [])
    hit_rates = metrics.get('episode_hit_rates', [])

    if not rl_influence or not hit_rates:
        print("No RL influence or hit rate data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(rl_influence, alpha=0.6, label='Raw')
    if len(rl_influence) > episode_window:
        smoothed = np.convolve(rl_influence, np.ones(episode_window)/episode_window, mode='valid')
        axes[0, 0].plot(range(episode_window-1, len(rl_influence)), smoothed,
                       linewidth=2, label=f'Smoothed ({episode_window} ep)')
    axes[0, 0].set_title('RL Influence Rate Over Training')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('RL Influence Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(hit_rates, alpha=0.6, label='Raw')
    if len(hit_rates) > episode_window:
        smoothed = np.convolve(hit_rates, np.ones(episode_window)/episode_window, mode='valid')
        axes[0, 1].plot(range(episode_window-1, len(hit_rates)), smoothed,
                       linewidth=2, label=f'Smoothed ({episode_window} ep)')
    axes[0, 1].set_title('Hit Rate Over Training')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Hit Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    min_len = min(len(rl_influence), len(hit_rates))
    axes[1, 0].scatter(rl_influence[:min_len], hit_rates[:min_len], alpha=0.5, s=10)
    axes[1, 0].set_title('RL Influence vs Hit Rate')
    axes[1, 0].set_xlabel('RL Influence Rate')
    axes[1, 0].set_ylabel('Hit Rate')
    axes[1, 0].grid(True, alpha=0.3)

    if len(rl_influence) >= 100:
        heatmap_data = []
        chunk_size = 20
        for i in range(0, len(rl_influence), chunk_size):
            chunk = rl_influence[i:i + chunk_size]
            if len(chunk) == chunk_size:
                heatmap_data.append(chunk)

        if heatmap_data:
            heatmap_array = np.array(heatmap_data).T
            sns.heatmap(heatmap_array, cmap='RdYlGn', ax=axes[1, 1],
                       cbar_kws={'label': 'RL Influence'})
            axes[1, 1].set_title('RL Influence Heatmap Over Time')
            axes[1, 1].set_xlabel('Episode Chunk')
            axes[1, 1].set_ylabel('Episode in Chunk')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_training_comparison(
    baseline_metrics: Dict[str, Any],
    rl_metrics: Dict[str, Any],
    save_path: str = None
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    baseline_hr = baseline_metrics.get('hit_rate', 0)
    rl_hit_rates = rl_metrics.get('episode_hit_rates', [])

    if rl_hit_rates:
        axes[0, 0].axhline(y=baseline_hr, color='r', linestyle='--',
                          linewidth=2, label='Baseline (LRU)')
        axes[0, 0].plot(rl_hit_rates, alpha=0.6, label='RL-Enhanced')

        smoothed = np.convolve(rl_hit_rates, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(rl_hit_rates)), smoothed,
                       linewidth=2, label='RL-Enhanced (Smoothed)')

        axes[0, 0].set_title('Hit Rate: Baseline vs RL-Enhanced')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Hit Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    rewards = rl_metrics.get('episode_rewards', [])
    if rewards:
        axes[0, 1].plot(rewards, alpha=0.6, label='Raw')
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[0, 1].plot(range(9, len(rewards)), smoothed,
                       linewidth=2, label='Smoothed')
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    if rl_hit_rates:
        improvement = [(hr - baseline_hr) / baseline_hr * 100 for hr in rl_hit_rates]
        axes[1, 0].plot(improvement, alpha=0.6)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].fill_between(range(len(improvement)), improvement, 0,
                                where=np.array(improvement) > 0, alpha=0.3,
                                color='green', label='Improvement')
        axes[1, 0].fill_between(range(len(improvement)), improvement, 0,
                                where=np.array(improvement) <= 0, alpha=0.3,
                                color='red', label='Degradation')
        axes[1, 0].set_title('Performance Improvement Over Baseline (%)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    latencies = rl_metrics.get('episode_avg_latency', [])
    if latencies:
        axes[1, 1].plot(latencies, alpha=0.6, label='Raw')
        smoothed = np.convolve(latencies, np.ones(10)/10, mode='valid')
        axes[1, 1].plot(range(9, len(latencies)), smoothed,
                       linewidth=2, label='Smoothed')
        axes[1, 1].set_title('Average Latency Over Training')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Avg Latency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_temporal_features_analysis(
    state_history: List[np.ndarray],
    save_path: str = None
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    time_since_access = [state[19] for state in state_history if len(state) > 19]
    if time_since_access:
        axes[0, 0].plot(time_since_access, alpha=0.6)
        axes[0, 0].set_title('Time Since Last Access (Current Request)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Normalized Time')
        axes[0, 0].grid(True, alpha=0.3)

    access_trend = [state[20] for state in state_history if len(state) > 20]
    if access_trend:
        axes[0, 1].plot(access_trend, alpha=0.6)
        axes[0, 1].set_title('Access Trend (Current Request)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Trend Score')
        axes[0, 1].grid(True, alpha=0.3)

    hit_rates = [state[5] for state in state_history if len(state) > 5]
    if hit_rates:
        axes[1, 0].plot(hit_rates, alpha=0.6)
        axes[1, 0].set_title('Running Hit Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Hit Rate')
        axes[1, 0].grid(True, alpha=0.3)

    occupancy = [state[0] for state in state_history if len(state) > 0]
    if occupancy:
        axes[1, 1].plot(occupancy, alpha=0.6)
        axes[1, 1].set_title('Cache Occupancy')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Occupancy Ratio')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
