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
    TimeOfDayWorkload,
    ZipfWorkload
)
from cache.policies import LRUCache, LFUCache
from visualization.dynamic_plots import (
    plot_workload_pattern,
    plot_rl_influence_heatmap,
    plot_training_comparison
)


def create_workload(workload_type: str, num_items: int, cache_size: int, seed: int = 42):
    workloads = {
        'temporal_shift': TemporalShiftWorkload(
            num_items=num_items,
            phase_length=200,
            num_popular_sets=3,
            popular_set_size=20,
            alpha=1.5,
            seed=seed
        ),
        'popularity_spike': PopularitySpikeWorkload(
            num_items=num_items,
            alpha=1.5,
            spike_probability=0.01,
            spike_duration=50,
            spike_intensity=0.9,
            seed=seed
        ),
        'workload_drift': WorkloadDriftWorkload(
            num_items=num_items,
            drift_rate=0.01,
            base_alpha=1.5,
            seed=seed
        ),
        'adversarial_scan': AdversarialLRUWorkload(
            num_items=num_items,
            cache_size=cache_size,
            pattern='scan',
            seed=seed
        ),
        'adversarial_loop': AdversarialLRUWorkload(
            num_items=num_items,
            cache_size=cache_size,
            pattern='loop',
            seed=seed
        ),
        'time_of_day': TimeOfDayWorkload(
            num_items=num_items,
            cycle_length=500,
            num_cycles=4,
            phase_overlap=0.1,
            seed=seed
        ),
        'static_zipf': ZipfWorkload(
            num_items=num_items,
            alpha=1.5,
            seed=seed
        )
    }

    if workload_type not in workloads:
        raise ValueError(f"Unknown workload type: {workload_type}")

    return workloads[workload_type]


def evaluate_baseline(workload_type: str, cache_size: int, num_items: int,
                      num_requests: int, policy: str = 'lru'):
    workload = create_workload(workload_type, num_items, cache_size)

    if policy == 'lru':
        cache = LRUCache(cache_size)
    elif policy == 'lfu':
        cache = LFUCache(cache_size)
    else:
        cache = LRUCache(cache_size)

    requests = workload.generate(num_requests)

    for req in requests:
        cache.access(req)

    metrics = cache.get_metrics()

    return {
        'hit_rate': metrics['hit_rate'],
        'total_accesses': metrics['total_accesses'],
        'hits': metrics['hits'],
        'misses': metrics['misses']
    }


def evaluate_rl_agent(model_path: str, workload_type: str, cache_size: int,
                      num_items: int, num_requests: int, rl_weight: float = 0.5,
                      base_policy: str = 'lru'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workload = create_workload(workload_type, num_items, cache_size)

    env = HybridCacheEnv(
        cache_capacity=cache_size,
        num_items=num_items,
        workload_generator=workload,
        episode_length=num_requests,
        state_size=30,
        base_policy=base_policy,
        rl_weight=rl_weight
    )

    agent = PriorityAgent(
        state_size=30,
        action_size=cache_size,
        device=device
    )

    agent.load(model_path)
    agent.epsilon = 0.0

    state, _ = env.reset()
    total_hits = 0
    total_requests = 0
    rl_influence_count = 0

    for _ in range(num_requests):
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        total_requests += 1
        if info.get('hit_rate', 0) > 0:
            total_hits = int(info['hit_rate'] * total_requests)

        if info.get('rl_influenced', 0) > 0:
            rl_influence_count += 1

        state = next_state

        if done or truncated:
            break

    metrics = env.cache.get_metrics()

    return {
        'hit_rate': metrics['hit_rate'],
        'rl_influence_rate': metrics.get('rl_influence_rate', 0),
        'total_accesses': metrics['total_accesses'],
        'hits': metrics['hits'],
        'misses': metrics['misses']
    }


def run_comprehensive_evaluation(
    workload_types: list,
    cache_size: int = 100,
    num_items: int = 1000,
    num_requests: int = 10000,
    model_prefix: str = './dynamic_agent',
    output_dir: str = './evaluation_results'
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    print("=" * 80)
    print("COMPREHENSIVE EVALUATION OF DYNAMIC WORKLOADS")
    print("=" * 80)

    for workload_type in workload_types:
        print(f"\nEvaluating workload: {workload_type}")
        print("-" * 80)

        baseline_lru = evaluate_baseline(workload_type, cache_size, num_items,
                                         num_requests, policy='lru')
        print(f"Baseline LRU Hit Rate: {baseline_lru['hit_rate']:.4f}")

        baseline_lfu = evaluate_baseline(workload_type, cache_size, num_items,
                                         num_requests, policy='lfu')
        print(f"Baseline LFU Hit Rate: {baseline_lfu['hit_rate']:.4f}")

        model_path = f"{model_prefix}_{workload_type}_model.pth"
        if Path(model_path).exists():
            rl_metrics = evaluate_rl_agent(model_path, workload_type, cache_size,
                                          num_items, num_requests)
            print(f"RL-Enhanced Hit Rate: {rl_metrics['hit_rate']:.4f}")
            print(f"RL Influence Rate: {rl_metrics['rl_influence_rate']:.4f}")

            improvement_lru = ((rl_metrics['hit_rate'] - baseline_lru['hit_rate']) /
                              baseline_lru['hit_rate']) * 100
            improvement_lfu = ((rl_metrics['hit_rate'] - baseline_lfu['hit_rate']) /
                              baseline_lfu['hit_rate']) * 100

            print(f"Improvement over LRU: {improvement_lru:.2f}%")
            print(f"Improvement over LFU: {improvement_lfu:.2f}%")

            results[workload_type] = {
                'baseline_lru': baseline_lru,
                'baseline_lfu': baseline_lfu,
                'rl_enhanced': rl_metrics,
                'improvement_lru_pct': improvement_lru,
                'improvement_lfu_pct': improvement_lfu
            }

            metrics_path = f"{model_prefix}_{workload_type}_metrics.json"
            if Path(metrics_path).exists():
                plot_rl_influence_heatmap(
                    metrics_path,
                    save_path=f"{output_dir}/{workload_type}_rl_influence.png"
                )

                with open(metrics_path, 'r') as f:
                    training_metrics = json.load(f)

                plot_training_comparison(
                    baseline_metrics={'hit_rate': baseline_lru['hit_rate']},
                    rl_metrics=training_metrics,
                    save_path=f"{output_dir}/{workload_type}_comparison.png"
                )

            workload = create_workload(workload_type, num_items, cache_size)
            sample_requests = workload.generate(5000)
            plot_workload_pattern(
                sample_requests,
                save_path=f"{output_dir}/{workload_type}_pattern.png"
            )

        else:
            print(f"Model not found at {model_path}, skipping RL evaluation")
            results[workload_type] = {
                'baseline_lru': baseline_lru,
                'baseline_lfu': baseline_lfu,
                'rl_enhanced': None
            }

    with open(f"{output_dir}/comprehensive_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for workload_type, data in results.items():
        print(f"\n{workload_type}:")
        print(f"  LRU: {data['baseline_lru']['hit_rate']:.4f}")
        print(f"  LFU: {data['baseline_lfu']['hit_rate']:.4f}")
        if data['rl_enhanced']:
            print(f"  RL:  {data['rl_enhanced']['hit_rate']:.4f} "
                  f"({data['improvement_lru_pct']:+.2f}% vs LRU)")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate hybrid RL cache on dynamic workloads')
    parser.add_argument('--workloads', nargs='+',
                       default=['temporal_shift', 'popularity_spike', 'time_of_day'],
                       help='Workload types to evaluate')
    parser.add_argument('--cache-size', type=int, default=100)
    parser.add_argument('--num-items', type=int, default=1000)
    parser.add_argument('--num-requests', type=int, default=10000)
    parser.add_argument('--model-prefix', type=str, default='./dynamic_agent')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')

    args = parser.parse_args()

    run_comprehensive_evaluation(
        workload_types=args.workloads,
        cache_size=args.cache_size,
        num_items=args.num_items,
        num_requests=args.num_requests,
        model_prefix=args.model_prefix,
        output_dir=args.output_dir
    )
