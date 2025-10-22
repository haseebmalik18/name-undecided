import json
import statistics

def mean(data):
    return sum(data) / len(data)

def std(data):
    return statistics.stdev(data) if len(data) > 1 else 0

with open('metrics_hybrid_colab.json', 'r') as f:
    data = json.load(f)

episode_hit_rates = data['episode_hit_rates']
episode_rewards = data['episode_rewards']
losses = data['losses']

print("=" * 60)
print("HYBRID RL-ENHANCED CACHE TRAINING RESULTS")
print("=" * 60)

print(f"\nTotal Episodes: {len(episode_hit_rates)}")
print(f"Total Training Steps: {len(losses)}")

print("\n" + "-" * 60)
print("HIT RATE ANALYSIS")
print("-" * 60)

first_100 = mean(episode_hit_rates[:100])
last_100 = mean(episode_hit_rates[-100:])
overall_mean = mean(episode_hit_rates)
overall_std = std(episode_hit_rates)

print(f"First 100 episodes:  {first_100:.4f}")
print(f"Last 100 episodes:   {last_100:.4f}")
print(f"Overall mean:        {overall_mean:.4f} ± {overall_std:.4f}")
print(f"Improvement:         {((last_100 - first_100) / first_100 * 100):.2f}%")

print("\n" + "-" * 60)
print("REWARD ANALYSIS")
print("-" * 60)

first_100_reward = mean(episode_rewards[:100])
last_100_reward = mean(episode_rewards[-100:])
overall_reward = mean(episode_rewards)

print(f"First 100 episodes:  {first_100_reward:.2f}")
print(f"Last 100 episodes:   {last_100_reward:.2f}")
print(f"Overall mean:        {overall_reward:.2f}")

print("\n" + "-" * 60)
print("TRAINING STABILITY")
print("-" * 60)

final_loss = mean(losses[-1000:]) if len(losses) > 1000 else mean(losses[-100:])
print(f"Final loss (last 1k steps): {final_loss:.6f}")

hit_rate_variance_early = std(episode_hit_rates[:100])
hit_rate_variance_late = std(episode_hit_rates[-100:])
print(f"Hit rate variance (first 100): {hit_rate_variance_early:.4f}")
print(f"Hit rate variance (last 100):  {hit_rate_variance_late:.4f}")

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"Final hit rate: {last_100:.4f}")
print(f"Training converged: {'Yes' if hit_rate_variance_late < 0.02 else 'No'}")
print(f"Performance trend: {'Improving' if last_100 > first_100 else 'Stable/Declining'}")
print("=" * 60)
