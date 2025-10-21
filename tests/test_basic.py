import sys
sys.path.append('../src')

from cache.simulator import CacheSimulator
from cache.policies import LRUCache, LFUCache
from workloads.generators import ZipfWorkload, PoissonWorkload
from agents.environment import CacheEnv
from agents.dqn_agent import DQNAgent


def test_cache_simulator():
    cache = CacheSimulator(capacity=10, miss_penalty=10.0, hit_latency=1.0)

    is_hit, latency = cache.access(1)
    assert is_hit == False
    assert latency == 10.0

    cache.admit(1)
    is_hit, latency = cache.access(1)
    assert is_hit == True
    assert latency == 1.0

    metrics = cache.get_metrics()
    assert metrics['hits'] == 1
    assert metrics['misses'] == 1
    assert metrics['hit_rate'] == 0.5


def test_lru_cache():
    lru = LRUCache(capacity=3)

    lru.access(1)
    lru.access(2)
    lru.access(3)

    assert lru.access(1) == True
    assert lru.access(2) == True

    lru.access(4)

    assert 3 not in lru.cache
    assert 1 in lru.cache
    assert 2 in lru.cache
    assert 4 in lru.cache


def test_workload_generators():
    zipf = ZipfWorkload(num_items=100, alpha=1.5, seed=42)
    requests = zipf.generate(1000)
    assert len(requests) == 1000
    assert all(0 <= r < 100 for r in requests)

    poisson = PoissonWorkload(num_items=100, lam=10.0, seed=42)
    requests = poisson.generate(1000)
    assert len(requests) == 1000
    assert all(0 <= r < 100 for r in requests)


def test_cache_env():
    workload = ZipfWorkload(num_items=100, alpha=1.5, seed=42)
    env = CacheEnv(
        cache_capacity=10,
        num_items=100,
        workload_generator=workload,
        episode_length=100
    )

    state, info = env.reset()
    assert state.shape == (env.state_size,)

    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)

    assert next_state.shape == (env.state_size,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_dqn_agent():
    agent = DQNAgent(state_size=10, action_size=2)

    state = [0.5] * 10
    action = agent.select_action(state, training=False)
    assert action in [0, 1]

    agent.store_transition(state, action, 1.0, state, False)
    agent.store_transition(state, action, 1.0, state, False)

    loss = agent.train()


if __name__ == '__main__':
    test_cache_simulator()
    test_lru_cache()
    test_workload_generators()
    test_cache_env()
    test_dqn_agent()
    print("All tests passed")
