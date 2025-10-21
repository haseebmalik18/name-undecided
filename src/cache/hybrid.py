from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class HybridCache:
    def __init__(
        self,
        capacity: int,
        base_policy: str = 'lru',
        rl_weight: float = 0.5,
        miss_penalty: float = 10.0,
        hit_latency: float = 1.0,
        item_size: int = 1
    ):
        self.capacity = capacity
        self.base_policy = base_policy.lower()
        self.rl_weight = rl_weight
        self.miss_penalty = miss_penalty
        self.hit_latency = hit_latency
        self.item_size = item_size

        self.cache: OrderedDict = OrderedDict()
        self.access_frequency: Dict[Any, int] = {}
        self.last_access_time: Dict[Any, int] = {}
        self.current_time = 0

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.admissions = 0
        self.total_latency = 0.0
        self.bandwidth_used = 0

        self.rl_influenced_evictions = 0

    def reset(self):
        self.cache.clear()
        self.access_frequency.clear()
        self.last_access_time.clear()
        self.current_time = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.admissions = 0
        self.total_latency = 0.0
        self.bandwidth_used = 0
        self.rl_influenced_evictions = 0

    def access(self, item_id: Any) -> Tuple[bool, float]:
        self.current_time += 1

        self.access_frequency[item_id] = self.access_frequency.get(item_id, 0) + 1
        self.last_access_time[item_id] = self.current_time

        if item_id in self.cache:
            self.hits += 1
            latency = self.hit_latency
            self.total_latency += latency
            self.cache.move_to_end(item_id)
            return True, latency
        else:
            self.misses += 1
            latency = self.miss_penalty
            self.total_latency += latency
            self.bandwidth_used += self.item_size
            return False, latency

    def admit(self, item_id: Any, rl_eviction_scores: Optional[Dict[Any, float]] = None) -> Optional[Any]:
        if item_id in self.cache:
            return None

        evicted = None
        if len(self.cache) >= self.capacity:
            evicted = self._evict_with_rl(rl_eviction_scores)

        self.cache[item_id] = True
        self.admissions += 1

        return evicted

    def _evict_with_rl(self, rl_scores: Optional[Dict[Any, float]] = None) -> Any:
        if not self.cache:
            return None

        cache_items = list(self.cache.keys())

        if rl_scores is None or len(rl_scores) == 0:
            evicted = self._base_eviction()
        else:
            base_scores = self._get_base_scores(cache_items)
            combined_scores = {}

            for item in cache_items:
                base_score = base_scores.get(item, 0.0)
                rl_score = rl_scores.get(item, 0.0)
                combined_scores[item] = (1 - self.rl_weight) * base_score + self.rl_weight * rl_score

            evicted = max(combined_scores, key=combined_scores.get)
            del self.cache[evicted]

            base_only_evict = max(base_scores, key=base_scores.get)
            if evicted != base_only_evict:
                self.rl_influenced_evictions += 1

        self.evictions += 1
        return evicted

    def _base_eviction(self) -> Any:
        if self.base_policy == 'lru':
            evicted, _ = self.cache.popitem(last=False)
            return evicted
        elif self.base_policy == 'lfu':
            cache_items = list(self.cache.keys())
            min_freq_item = min(cache_items, key=lambda x: self.access_frequency.get(x, 0))
            del self.cache[min_freq_item]
            return min_freq_item
        else:
            evicted, _ = self.cache.popitem(last=False)
            return evicted

    def _get_base_scores(self, items: List[Any]) -> Dict[Any, float]:
        scores = {}

        if self.base_policy == 'lru':
            max_time = max([self.last_access_time.get(item, 0) for item in items], default=1)
            for item in items:
                recency = self.current_time - self.last_access_time.get(item, 0)
                scores[item] = recency / max(max_time, 1)

        elif self.base_policy == 'lfu':
            max_freq = max([self.access_frequency.get(item, 0) for item in items], default=1)
            for item in items:
                freq = self.access_frequency.get(item, 0)
                scores[item] = 1.0 - (freq / max(max_freq, 1))

        else:
            for i, item in enumerate(items):
                scores[item] = i / max(len(items), 1)

        return scores

    def get_state(self) -> Dict[str, Any]:
        cache_items = list(self.cache.keys())

        return {
            'cache_size': len(self.cache),
            'capacity': self.capacity,
            'occupancy': len(self.cache) / self.capacity,
            'cache_items': cache_items,
            'access_frequency': self.access_frequency.copy(),
            'last_access_time': self.last_access_time.copy(),
            'current_time': self.current_time,
        }

    def get_metrics(self) -> Dict[str, float]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        avg_latency = self.total_latency / total_requests if total_requests > 0 else 0.0
        rl_influence_rate = self.rl_influenced_evictions / max(self.evictions, 1)

        return {
            'hit_rate': hit_rate,
            'miss_rate': 1 - hit_rate,
            'total_requests': total_requests,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'admissions': self.admissions,
            'avg_latency': avg_latency,
            'bandwidth_used': self.bandwidth_used,
            'rl_influenced_evictions': self.rl_influenced_evictions,
            'rl_influence_rate': rl_influence_rate,
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, item_id: Any) -> bool:
        return item_id in self.cache
