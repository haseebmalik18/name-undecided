from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
import numpy as np


class CacheSimulator:
    def __init__(
        self,
        capacity: int,
        miss_penalty: float = 10.0,
        hit_latency: float = 1.0,
        item_size: int = 1
    ):
        self.capacity = capacity
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

    def admit(self, item_id: Any) -> bool:
        if item_id in self.cache:
            return True

        if len(self.cache) < self.capacity:
            self.cache[item_id] = True
            self.admissions += 1
            return True
        else:
            return False

    def evict(self, item_id: Optional[Any] = None) -> Optional[Any]:
        if not self.cache:
            return None

        if item_id is not None and item_id in self.cache:
            del self.cache[item_id]
            evicted = item_id
        else:
            evicted, _ = self.cache.popitem(last=False)

        self.evictions += 1
        return evicted

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
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, item_id: Any) -> bool:
        return item_id in self.cache
