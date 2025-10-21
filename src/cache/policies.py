from typing import Any, Optional
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, item_id: Any) -> bool:
        if item_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(item_id)
            return True
        else:
            self.misses += 1
            self.cache[item_id] = True
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            return False

    def reset(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.frequency = {}
        self.hits = 0
        self.misses = 0

    def access(self, item_id: Any) -> bool:
        if item_id in self.cache:
            self.hits += 1
            self.frequency[item_id] += 1
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                min_freq_item = min(self.frequency, key=self.frequency.get)
                del self.cache[min_freq_item]
                del self.frequency[min_freq_item]

            self.cache[item_id] = True
            self.frequency[item_id] = 1
            return False

    def reset(self):
        self.cache.clear()
        self.frequency.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
