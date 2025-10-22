import numpy as np
from typing import List, Iterator, Optional


class ZipfWorkload:
    def __init__(self, num_items: int, alpha: float = 1.5, seed: Optional[int] = None):
        self.num_items = num_items
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def generate(self, num_requests: int) -> List[int]:
        return self.rng.zipf(self.alpha, num_requests) % self.num_items

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.rng.zipf(self.alpha) % self.num_items


class PoissonWorkload:
    def __init__(self, num_items: int, lam: float = 10.0, seed: Optional[int] = None):
        self.num_items = num_items
        self.lam = lam
        self.rng = np.random.RandomState(seed)

    def generate(self, num_requests: int) -> List[int]:
        return self.rng.poisson(self.lam, num_requests) % self.num_items

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.rng.poisson(self.lam) % self.num_items


class TraceWorkload:
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.trace_data = self._load_trace()
        self.index = 0

    def _load_trace(self) -> List[int]:
        with open(self.trace_file, 'r') as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            requests.append(self.trace_data[self.index % len(self.trace_data)])
            self.index += 1
        return requests

    def reset(self):
        self.index = 0

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.trace_data[self.index % len(self.trace_data)]
            self.index += 1


class MixedWorkload:
    def __init__(
        self,
        num_items: int,
        phases: List[dict],
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.phases = phases
        self.rng = np.random.RandomState(seed)
        self.current_phase = 0
        self.requests_in_phase = 0

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            phase = self.phases[self.current_phase % len(self.phases)]
            phase_type = phase['type']
            phase_length = phase.get('length', float('inf'))

            if phase_type == 'zipf':
                alpha = phase.get('alpha', 1.5)
                item = self.rng.zipf(alpha) % self.num_items
            elif phase_type == 'uniform':
                item = self.rng.randint(0, self.num_items)
            elif phase_type == 'poisson':
                lam = phase.get('lambda', 10.0)
                item = self.rng.poisson(lam) % self.num_items
            else:
                item = self.rng.randint(0, self.num_items)

            requests.append(item)
            self.requests_in_phase += 1

            if self.requests_in_phase >= phase_length:
                self.current_phase += 1
                self.requests_in_phase = 0

        return requests


class TemporalShiftWorkload:
    def __init__(
        self,
        num_items: int,
        phase_length: int = 200,
        num_popular_sets: int = 3,
        popular_set_size: int = 20,
        alpha: float = 1.5,
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.phase_length = phase_length
        self.num_popular_sets = num_popular_sets
        self.popular_set_size = popular_set_size
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        self.request_count = 0

        self.popular_sets = []
        items = list(range(num_items))
        self.rng.shuffle(items)
        for i in range(num_popular_sets):
            start = i * popular_set_size
            end = start + popular_set_size
            self.popular_sets.append(items[start:end])

    def _current_phase(self) -> int:
        return (self.request_count // self.phase_length) % self.num_popular_sets

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            phase = self._current_phase()
            popular_items = self.popular_sets[phase]

            if self.rng.random() < 0.8:
                idx = min(self.rng.zipf(self.alpha), len(popular_items)) - 1
                item = popular_items[idx % len(popular_items)]
            else:
                item = self.rng.randint(0, self.num_items)

            requests.append(item)
            self.request_count += 1

        return requests

    def reset(self):
        self.request_count = 0


class PopularitySpikeWorkload:
    def __init__(
        self,
        num_items: int,
        alpha: float = 1.5,
        spike_probability: float = 0.01,
        spike_duration: int = 50,
        spike_intensity: float = 0.9,
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.alpha = alpha
        self.spike_probability = spike_probability
        self.spike_duration = spike_duration
        self.spike_intensity = spike_intensity
        self.rng = np.random.RandomState(seed)

        self.current_spike_item = None
        self.spike_remaining = 0
        self.request_count = 0

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            if self.spike_remaining == 0 and self.rng.random() < self.spike_probability:
                self.current_spike_item = self.rng.randint(0, self.num_items)
                self.spike_remaining = self.spike_duration

            if self.spike_remaining > 0:
                if self.rng.random() < self.spike_intensity:
                    item = self.current_spike_item
                else:
                    item = self.rng.zipf(self.alpha) % self.num_items
                self.spike_remaining -= 1
            else:
                item = self.rng.zipf(self.alpha) % self.num_items

            requests.append(item)
            self.request_count += 1

        return requests

    def reset(self):
        self.current_spike_item = None
        self.spike_remaining = 0
        self.request_count = 0


class WorkloadDriftWorkload:
    def __init__(
        self,
        num_items: int,
        drift_rate: float = 0.01,
        base_alpha: float = 1.5,
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.drift_rate = drift_rate
        self.base_alpha = base_alpha
        self.rng = np.random.RandomState(seed)

        self.weights = np.ones(num_items)
        self.request_count = 0

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            probs = self.weights / self.weights.sum()
            item = self.rng.choice(self.num_items, p=probs)
            requests.append(item)

            if self.rng.random() < self.drift_rate:
                drift_item = self.rng.randint(0, self.num_items)
                self.weights[drift_item] *= self.rng.uniform(1.5, 2.5)

            self.weights *= 0.999
            self.weights = np.maximum(self.weights, 0.1)
            self.request_count += 1

        return requests

    def reset(self):
        self.weights = np.ones(self.num_items)
        self.request_count = 0


class AdversarialLRUWorkload:
    def __init__(
        self,
        num_items: int,
        cache_size: int,
        pattern: str = 'scan',
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.cache_size = cache_size
        self.pattern = pattern
        self.rng = np.random.RandomState(seed)
        self.request_count = 0

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            if self.pattern == 'scan':
                item = self.request_count % (self.cache_size + 1)
            elif self.pattern == 'loop':
                item = self.request_count % (self.cache_size * 2)
            else:
                item = self.rng.randint(0, self.num_items)

            requests.append(item)
            self.request_count += 1

        return requests

    def reset(self):
        self.request_count = 0


class TimeOfDayWorkload:
    def __init__(
        self,
        num_items: int,
        cycle_length: int = 500,
        num_cycles: int = 4,
        phase_overlap: float = 0.1,
        seed: Optional[int] = None
    ):
        self.num_items = num_items
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles
        self.phase_overlap = phase_overlap
        self.rng = np.random.RandomState(seed)
        self.request_count = 0

        self.cycle_items = []
        items_per_cycle = num_items // num_cycles
        for i in range(num_cycles):
            start = i * items_per_cycle
            end = start + items_per_cycle
            self.cycle_items.append(list(range(start, end)))

    def generate(self, num_requests: int) -> List[int]:
        requests = []
        for _ in range(num_requests):
            position = (self.request_count % self.cycle_length) / self.cycle_length
            phase = int(position * self.num_cycles)

            main_items = self.cycle_items[phase]

            if self.rng.random() < self.phase_overlap:
                adjacent_phase = (phase + 1) % self.num_cycles
                main_items = main_items + self.cycle_items[adjacent_phase]

            if len(main_items) > 0:
                idx = min(self.rng.zipf(1.5), len(main_items)) - 1
                item = main_items[idx % len(main_items)]
            else:
                item = self.rng.randint(0, self.num_items)

            requests.append(item)
            self.request_count += 1

        return requests

    def reset(self):
        self.request_count = 0
