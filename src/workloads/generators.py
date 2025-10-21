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
