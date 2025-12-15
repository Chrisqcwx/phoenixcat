from collections import deque, Counter
from typing import Iterable, Set, Any, Literal


class IntervalCounter:
    def __init__(self, interval: int, couter_type: Literal['start', 'end'] = 'start'):
        self.interval = interval
        self.count = 0
        self.couter_type = couter_type

    def step(self):
        # self.count += 1
        # return self.is_interval()
        if self.couter_type == 'start':
            result = self.is_interval()
            self.count += 1
        else:
            self.count += 1
            result = self.is_interval()
        return result

    def reset(self):
        self.count = 0

    def is_interval(self):
        return self.count % self.interval == 0


class RecentKCommon:

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.window: deque[Set[Any]] = deque()
        self.counts: Counter = Counter()

    def push(self, lst: Iterable[Any]) -> Set[Any]:
        s = set(lst)
        self.window.append(s)
        for e in s:
            self.counts[e] += 1

        if len(self.window) > self.k:
            old = self.window.popleft()
            for e in old:
                self.counts[e] -= 1
                if self.counts[e] == 0:
                    del self.counts[e]
        return self.current_common()

    def current_common(self) -> Set[Any]:
        return {e for e, c in self.counts.items() if c == self.k}
