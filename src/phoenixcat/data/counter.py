from typing import Literal


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
