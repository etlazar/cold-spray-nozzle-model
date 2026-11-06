import numpy as np


def colon(start, step, stop):
    n = int(np.floor((stop - start) / step + 1e-10))
    return start + step * np.arange(n + 1)


class Progress:
    def __init__(self, label, total, enabled=True, callback=None):
        self.label = label
        self.total = total
        self.enabled = enabled
        self.next_pct = 5
        # callback(fraction) is throttled to ~200 calls per loop
        self.callback = callback
        self.cb_every = max(1, total // 200)
        if enabled:
            print(f'{label}:   0%', end='', flush=True)

    def update(self, i):
        if self.callback is not None and i % self.cb_every == 0:
            self.callback(i / self.total)
        if not self.enabled:
            return
        pct = 100 * i / self.total
        if pct >= self.next_pct:
            print(f'\r{self.label}: {round(pct):3d}%', end='', flush=True)
            self.next_pct += 5

    def done(self):
        if self.enabled:
            print(f'\r{self.label}: 100%')
