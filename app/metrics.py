import threading
from collections import defaultdict
from typing import Dict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

    def inc(self, key: str, value: int = 1):
        with self._lock:
            self.counters[key] += value

    def observe(self, key: str, value: float):
        with self._lock:
            self.timers[key].append(value)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "timers": {
                    k: {
                        "count": len(v),
                        "avg_ms": (sum(v) / len(v)) * 1000 if v else 0.0,
                        "p95_ms": sorted(v)[int(0.95 * len(v))] * 1000 if v else 0.0,
                    }
                    for k, v in self.timers.items()
                },
            }


metrics = Metrics()
