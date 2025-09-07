import time
from typing import Dict
from threading import Lock


class MetricsService:
    def __init__(self):
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.start_time = time.time()
            self.total_queries = 0
            self.fallback_count = 0
            self.tool_counts: Dict[str, int] = {}
            self.intent_counts: Dict[str, int] = {}

    def record_query(self) -> None:
        with self._lock:
            self.total_queries += 1

    def inc_tool(self, tool_name: str) -> None:
        if not tool_name:
            return
        with self._lock:
            self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1

    def inc_fallback(self) -> None:
        with self._lock:
            self.fallback_count += 1

    def inc_intent(self, intent_name: str) -> None:
        if not intent_name:
            return
        with self._lock:
            self.intent_counts[intent_name] = self.intent_counts.get(intent_name, 0) + 1

    def get_stats(self) -> Dict:
        with self._lock:
            elapsed = max(1.0, time.time() - self.start_time)
            return {
                'total_queries': self.total_queries,
                'fallback_count': self.fallback_count,
                'fallback_rate': (self.fallback_count / self.total_queries) if self.total_queries else 0.0,
                'tool_counts': dict(self.tool_counts),
                'intent_counts': dict(self.intent_counts),
                'qps': self.total_queries / elapsed,
                'uptime_seconds': elapsed,
            }


metrics_service = MetricsService()


