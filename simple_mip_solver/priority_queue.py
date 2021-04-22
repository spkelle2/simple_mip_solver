from operator import attrgetter
from typing import List, Any


class PriorityQueue:

    def __init__(self, metric: str, initial_items: List[Any] = None):
        assert isinstance(metric, str), 'metric must be a string'
        if initial_items:
            assert isinstance(initial_items, list), 'initial items must be list'
            assert all(hasattr(n, metric) for n in initial_items), \
                f'each item must have a {metric} attribute'
            assert all(isinstance(getattr(n, metric), float) or
                       isinstance(getattr(n, metric), int) for n in initial_items), \
                f"each item's {metric} must be an integer or float"
        self._metric = metric
        self._items = initial_items or []

    def __bool__(self):
        return bool(self._items)

    def push(self, item):
        assert hasattr(item, self._metric), f'added item must have {self._metric} attribute'
        assert isinstance(getattr(item, self._metric), float) or \
            isinstance(getattr(item, self._metric), int), \
            f"each item's {self._metric} must be an integer or float"
        self._items.append(item)

    def min(self):
        return min(self._items, key=attrgetter(self._metric))

    def pop(self):
        n = self.min()
        self._items.remove(n)
        return n

    def bound(self, bound):
        assert isinstance(bound, float) or isinstance(bound, int)
        self._items = [item for item in self._items if
                       getattr(item, self._metric) < bound]
