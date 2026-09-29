"""Shared LRU budget for retained NumPy buffers, not a process RSS limit.

Displayed/active arrays and native model/GPU allocations can outlive eviction.
The registry keeps weak cache references; closing a panel never pins its engine.
"""
from collections import OrderedDict
import os
from threading import RLock
import weakref
import numpy as np


def default_limit():
    try:
        physical = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError):
        physical = 8 * 1024**3
    return min(8 * 1024**3, max(128 * 1024**2, physical // 8))


class CacheBudget:
    def __init__(self, limit):
        self.limit = int(limit)
        self.bytes = 0
        self.peak = 0
        self.evictions = 0
        self.entries = OrderedDict()
        self.lock = RLock()

    def register(self, cache):
        def gone(reference):
            with self.lock:
                for token in list(self.entries):
                    if token[0] is reference:
                        self.bytes -= self.entries.pop(token)
        return weakref.ref(cache, gone)

    def remove(self, reference, key):
        self.bytes -= self.entries.pop((reference, key), 0)

    def reserve(self, size):
        # Caller holds the same lock as every cache operation.
        if size > self.limit:
            return
        while self.entries and self.bytes + size > self.limit:
            (reference, key), count = self.entries.popitem(last=False)
            self.bytes -= count
            owner = reference()
            if owner is not None:
                owner._remove_local(key)
            self.evictions += 1

    def snapshot(self):
        with self.lock:
            return dict(limit=self.limit, bytes=self.bytes, peak=self.peak,
                        entries=len(self.entries), evictions=self.evictions)

    def clear(self):
        with self.lock:
            for reference, key in list(self.entries):
                owner = reference()
                if owner is not None:
                    owner._remove_local(key)
            self.entries.clear()
            self.bytes = 0


GLOBAL_CACHE_BUDGET = CacheBudget(default_limit())


class ArrayCache:
    def __init__(self, megabytes=128, *, budget=None):
        self.limit = int(megabytes * 1024 * 1024)
        self.bytes = 0
        self.items = OrderedDict()
        self.sizes = {}
        self.hits = 0
        self.budget = GLOBAL_CACHE_BUDGET if budget is None else budget
        self.reference = self.budget.register(self)

    @staticmethod
    def size(value):
        # Charge an entire retained base, not only a small slice's logical size.
        # Aliases within one entry count once; across entries charging is
        # deliberately conservative and may evict earlier than strictly needed.
        seen = set()
        def count(item):
            if id(item) in seen:
                return 0
            seen.add(id(item))
            if isinstance(item, np.ndarray):
                if isinstance(item.base, np.ndarray):
                    return count(item.base)
                return max(item.nbytes, getattr(item.base, 'nbytes', 0))
            if isinstance(item, dict):
                return sum(count(v) for v in item.values())
            if isinstance(item, (tuple, list)):
                return sum(count(v) for v in item)
            return 0
        return count(value)

    def _remove_local(self, key):
        self.items.pop(key, None)
        self.bytes -= self.sizes.pop(key, 0)

    def _remove(self, key):
        self._remove_local(key)
        self.budget.remove(self.reference, key)

    def get(self, key):
        with self.budget.lock:
            if key not in self.items:
                return None
            self.items.move_to_end(key)
            self.budget.entries.move_to_end((self.reference, key))
            self.hits += 1
            return self.items[key]

    def reserve(self, size):
        with self.budget.lock:
            if size <= min(self.limit, self.budget.limit):
                while self.items and self.bytes + size > self.limit:
                    self._remove(next(iter(self.items)))
                self.budget.reserve(size)

    def put(self, key, value):
        size = self.size(value)
        with self.budget.lock:
            self._remove(key)
            if size <= min(self.limit, self.budget.limit):
                self.reserve(size)
                self.items[key] = value
                self.sizes[key] = size
                self.bytes += size
                self.budget.entries[self.reference, key] = size
                self.budget.bytes += size
                self.budget.peak = max(self.budget.peak, self.budget.bytes)
        return value

    def clear(self):
        with self.budget.lock:
            for key in list(self.items):
                self._remove(key)
