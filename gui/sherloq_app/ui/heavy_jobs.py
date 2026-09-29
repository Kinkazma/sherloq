"""Application-wide FIFO admission for expensive isolated analyses.

One active comparison/Noiseprint/TruFor/PRNU analysis at a time. Interactive workers
and file loading keep their separate bounded pools; no GUI thread waits here.
"""
from collections import deque
import weakref
from PySide6.QtCore import QTimer, QCoreApplication


class HeavyJobs:
    def __init__(self):
        self.queue = deque()
        self.active = None

    def request(self, owner):
        if self.active is not None and self.active() is owner:
            raise RuntimeError('Heavy analysis is already admitted.')
        if any(reference() is owner for reference in self.queue):
            return
        self.queue.append(weakref.ref(owner))
        self._schedule()

    def release(self, owner):
        self.queue = deque(reference for reference in self.queue if reference() not in (None, owner))
        if self.active is not None and self.active() is owner:
            self.active = None
        self._schedule()

    def _schedule(self):
        if self.queue and QCoreApplication.instance() is not None and not QCoreApplication.closingDown():
            QTimer.singleShot(0, self._pump)

    def _pump(self):
        if self.active is not None and self.active() is not None:
            return
        self.active = None
        while self.queue:
            reference = self.queue.popleft()
            owner = reference()
            if owner is not None:
                self.active = reference
                owner._start_admitted()
                return


HEAVY_JOBS = HeavyJobs()
