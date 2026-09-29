"""Classify channel ordering once, retain the historical tie precedence."""
from itertools import product
import cv2 as cv
import numpy as np
from .interactive import ArrayCache


def _tables():
    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    tables = {}
    for mode, inclusive in product(('min', 'avg', 'max'), (False, True)):
        table = np.zeros((256, 1, 3), np.uint8)
        for b, g, r in product(range(3), repeat=3):
            key = (b > g)+(b >= g)+3*((b > r)+(b >= r))+9*((g > r)+(g >= r))
            selected = 0
            for index, value in enumerate((b, g, r)):
                others = [v for j, v in enumerate((b, g, r)) if j != index]
                low, high = min(others), max(others)
                if mode == 'min':
                    match = value <= low if inclusive else value < low
                elif mode == 'max':
                    match = value >= high if inclusive else value > high
                else:
                    match = low <= value <= high if inclusive else low < value < high
                if match:
                    selected = index+1  # Original assignment order: blue, green, red.
            table[key, 0] = colors[selected]
        tables[mode, inclusive] = table
    return tables


TABLES = _tables()


class StatsEngine:
    def __init__(self, image):
        self.image = image
        self.order = ArrayCache(128)
        self.results = ArrayCache(384)

    def ranking(self):
        rank = self.order.get('rank')
        if rank is None:
            b, g, r = cv.split(self.image)
            rank = (b > g).astype(np.uint8)
            rank += b >= g
            for left, right, weight in ((b, r, 3), (g, r, 9)):
                relation = (left > right).astype(np.uint8)
                relation += left >= right
                relation *= weight
                rank += relation
            self.order.put('rank', rank)
        return rank

    def compute(self, params):
        cached=self.results.get(params)
        if cached is not None:return cached
        from .memory_resources import MEMORY,MiB
        def bounded():
            from .bounded_ops import local_result
            result=local_result(self.image,lambda roi:StatsEngine(roi)._compute(params),halo=0)[0]
            return self.results.put(params,result)
        return MEMORY.execute(self.image.shape[0]*self.image.shape[1]*12,64*MiB,
                              lambda:self._compute(params),bounded)

    def _compute(self, params):
        result = self.results.get(params)
        if result is None:
            result = cv.applyColorMap(self.ranking(), TABLES[params])
            self.results.put(params, result)
        return result
