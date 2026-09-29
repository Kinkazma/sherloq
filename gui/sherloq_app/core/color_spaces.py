"""Lazy color channels with the historical uint8 conversion conventions."""
import cv2 as cv
import numpy as np
from .interactive import ArrayCache


class SpaceEngine:
    CODES = {'ycrcb': cv.COLOR_BGR2YCrCb, 'xyz': cv.COLOR_BGR2XYZ,
             'lab': cv.COLOR_BGR2Lab, 'luv': cv.COLOR_BGR2Luv}

    def __init__(self, image):
        self.image = image
        self.scaled = ArrayCache(256)
        self.spaces = ArrayCache(384)
        self.displays = ArrayCache(128)

    def normalized(self):
        scaled = self.scaled.get('bgr')
        if scaled is None:
            scaled = self.image.astype(np.float32)
            scaled /= 255
            self.scaled.put('bgr', scaled)
        return scaled

    def channel(self, space, index):
        if space == 'rgb':
            return self.image[:, :, 2-index]
        key = (space, index) if space == 'gray' else space
        value = self.spaces.get(key)
        if value is None:
            if space in self.CODES:
                value = cv.cvtColor(self.image, self.CODES[space])
            elif space in ('hsv', 'hls'):
                code = cv.COLOR_BGR2HSV if space == 'hsv' else cv.COLOR_BGR2HLS
                value = cv.cvtColor(self.normalized(), code)
                value *= 255
                value[:, :, 0] /= 360
                value = value.astype(np.uint8)
            elif space == 'gray':
                scaled = self.normalized()
                if index == 0:
                    high = np.maximum(np.maximum(scaled[:, :, 0], scaled[:, :, 1]), scaled[:, :, 2])
                    low = np.minimum(np.minimum(scaled[:, :, 0], scaled[:, :, 1]), scaled[:, :, 2])
                    value = (high + low) / 2
                elif index == 1:
                    value = .21*scaled[:, :, 2] + .72*scaled[:, :, 1] + .07*scaled[:, :, 0]
                elif index == 2:
                    value = np.mean(scaled, axis=2)
                elif index == 3:
                    value = cv.cvtColor(scaled, cv.COLOR_BGR2GRAY)
                else:
                    raise ValueError('Unknown grayscale channel')
                # The original four-channel array was float64 BEFORE *255.
                value = (value.astype(np.float64) * 255).astype(np.uint8)
            elif space == 'cmyk':
                scaled = self.normalized()
                # min(1-B, 1-G, 1-R) has the same float32 value as
                # 1-max(B, G, R), without a three-plane reduction temporary.
                black = 1-np.maximum(np.maximum(scaled[:, :, 0], scaled[:, :, 1]), scaled[:, :, 2])
                denominator = 1-black
                value = np.empty((*self.image.shape[:2], 4), np.uint8)
                for channel in range(3):
                    inverse = 1-scaled[:, :, 2-channel]
                    inverse -= black
                    # At RGB black, inverse is already zero: C=M=Y=0.
                    np.divide(inverse, denominator, out=inverse, where=denominator != 0)
                    inverse *= 255
                    value[:, :, channel] = inverse
                value[:, :, 3] = black*255
            else:
                raise ValueError('Unknown color space')
            self.spaces.put(key, value)
        return value if space == 'gray' else value[:, :, index]

    def compute(self, params):
        result = self.displays.get(params)
        if result is None:
            result = cv.cvtColor(self.channel(*params), cv.COLOR_GRAY2BGR)
            self.displays.put(params, result)
        return result
