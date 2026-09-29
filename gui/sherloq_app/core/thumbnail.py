"""Decode the embedded thumbnail without a temporary file; preserve OpenCV math."""
import cv2 as cv
import numpy as np


def analyze_thumbnail(data, image):
    if not data:
        return None
    thumbnail = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
    if thumbnail is None:
        return None
    resized = cv.resize(thumbnail, image.shape[1::-1], interpolation=cv.INTER_LANCZOS4)
    return resized, cv.absdiff(image, resized)
