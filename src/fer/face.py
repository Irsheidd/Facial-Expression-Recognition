"""Face cropping for real photos.

The model was trained on tightly cropped 48x48 faces. Feeding it a whole photo
resized to 48x48 mostly shows it background, so prediction crops the largest
detected face first (Haar cascade, same parameters as the Phase I notebook).
"""

import cv2
import numpy as np

from fer.config import IMG_SIZE

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def crop_largest_face(gray, size=IMG_SIZE):
    """Return (face resized to size x size, whether a face was found).

    With no detection the whole image is resized instead, so callers can still
    predict but should treat the result as unreliable.
    """
    gray = np.asarray(gray, dtype=np.uint8)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return cv2.resize(gray, (size, size)), False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(gray[y : y + h, x : x + w], (size, size)), True
