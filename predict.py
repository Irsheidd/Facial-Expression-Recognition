"""Predict the expression in one or more photos.

    python predict.py --model models/custom_cnn_final.keras photo1.jpg photo2.png

Each photo is converted to grayscale and the largest face is cropped before
prediction. If no face is found the whole image is used and a warning is printed.
"""

import argparse
from pathlib import Path

import cv2
import keras
import numpy as np

from fer.config import EMOTIONS, IMG_SIZE
from fer.face import crop_largest_face


def predict_file(model, path, top_k=3, cropped=False):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if cropped:
        face, found = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)), True
    else:
        face, found = crop_largest_face(gray)
    probs = model.predict(face[None, :, :, None].astype("float32") / 255.0, verbose=0)[0]
    ranked = np.argsort(probs)[::-1][:top_k]
    return found, [(EMOTIONS[i], float(probs[i])) for i in ranked]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--cropped", action="store_true", help="images are already face crops (e.g. FER-2013 test images)")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    for path in args.images:
        found, top = predict_file(model, path, cropped=args.cropped)
        ranked = ", ".join(f"{emotion} {p:.0%}" for emotion, p in top)
        warning = "" if found else "  [no face detected: used the whole image, low reliability]"
        print(f"{path.name}: {ranked}{warning}")


if __name__ == "__main__":
    main()
