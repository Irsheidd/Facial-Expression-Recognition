"""Loading FER-2013 from its folder layout (<split>/<emotion>/<image>.png)."""

from pathlib import Path

import numpy as np
from keras.utils import img_to_array, load_img
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # legacy API kept for parity with the notebook

from fer.config import EMOTIONS, IMG_SIZE, TrainConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_split(split_dir, limit_per_class=None):
    """Load one split as grayscale 48x48 images scaled to [0, 1].

    Labels follow the alphabetical folder order, which is EMOTIONS. A folder
    set that does not match EMOTIONS raises, so a wrong path fails loudly
    instead of training on mislabeled data.
    """
    split_dir = Path(split_dir)
    found = tuple(sorted(p.name for p in split_dir.iterdir() if p.is_dir()))
    if found != EMOTIONS:
        raise ValueError(f"expected class folders {EMOTIONS} in {split_dir}, found {found}")

    images, labels = [], []
    for label, emotion in enumerate(EMOTIONS):
        files = sorted(p for p in (split_dir / emotion).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if limit_per_class:
            files = files[:limit_per_class]
        for path in files:
            img = load_img(path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
            images.append(img_to_array(img) / 255.0)
            labels.append(label)

    return np.asarray(images, dtype="float32"), np.asarray(labels, dtype="int64")


def balanced_class_weights(y):
    """Weights inversely proportional to class frequency ('disgust' is ~1.5% of FER-2013)."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def augmenting_generator(cfg: TrainConfig):
    return ImageDataGenerator(
        rotation_range=cfg.rotation_range,
        width_shift_range=cfg.width_shift_range,
        height_shift_range=cfg.height_shift_range,
        horizontal_flip=cfg.horizontal_flip,
        zoom_range=cfg.zoom_range,
        fill_mode="nearest",
    )
