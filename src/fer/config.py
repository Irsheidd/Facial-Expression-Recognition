"""Settings shared by training, evaluation and prediction.

The defaults are the ones used in notebooks/02_custom_cnn.ipynb, so a run of
train.py with no flags reproduces the notebook's setup.
"""

from dataclasses import dataclass

# Class folders in FER-2013, in the alphabetical order the loader assigns labels.
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMG_SIZE = 48


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    seed: int = 42
    # Augmentation, applied to training images only.
    rotation_range: float = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
