"""Phase III: an ImageNet-pretrained EfficientNet-B0 fine-tuned on FER-2013.

FER faces are 48x48 grayscale; the backbone was pretrained on 224x224 color
photos. Inside the model the face is upscaled to 96x96, copied to three
channels and rescaled to 0-255 (EfficientNet does its own normalization), so
the same 48x48 arrays used by the CNN feed this model unchanged.
"""

import numpy as np
from keras import Input, Model, layers
from keras.applications import EfficientNetB0

from fer.config import EMOTIONS, IMG_SIZE

UPSCALED = 96


def build_transfer_model(weights="imagenet", num_classes=len(EMOTIONS)):
    """Returns (model, backbone). weights=None builds the same graph without downloading."""
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Resizing(UPSCALED, UPSCALED, interpolation="bilinear")(inputs)
    x = layers.Concatenate()([x, x, x])
    x = layers.Rescaling(255.0)(x)
    # Augmentation layers are only active during training.
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    backbone = EfficientNetB0(include_top=False, weights=weights, pooling="avg")
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="efficientnet_b0_fer"), backbone


def unfreeze_backbone(backbone):
    """Fine-tune every layer except batch norm, whose ImageNet statistics are kept."""
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def softened_class_weights(y):
    """Square root of the 'balanced' weights.

    Full balancing (9.4x for disgust) lifted rare-class recall in Phase II but
    cost overall accuracy; the square root keeps most of the recall gain.
    """
    counts = np.bincount(y, minlength=len(EMOTIONS)).astype("float64")
    balanced = len(y) / (len(EMOTIONS) * np.maximum(counts, 1))
    return {i: float(np.sqrt(w)) for i, w in enumerate(balanced)}


def predict_with_flip(model, X, batch_size=256):
    """Test-time augmentation: average the prediction for each face and its mirror image."""
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    probs += model.predict(X[:, :, ::-1, :], batch_size=batch_size, verbose=0)
    return probs / 2.0
