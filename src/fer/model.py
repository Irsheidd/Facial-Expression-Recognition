"""The custom CNN from notebooks/02_custom_cnn.ipynb (64.63% test accuracy on FER-2013)."""

from keras import Input, Sequential, layers, optimizers

from fer.config import EMOTIONS, IMG_SIZE


def _conv_block(filters):
    return [
        layers.Conv2D(filters, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(filters, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    ]


def _dense_block(units):
    return [
        layers.Dense(units),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
    ]


def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTIONS)):
    """Four conv blocks (64-128-256-512) and two dense layers, about 7.19M parameters."""
    return Sequential(
        [Input(shape=input_shape)]
        + _conv_block(64)
        + _conv_block(128)
        + _conv_block(256)
        + _conv_block(512)
        + [layers.Flatten()]
        + _dense_block(512)
        + _dense_block(256)
        + [layers.Dense(num_classes, activation="softmax")],
        name="custom_cnn",
    )


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
