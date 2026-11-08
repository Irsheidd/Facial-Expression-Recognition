import numpy as np
from keras import layers

from fer.config import EMOTIONS, IMG_SIZE
from fer.data import balanced_class_weights
from fer.transfer import build_transfer_model, predict_with_flip, softened_class_weights, unfreeze_backbone


def test_transfer_model_takes_the_same_48x48_input_as_the_cnn():
    # weights=None builds the same graph without downloading ImageNet weights
    model, _ = build_transfer_model(weights=None)
    probs = model.predict(np.zeros((2, IMG_SIZE, IMG_SIZE, 1), dtype="float32"), verbose=0)
    assert probs.shape == (2, len(EMOTIONS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_unfreezing_keeps_batch_norm_frozen():
    _, backbone = build_transfer_model(weights=None)
    backbone.trainable = False
    unfreeze_backbone(backbone)
    batch_norm = [layer for layer in backbone.layers if isinstance(layer, layers.BatchNormalization)]
    others = [layer for layer in backbone.layers if layer.weights and layer not in batch_norm]
    assert batch_norm and not any(layer.trainable for layer in batch_norm)
    assert all(layer.trainable for layer in others)


def test_softened_weights_are_square_root_of_balanced():
    y = np.array([0] * 100 + [1] * 10 + [2] * 100 + [3] * 50 + [4] * 50 + [5] * 50 + [6] * 50)
    balanced = balanced_class_weights(y)
    softened = softened_class_weights(y)
    for k in balanced:
        assert np.isclose(softened[k], np.sqrt(balanced[k]))
    assert softened[1] == max(softened.values())


class _FirstRowModel:
    """Stand-in model whose 'probabilities' are the first row of pixels."""

    def predict(self, X, batch_size=None, verbose=0):
        return X[:, 0, :, 0].copy()


def test_flip_prediction_averages_image_and_mirror():
    X = np.array([[1.0, 2.0, 6.0]]).reshape(1, 1, 3, 1)
    np.testing.assert_allclose(predict_with_flip(_FirstRowModel(), X), [[3.5, 2.0, 3.5]])
