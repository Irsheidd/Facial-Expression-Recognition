import cv2
import numpy as np
import pytest

from fer.config import EMOTIONS, IMG_SIZE
from fer.data import balanced_class_weights, load_split
from fer.face import crop_largest_face
from fer.model import build_custom_cnn, compile_model


def test_model_outputs_one_probability_per_emotion():
    model = compile_model(build_custom_cnn())
    probs = model.predict(np.zeros((2, IMG_SIZE, IMG_SIZE, 1), dtype="float32"), verbose=0)
    assert probs.shape == (2, len(EMOTIONS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_model_matches_notebook_parameter_count():
    # notebooks/02_custom_cnn.ipynb reports 7,187,911 total parameters
    assert build_custom_cnn().count_params() == 7_187_911


def _write_fake_split(root, per_class=2):
    rng = np.random.default_rng(0)
    for emotion in EMOTIONS:
        (root / emotion).mkdir(parents=True)
        for i in range(per_class):
            cv2.imwrite(str(root / emotion / f"{i}.png"), rng.integers(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))


def test_load_split_labels_follow_emotion_order(tmp_path):
    _write_fake_split(tmp_path)
    X, y = load_split(tmp_path)
    assert X.shape == (2 * len(EMOTIONS), IMG_SIZE, IMG_SIZE, 1)
    assert 0.0 <= X.min() and X.max() <= 1.0
    assert list(y) == [i for i in range(len(EMOTIONS)) for _ in range(2)]


def test_load_split_rejects_unexpected_class_folders(tmp_path):
    (tmp_path / "happy").mkdir()
    with pytest.raises(ValueError, match="expected class folders"):
        load_split(tmp_path)


def test_rare_class_gets_largest_weight():
    y = np.array([0] * 100 + [1] * 10 + [2] * 100)
    weights = balanced_class_weights(y)
    assert weights[1] > weights[0] and weights[1] > weights[2]


def test_crop_falls_back_to_whole_image_without_a_face():
    face, found = crop_largest_face(np.full((120, 160), 128, dtype=np.uint8))
    assert not found
    assert face.shape == (IMG_SIZE, IMG_SIZE)
