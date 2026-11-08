"""Phase III: fine-tune an ImageNet-pretrained EfficientNet-B0 on FER-2013.

    python train_transfer.py --data-dir data/fer2013 --out-dir models/transfer

Stage 1 trains only the new classification head with the backbone frozen.
Stage 2 unfreezes the backbone (batch norm stays frozen) and fine-tunes
everything at a lower, cosine-decaying learning rate. The test split is only
touched once, at the end, with test-time flipping.
"""

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks, losses, optimizers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from fer.config import EMOTIONS
from fer.data import load_split
from fer.transfer import build_transfer_model, predict_with_flip, softened_class_weights, unfreeze_backbone


def dataset(X, y, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, keras.utils.to_categorical(y, len(EMOTIONS))))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("models/transfer"), type=Path)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, help="load only N images per class (smoke tests)")
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_split(args.data_dir / "train", args.limit_per_class)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    train_ds = dataset(X_train, y_train, args.batch_size, True, args.seed)
    val_ds = dataset(X_val, y_val, args.batch_size, False, args.seed)
    weights = softened_class_weights(y_train)
    loss = losses.CategoricalCrossentropy(label_smoothing=0.1)
    start = time.time()

    model, backbone = build_transfer_model()
    backbone.trainable = False
    model.compile(optimizer=optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    head = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, class_weight=weights, verbose=2)

    unfreeze_backbone(backbone)
    steps = args.finetune_epochs * len(train_ds)
    schedule = optimizers.schedules.CosineDecay(1e-4, decay_steps=steps, alpha=0.05)
    model.compile(optimizer=optimizers.Adam(schedule), loss=loss, metrics=["accuracy"])
    best_path = args.out_dir / "efficientnet_b0_fer.keras"
    fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=weights,
        callbacks=[
            callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )
    train_minutes = (time.time() - start) / 60

    X_test, y_test = load_split(args.data_dir / "test", args.limit_per_class)
    plain = np.argmax(model.predict(X_test, batch_size=256, verbose=0), axis=1)
    flipped = np.argmax(predict_with_flip(model, X_test), axis=1)
    report = classification_report(y_test, flipped, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS,
                                   digits=4, output_dict=True, zero_division=0)
    metrics = {
        "model": "EfficientNet-B0 (ImageNet) fine-tuned, 96x96 input",
        "test_samples": int(len(y_test)),
        "accuracy_without_tta": float(accuracy_score(y_test, plain)),
        "accuracy": float(accuracy_score(y_test, flipped)),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {e: report[e] for e in EMOTIONS},
        "confusion_matrix": confusion_matrix(y_test, flipped).tolist(),
        "train_minutes_cpu": round(train_minutes, 1),
        "history": {
            "head": {k: [float(v) for v in vals] for k, vals in head.history.items()},
            "finetune": {k: [float(v) for v in vals] for k, vals in fine.history.items()},
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"test accuracy {metrics['accuracy']:.4f} (without flip {metrics['accuracy_without_tta']:.4f}), "
          f"macro F1 {metrics['macro_f1']:.4f}, {train_minutes:.0f} min")


if __name__ == "__main__":
    main()
