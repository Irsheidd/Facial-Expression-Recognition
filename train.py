"""Train the custom CNN on FER-2013.

    python train.py --data-dir data/fer2013 --out-dir models

--data-dir must contain train/ and test/, each with one folder per emotion.
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras import callbacks
from sklearn.model_selection import train_test_split

from fer.config import TrainConfig
from fer.data import augmenting_generator, balanced_class_weights, load_split
from fer.model import build_custom_cnn, compile_model


def plot_history(history, path):
    fig, (acc, loss) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in ((acc, "accuracy", "Accuracy"), (loss, "loss", "Loss")):
        ax.plot(history[key], label=f"train {key}")
        ax.plot(history[f"val_{key}"], label=f"validation {key}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("models"), type=Path)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--limit-per-class", type=int, help="load only N images per class (smoke tests)")
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    keras.utils.set_random_seed(cfg.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_split(args.data_dir / "train", args.limit_per_class)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.validation_split, random_state=cfg.seed, stratify=y
    )
    print(f"train {len(X_train)}  validation {len(X_val)}")

    model = compile_model(build_custom_cnn(), cfg.learning_rate)
    best_path = args.out_dir / "best_model.keras"
    history = model.fit(
        augmenting_generator(cfg).flow(X_train, y_train, batch_size=cfg.batch_size, seed=cfg.seed),
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        class_weight=balanced_class_weights(y_train),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
        verbose=2,
    )

    model.save(args.out_dir / "custom_cnn_final.keras")
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (args.out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, args.out_dir / "training_curves.png")
    print(f"saved model, history and curves to {args.out_dir}")


if __name__ == "__main__":
    main()
