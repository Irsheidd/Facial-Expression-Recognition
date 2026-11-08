"""Evaluate a trained model on the FER-2013 test split.

    python evaluate.py --model models/custom_cnn_final.keras --data-dir data/fer2013

Writes metrics.json (accuracy, macro/weighted F1, per-class report, top confusions)
and confusion_matrix.png to --out-dir.
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from fer.config import EMOTIONS
from fer.data import load_split


def top_confusions(cm, n=5):
    off_diagonal = [(cm[t, p], EMOTIONS[t], EMOTIONS[p]) for t in range(len(cm)) for p in range(len(cm)) if t != p]
    return [{"true": t, "predicted": p, "count": int(c)} for c, t, p in sorted(off_diagonal, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports"), type=Path)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_split(args.data_dir / "test", args.limit_per_class)
    model = keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    labels = list(range(len(EMOTIONS)))
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=EMOTIONS, digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    metrics = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {e: report[e] for e in EMOTIONS},
        "top_confusions": top_confusions(cm),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(args.out_dir / "confusion_matrix.png", dpi=150)

    print(classification_report(y_test, y_pred, labels=labels, target_names=EMOTIONS, digits=4, zero_division=0))
    print(f"accuracy {metrics['accuracy']:.4f}  macro F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
