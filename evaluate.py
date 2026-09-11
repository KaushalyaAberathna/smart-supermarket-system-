import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

import config
from classification import load_class_names
from train_model import load_split_csv, make_dataset


TARGET_ACCURACY = 0.80  # project requirement: >= 80% classification accuracy

# Sequential single-hue ramp (light -> dark blue), consistent with the
# categorical palette's blue slot used elsewhere in the project -- magnitude
# encodings use one hue, never a rainbow colormap.
_SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def load_test_predictions():
    """Run the trained model over the held-out test split.

    Returns
    -------
    y_true, y_pred : np.ndarray of integer class indices
    class_names : list[str], in the same index order the model outputs
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{config.MODEL_PATH}'. Run train_model.py first."
        )

    test_csv_path = os.path.join(config.SPLITS_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            f"No test split found at '{test_csv_path}'. Run train_model.py first "
            "(it writes dataset/splits/{train,val,test}.csv)."
        )

    class_names = load_class_names()
    class_to_index = {name: i for i, name in enumerate(class_names)}

    test_pairs = load_split_csv(test_csv_path)
    # y_true is derived directly from the CSV's label column, in file order --
    # safe because make_dataset() below is called with shuffle=False, so
    # predictions come back in that same order.
    y_true = np.array([class_to_index[label] for _, label in test_pairs])

    test_ds = make_dataset(
        test_pairs, class_to_index, config.CLASSIFIER_BATCH_SIZE, config.CLASSIFIER_IMG_SIZE,
        shuffle=False, augment=False,
    )

    model = tf.keras.models.load_model(config.MODEL_PATH)
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    return y_true, y_pred, class_names


def compute_metrics(y_true, y_pred, class_names):
    """Compute accuracy, macro/weighted precision-recall-F1, the full
    per-class classification report, and the confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "classification_report": report_text,
        "confusion_matrix": cm,
    }


def print_metrics_summary(metrics):
    print("=" * 40)
    print("CLASSIFICATION EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Accuracy            : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision     : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall        : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-score      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted Precision  : {metrics['weighted_precision'] * 100:.2f}%")
    print(f"Weighted Recall     : {metrics['weighted_recall'] * 100:.2f}%")
    print(f"Weighted F1-score   : {metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 40)

    if metrics["accuracy"] >= TARGET_ACCURACY:
        print(f"Target accuracy ({TARGET_ACCURACY * 100:.0f}%) : MET "
              f"({metrics['accuracy'] * 100:.2f}% >= {TARGET_ACCURACY * 100:.0f}%)")
    else:
        print(f"Target accuracy ({TARGET_ACCURACY * 100:.0f}%) : NOT MET "
              f"({metrics['accuracy'] * 100:.2f}% < {TARGET_ACCURACY * 100:.0f}%)")
    print("=" * 40)

    print("\nPer-class report:")
    print(metrics["classification_report"])


def plot_confusion_matrix(cm, class_names, save_path=None, show=config.SHOW_PLOTS, normalize=True):
    """Plot the confusion matrix as a heatmap.

    Row-normalized by default (each row sums to 1 = that true class's
    predictions, so the diagonal reads directly as per-class recall). With
    25 classes, per-cell numeric annotations would be unreadable clutter, so
    magnitude is conveyed by color alone here; exact counts are saved
    separately as a CSV for anyone who needs precise numbers (see
    save_confusion_matrix_csv).
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )
        title = "Confusion Matrix (row-normalized -- diagonal = per-class recall)"
        vmax = 1.0
    else:
        cm_display = cm
        title = "Confusion Matrix (counts)"
        vmax = cm.max()

    cmap = LinearSegmentedColormap.from_list("sequential_blue", _SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label", color="#52514e")
    ax.set_ylabel("True label", color="#52514e")
    ax.set_title(title, fontsize=12, color="#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved confusion matrix heatmap to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_confusion_matrix_csv(cm, class_names, save_path):
    """Save the raw-count confusion matrix as a CSV (class_names header row
    + row labels), for exact numbers the heatmap intentionally omits.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        f.write("true_label\\predicted_label," + ",".join(class_names) + "\n")
        for name, row in zip(class_names, cm):
            f.write(name + "," + ",".join(str(v) for v in row) + "\n")
    print(f"[evaluate] Saved confusion matrix counts to: {save_path}")




def save_metrics_json(metrics, save_path):
    """Save the scalar metrics (not the confusion matrix / text report) as
    JSON, so main.py or the report generation can consume them programmatically.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scalar_metrics = {k: v for k, v in metrics.items() if k not in ("classification_report", "confusion_matrix")}
    scalar_metrics["target_accuracy"] = TARGET_ACCURACY
    scalar_metrics["target_met"] = metrics["accuracy"] >= TARGET_ACCURACY
    with open(save_path, "w") as f:
        json.dump(scalar_metrics, f, indent=2)
    print(f"[evaluate] Saved evaluation metrics to: {save_path}")
