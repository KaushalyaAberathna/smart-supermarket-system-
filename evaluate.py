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