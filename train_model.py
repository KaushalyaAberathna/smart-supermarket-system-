"""
train_model.py

MODULE 9 (pipeline stage 9 of 10, training-side) -- TRAINING MobileNetV2 ON
THE FREIBURG GROCERIES DATASET

Builds a stratified train/val/test split of dataset/freiburg_groceries/,
trains the model architecture defined in classification.py using the
two-phase strategy agreed on:

    Phase 1: MobileNetV2 base frozen, train only the new classification
             head, for config.CLASSIFIER_EPOCHS_HEAD epochs.
    Phase 2: unfreeze the top base layers (from config.FINE_TUNE_AT_LAYER
             onward) and fine-tune end-to-end at a much lower learning
             rate, for config.CLASSIFIER_EPOCHS_FINE_TUNE more epochs.

Saves the trained model to config.MODEL_PATH and the label-index mapping to
config.CLASS_INDEX_PATH, which classification.py's ProductClassifier loads
at inference time. Also saves a training-curves figure so accuracy/loss
over both phases can be inspected (and dropped straight into the project
report's "Experimental Results" section).
"""

import os
import csv
import json
import random
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import config
from classification import build_model


# --------------------------------------------------------------------------
# DATASET SPLITTING
# --------------------------------------------------------------------------
def build_dataset_split():
    """Create a stratified train/val/test split of dataset/freiburg_groceries/.

    WHY stratified per-class rather than one global shuffle-then-cut: class
    sizes range from 97 (CORN) to 372 (CANDY) images. A single global split
    could easily under-represent a small class in the test set purely by
    chance; splitting inside each class folder guarantees every class
    contributes ~70/15/15 to train/val/test regardless of its size.

    The split is deterministic (seeded by config.RANDOM_SEED), so re-running
    this function reproduces the exact same split -- and evaluate.py reads
    the persisted CSVs directly rather than recomputing, so the test set it
    reports on is guaranteed identical to what train_model.py held out.

    If config.OVERSAMPLE_MINORITY_CLASSES is set, the TRAIN split (only) is
    then topped up: classes below the mean per-class train count get extra
    (path, class_name) rows duplicated back in via sampling-with-replacement,
    up to the mean. Combined with per-epoch random augmentation, a repeated
    path is not a bit-for-bit duplicate sample in training -- it just gives
    small classes (CORN, FLOUR, FISH, ...) proportionally more exposure per
    epoch, without the loss-scale side effects a "balanced" class_weight had
    (see config.USE_CLASS_WEIGHT's comment). val/test are left untouched, so
    evaluation always reflects the dataset's real, unaltered distribution.

    Returns
    -------
    dict with keys "train", "val", "test", each a list of
    (relative_path, class_name) tuples.
    """
    rng = random.Random(config.RANDOM_SEED)
    splits = {"train": [], "val": [], "test": []}

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * config.TRAIN_SPLIT))
        n_val = int(round(n * config.VAL_SPLIT))
        # test gets the remainder, so every file is used exactly once even
        # if rounding doesn't land the three counts on an exact split.
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        splits["train"] += [(f"{class_name}/{f}", class_name) for f in train_files]
        splits["val"] += [(f"{class_name}/{f}", class_name) for f in val_files]
        splits["test"] += [(f"{class_name}/{f}", class_name) for f in test_files]

    if config.OVERSAMPLE_MINORITY_CLASSES:
        splits["train"] = _oversample_minority_classes(splits["train"], rng)

    # Shuffle the overall (cross-class) order too, so batches mix classes
    # instead of running through one class at a time.
    for key in splits:
        rng.shuffle(splits[key])

    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    _write_split_csv(splits["train"], os.path.join(config.SPLITS_DIR, "train.csv"))
    _write_split_csv(splits["val"], os.path.join(config.SPLITS_DIR, "val.csv"))
    _write_split_csv(splits["test"], os.path.join(config.SPLITS_DIR, "test.csv"))

    print(
        f"[train_model] Split sizes -- train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])} "
        f"(total {sum(len(v) for v in splits.values())})"
    )
    return splits