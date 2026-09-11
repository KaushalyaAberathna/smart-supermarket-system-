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

def _oversample_minority_classes(train_pairs, rng):
    """Top up classes below the mean train-set size up to that mean, by
    duplicating (path, class_name) rows sampled with replacement. See
    build_dataset_split's docstring for why this is preferred over
    class_weight-based loss reweighting.
    """
    by_class = {}
    for pair in train_pairs:
        by_class.setdefault(pair[1], []).append(pair)

    mean_count = sum(len(v) for v in by_class.values()) / len(by_class)
    target = int(round(mean_count))

    topped_up = list(train_pairs)
    for pairs in by_class.values():
        deficit = target - len(pairs)
        if deficit > 0:
            topped_up += [rng.choice(pairs) for _ in range(deficit)]

    return topped_up


def _write_split_csv(pairs, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(pairs)


def load_split_csv(path):
    """Read a split CSV written by build_dataset_split(). Used by
    evaluate.py to load the exact same test split used here.
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [(row[0], row[1]) for row in reader]


# tf.data PIPELINE

def _build_augmenter():
    """Augmentation for the training split only. WHY: several Freiburg
    classes have well under 200 images (e.g. CORN=97, FLOUR=109) -- flip/
    rotation/zoom/contrast/translation cheaply synthesize more visual variety
    and reduce overfitting on those smaller classes. Kept deliberately mild
    (small factors) since product packaging orientation/text is still
    meaningful (an upside-down box is unusual), not the aggressive
    augmentation used for e.g. natural scene photos. Contrast/translation
    were added alongside the original flip/rotation/zoom to also cover the
    lighting and framing variation Freiburg's shelf photos show across shots.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def make_dataset(
    pairs, class_to_index, batch_size, img_size,
    shuffle=False, augment=False, one_hot=False, num_classes=None,
):
    """Build a tf.data.Dataset of (image, label) batches from a list of
    (relative_path, class_name) pairs.

    Pixel-range normalization ([-1, 1] for MobileNetV2) is NOT done here --
    it's baked into the model graph itself (see classification.build_model),
    so this pipeline only decodes, resizes, and (for training) augments,
    keeping raw 0-255 float images flowing out.

    one_hot=True yields (image, one-hot-vector) instead of (image, int label)
    -- used only for training/validation, so train_model.py can pair it with
    CategoricalCrossentropy(label_smoothing=...); evaluate.py keeps the
    default sparse-int labels since it only needs argmax comparisons, not a
    loss.
    """
    paths = [os.path.join(config.DATASET_DIR, p) for p, _ in pairs]
    labels = [class_to_index[lbl] for _, lbl in pairs]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        if one_hot:
            label = tf.one_hot(label, num_classes)
        return image, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = _build_augmenter()
        ds = ds.map(lambda x, l: (augmenter(x, training=True), l), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    if augment and one_hot and config.USE_MIXUP:
        ds = ds.map(lambda x, l: _mixup_batch(x, l, config.MIXUP_ALPHA), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _mixup_batch(images, one_hot_labels, alpha):

    batch_size = tf.shape(images)[0]
    shuffled_idx = tf.random.shuffle(tf.range(batch_size))

    gamma1 = tf.random.gamma([batch_size], alpha, 1.0)
    gamma2 = tf.random.gamma([batch_size], alpha, 1.0)
    lam = gamma1 / (gamma1 + gamma2)

    lam_image = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_label = tf.reshape(lam, [batch_size, 1])

    mixed_images = lam_image * images + (1.0 - lam_image) * tf.gather(images, shuffled_idx)
    mixed_labels = lam_label * one_hot_labels + (1.0 - lam_label) * tf.gather(one_hot_labels, shuffled_idx)
    return mixed_images, mixed_labels




# TRAINING

def train(
    epochs_head=config.CLASSIFIER_EPOCHS_HEAD,
    epochs_fine_tune=config.CLASSIFIER_EPOCHS_FINE_TUNE,
    steps_per_epoch=None,
    validation_steps=None,
):
    """Run the full two-phase training procedure and save the trained model.

    epochs_head/epochs_fine_tune/steps_per_epoch/validation_steps default to
    config values (the real training run) but can be overridden -- mainly so
    this function can be smoke-tested with a couple of steps/epochs before
    committing to a full ~20-40 minute CPU run.
    """
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    splits = build_dataset_split()
    class_to_index = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    # Class sizes range from 97 (CORN) to 372 (CANDY) images. "balanced"
    # class weighting (reweighting each class inversely to its training
    # frequency) was tried to help small/confused classes like RICE and
    # VINEGAR, but in practice over-corrected and hurt the largest class
    # (CANDY recall dropped from 0.79 to 0.45), reducing overall accuracy.
    # Kept as an option (config.USE_CLASS_WEIGHT) but off by default.
    class_weight = None
    if config.USE_CLASS_WEIGHT:
        train_label_indices = [class_to_index[lbl] for _, lbl in splits["train"]]
        class_weight_values = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(config.NUM_CLASSES),
            y=train_label_indices,
        )
        class_weight = {i: w for i, w in enumerate(class_weight_values)}

    train_ds = make_dataset(
        splits["train"], class_to_index, config.CLASSIFIER_BATCH_SIZE, config.CLASSIFIER_IMG_SIZE,
        shuffle=True, augment=config.USE_DATA_AUGMENTATION,
        one_hot=True, num_classes=config.NUM_CLASSES,
    )
    val_ds = make_dataset(
        splits["val"], class_to_index, config.CLASSIFIER_BATCH_SIZE, config.CLASSIFIER_IMG_SIZE,
        shuffle=False, augment=False,
        one_hot=True, num_classes=config.NUM_CLASSES,
    )

    model, base_model = build_model()
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Label smoothing needs one-hot targets, hence CategoricalCrossentropy
    # (not the sparse variant) paired with the one_hot=True datasets above.
    # Softening the targets (e.g. true class -> 0.9 instead of 1.0) discourages
    # the model from becoming over-confident on a modest-sized, sometimes
    # visually ambiguous dataset (bottles/bags that look alike across classes).
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=config.CLASSIFIER_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    start_time = time.time()

    # ---------------- Phase 1: frozen base, train the new head ----------------
    print("\n=== Phase 1: training classification head (MobileNetV2 base frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.CLASSIFIER_LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_head,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=callbacks, shuffle=False,  # train_ds is already shuffled by make_dataset()
        class_weight=class_weight,
    )

    # ---------------- Phase 2: unfreeze top layers, fine-tune ----------------
    history2 = None
    if epochs_fine_tune > 0:
        print(f"\n=== Phase 2: fine-tuning MobileNetV2 layers from index {config.FINE_TUNE_AT_LAYER} onward ===")
        base_model.trainable = True
        for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
            layer.trainable = False  # keep early (generic) feature layers frozen

        # Recompile is required after changing .trainable flags, and starts
        # from a low learning rate -- large updates to pretrained weights at
        # this stage would destroy the useful ImageNet features instead of
        # gently adapting them to grocery packaging. ReduceLROnPlateau then
        # halves it further whenever val_accuracy stalls, letting the model
        # keep making small gains instead of oscillating around a plateau.
        #
        # AdamW (not Adam) here specifically: the Dense head's L2/Dropout
        # only regularize ~32k head parameters -- once this phase unfreezes
        # dozens of MobileNetV2 layers, THOSE (far more numerous) backbone
        # weights had no regularization at all under plain Adam, which is
        # what let an earlier run's train accuracy hit 98%+ while test
        # accuracy stayed at ~77%. AdamW's decoupled weight decay applies to
        # every trainable weight in this phase, backbone included.
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=config.FINE_TUNE_LEARNING_RATE,
                weight_decay=config.FINE_TUNE_WEIGHT_DECAY,
            ),
            loss=loss_fn,
            metrics=["accuracy"],
        )
        fine_tune_callbacks = callbacks + [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", mode="max",
                factor=0.5, patience=2, min_lr=1e-7, verbose=1,
            ),
        ]

        initial_epoch = history1.epoch[-1] + 1
        total_epochs = initial_epoch + epochs_fine_tune
        history2 = model.fit(
            train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            callbacks=fine_tune_callbacks, shuffle=False,  # train_ds is already shuffled by make_dataset()
            class_weight=class_weight,
        )
    else:
        print("\n[train_model] epochs_fine_tune=0 -- skipping Phase 2.")

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n[train_model] Training finished in {elapsed_minutes:.1f} minutes.")

    # EarlyStopping's restore_best_weights leaves the in-memory model at its
    # best val_accuracy checkpoint, which may differ from whatever
    # ModelCheckpoint last wrote to disk mid-training -- save explicitly here
    # so the persisted file always matches the final in-memory weights.
    model.save(config.MODEL_PATH)
    print(f"[train_model] Saved final model to: {config.MODEL_PATH}")

    index_to_class = {str(i): name for i, name in enumerate(config.CLASS_NAMES)}
    os.makedirs(os.path.dirname(config.CLASS_INDEX_PATH), exist_ok=True)
    with open(config.CLASS_INDEX_PATH, "w") as f:
        json.dump(index_to_class, f, indent=2)
    print(f"[train_model] Saved class index mapping to: {config.CLASS_INDEX_PATH}")

    plot_training_history(
        history1, history2,
        save_path=os.path.join(config.OUTPUT_DIR, "training_history.png"),
    )

    return model, history1, history2



# TRAINING CURVES

def plot_training_history(history1, history2=None, save_path=None, show=config.SHOW_PLOTS):
    """Plot accuracy and loss (train vs. val) across both training phases,
    with a marker at the Phase 1 -> Phase 2 boundary.
    """
    acc = list(history1.history["accuracy"])
    val_acc = list(history1.history["val_accuracy"])
    loss = list(history1.history["loss"])
    val_loss = list(history1.history["val_loss"])
    phase_boundary = len(acc) - 0.5

    if history2 is not None:
        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs_range, acc, label="Train", color="#2a78d6", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Validation", color="#eb6834", linewidth=2)
    if history2 is not None:
        axes[0].axvline(phase_boundary, color="#898781", linestyle="--", linewidth=1, label="Fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(epochs_range, loss, label="Train", color="#2a78d6", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Validation", color="#eb6834", linewidth=2)
    if history2 is not None:
        axes[1].axvline(phase_boundary, color="#898781", linestyle="--", linewidth=1, label="Fine-tuning starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train_model] Saved training curves to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ENTRY POINT
if __name__ == "__main__":
    train()