"""
Defines the MobileNetV2-based model architecture (shared with train_model.py,
so training and inference never drift out of sync) and the inference-time
ProductClassifier that turns a segmented product crop into a predicted class
label.

--------------------------------------------------------------------------
WHY MobileNetV2 was selected
--------------------------------------------------------------------------
- It is built from depthwise-separable convolutions, giving a strong
  accuracy-per-FLOP tradeoff designed explicitly for mobile/edge inference.
  That matches this project's "runs entirely locally, no cloud services"
  requirement -- the whole pipeline, including classification, needs to run
  comfortably on a laptop CPU, not a datacenter GPU.
- ImageNet-pretrained weights give MobileNetV2 transferable low/mid-level
  visual features (edges, textures, colour blobs, shapes) learned from 1.4M
  general images. Freiburg Groceries has only ~4,800 images across 25
  classes -- far too few to train a strong CNN from scratch -- so transfer
  learning is what makes reaching a good accuracy feasible at this dataset
  size.
- It ships as a small (~14 MB) set of weights, keeping the project easy to
  package, version, and run without special hardware.
"""

import os
import json

import cv2
import numpy as np
import tensorflow as tf

import config


# --------------------------------------------------------------------------
# MODEL ARCHITECTURE (shared by classification.py and train_model.py)
# --------------------------------------------------------------------------
def build_model(num_classes=config.NUM_CLASSES, input_shape=config.CLASSIFIER_IMG_SIZE + (3,)):
    """Build the MobileNetV2 transfer-learning model.

    Returns
    -------
    model : tf.keras.Model
        Full model: MobileNetV2 base + custom classification head.
    base_model : tf.keras.Model
        Reference to just the MobileNetV2 base, so train_model.py can toggle
        `.trainable` and select which layers to unfreeze for Phase 2
        fine-tuning without rebuilding the whole model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,      # drop ImageNet's original 1000-class head
        weights="imagenet",     # start from pretrained ImageNet features
    )
    base_model.trainable = False  # Phase 1 default: base frozen

    inputs = tf.keras.Input(shape=input_shape)
    # preprocess_input rescales pixels to the [-1, 1] range MobileNetV2 was
    # trained on; baking it into the graph means callers only need to pass
    # standard 0-255 RGB images, in both training and inference.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # BatchNorm re-centers/re-scales the frozen base's pooled features before
    # they hit the trainable head -- stabilizes Phase 1 training since the
    # base's own BatchNorm layers stay frozen (training=False above) and so
    # never adapt their statistics to grocery images themselves.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # reduces overfitting given the modest dataset size
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(3e-4),
    )(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_freiburg_groceries")
    return model, base_model

def preprocess_crop(crop_bgr, target_size=config.CLASSIFIER_IMG_SIZE):
    
    resized = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)

def load_class_names(class_index_path=config.CLASS_INDEX_PATH):
    """Rebuild the ordered class-name list a trained model's output indices
    correspond to. train_model.py saves this mapping after training so
    inference/evaluation is never at risk of silently mis-mapping
    predictions if config.CLASS_NAMES were ever edited after training.
    Shared by ProductClassifier (inference) and evaluate.py (test-set scoring).
    """
    if os.path.exists(class_index_path):
        with open(class_index_path, "r") as f:
            index_to_class = json.load(f)
        return [index_to_class[str(i)] for i in range(len(index_to_class))]
    # Fallback (no saved mapping yet): config.CLASS_NAMES is already
    # alphabetically sorted, matching Keras' directory-based label order.
    return config.CLASS_NAMES


class ProductClassifier:
    """Loads a trained model once and classifies product crops with it."""

    def __init__(
        self,
        model_path=config.MODEL_PATH,
        class_index_path=config.CLASS_INDEX_PATH,
        confidence_threshold=config.CLASSIFICATION_CONFIDENCE_THRESHOLD,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'.\n"
                "Run train_model.py first to fine-tune MobileNetV2 on the "
                "Freiburg Groceries dataset -- classification.py only "
                "performs inference with an already-trained model."
            )
        self.model = tf.keras.models.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = load_class_names(class_index_path)


        def predict(self, crop_bgr):
                """Classify one crop. Returns (label, confidence).
        
                If the top prediction's confidence is below
                config.CLASSIFICATION_CONFIDENCE_THRESHOLD, the label is "Unknown"
                rather than a low-confidence guess -- better to flag uncertainty
                than to mislabel a product in a checkout-style report.
                """
                x = preprocess_crop(crop_bgr)
                x = np.expand_dims(x, axis=0)
                probs = self.model.predict(x, verbose=0)[0]
        
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                label = self.class_names[idx] if confidence >= self.confidence_threshold else "Unknown"
                return label, confidence