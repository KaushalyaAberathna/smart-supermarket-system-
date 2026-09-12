import argparse
import os

import cv2

import config
from classification import ProductClassifier
from category_mapping import get_category


class ProductPredictor:
    """Loads the trained classifier once and predicts (label, confidence,
    category) for a single product image/crop. Never retrains -- if no
    trained model exists yet, it fails fast with a clear message pointing at
    train_model.py rather than silently doing something else.
    """

    def __init__(self, model_path=config.MODEL_PATH, class_index_path=config.CLASS_INDEX_PATH):
        self.classifier = ProductClassifier(model_path=model_path, class_index_path=class_index_path)

    def predict(self, crop_bgr):
        """Classify one BGR product image/crop.

        Returns
        -------
        dict with keys:
            "product"    -- predicted class name, or "Unknown" if the top
                             prediction's confidence is below
                             config.CLASSIFICATION_CONFIDENCE_THRESHOLD
            "confidence" -- float in [0, 1]
            "category"   -- supermarket category (category_mapping.py),
                             "Unknown" if the product label itself is "Unknown"
        """
        label, confidence = self.classifier.predict(crop_bgr)
        return {
            "product": label,
            "confidence": confidence,
            "category": get_category(label),
        }

    def predict_path(self, image_path):
        """Convenience wrapper: read an image file from disk, then predict().
        Intended for a single already-cropped product photo, not a
        multi-product basket photo (see main.py for the full detect ->
        segment -> classify pipeline on those).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image at '{image_path}'.")
        return self.predict(image)
