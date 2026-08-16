"""
Crops every bounding box found by detection.py out of the original image.
Each crop becomes one "product candidate" -- a standalone image that
classification.py  will resize and feed into MobileNetV2.

"""

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config

class ProductSegmenter:
    """Crops detected bounding boxes out of an image into product candidates."""

    def crop(self, image, detections):
        #Crop each bounding box out of the original image.
        crops = []
        for det in detections:
            x, y, w, h = det["bbox"]
            if w <= 0 or h <= 0:
                # Can happen if rounding during detection's scale-back pushed
                # a box fully outside the image bounds; skip, it's not a
                # usable product candidate.
                print(f"[segmentation] Skipping product {det['id']}: degenerate bbox {det['bbox']}.")
                continue

            crop_img = image[y:y + h, x:x + w].copy()
            crops.append({
                "id": det["id"],
                "bbox": det["bbox"],
                "area_px": det["area_px"],
                "crop": crop_img,
            })

        return crops