"""
Detects individual supermarket products using classical image processing
(contour analysis on the binary mask produced by preprocessing.py). 

WHY contours instead of a learned detector (e.g. YOLO)?
The project brief requires the *detection* stage to be classical image
processing (OpenCV), reserving the deep-learning model (MobileNetV2) for
*classification* of each already-detected region. This mirrors a simple,
fully local, explainable "segment first, then classify" pipeline rather than
an end-to-end trained detector.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from preprocessing import preprocess_image

class ProductDetector:
    """Finds candidate product regions in an image via contour analysis."""

    def __init__(self, min_contour_area=config.MIN_CONTOUR_AREA):
        self.min_contour_area = min_contour_area

            
    def find_contours(self, mask):
        """Extract external contours from a binary mask.

        cv2.RETR_EXTERNAL only keeps outermost contours (we don't care about
        holes inside a product's silhouette -- those were already patched by
        the morphological closing step in preprocessing). CHAIN_APPROX_SIMPLE
        compresses straight contour segments to their endpoints, which is
        enough for bounding-box computation and cheaper to store.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
        
    def filter_contours(self, contours):
        """Discard contours smaller than min_contour_area.

        WHY: leftover noise (label text fragments, mask speckle that survived
        morphology, shadows) produces many tiny contours that are not real
        products. Real supermarket products, even small ones, occupy a
        meaningfully larger area than noise at the working resolution set in
        config.PREPROCESS_RESIZE_WIDTH.
        """
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        
    def detect(self, image):
        """Run preprocessing + contour detection on a raw BGR image.

        Returns
        -------
        detections : list[dict]
            One entry per detected product, each with:
              - "id": 1-based product number (reading order: top-to-bottom,
                left-to-right, matching how a person would scan a basket photo)
              - "bbox": (x, y, w, h) in the ORIGINAL image's coordinate space
                (used for cropping full-resolution product images in
                segmentation.py)
              - "bbox_resized": (x, y, w, h) in the working (resized) image's
                coordinate space (matches the mask/contour directly)
              - "area_px": contour area, measured in the working resolution,
                so it is comparable to config.MIN_CONTOUR_AREA regardless of
                the original photo's resolution
              - "contour_resized": the raw contour points (working resolution)
        steps : dict
            The full preprocessing intermediate-results dict from
            preprocessing.py (useful for debugging/visualization).
        """
        steps = preprocess_image(image, display=False)
        mask = steps["final_mask"]
        resized = steps["resized"]

        contours = self.find_contours(mask)
        filtered = self.filter_contours(contours)

        # Sort in reading order: group into coarse rows (bucketed by y // 50
        # so boxes that are roughly on the same row don't get shuffled by
        # small y differences), then left-to-right within each row.
        def sort_key(c):
            x, y, _, _ = cv2.boundingRect(c)
            return (y // 50, x)

        filtered.sort(key=sort_key)

        # Scale factor from working (resized) resolution back to the
        # original photo's resolution, so cropped products for classification
        # (Module 4) are at full quality, not the downscaled working size.
        scale_x = image.shape[1] / float(resized.shape[1])
        scale_y = image.shape[0] / float(resized.shape[0])

        detections = []
        for idx, contour in enumerate(filtered, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            ox, oy = int(round(x * scale_x)), int(round(y * scale_y))
            ow, oh = int(round(w * scale_x)), int(round(h * scale_y))
            # Clip to image bounds in case rounding pushes the box outside.
            ox, oy = max(0, ox), max(0, oy)
            ow = min(ow, image.shape[1] - ox)
            oh = min(oh, image.shape[0] - oy)

            detections.append({
                "id": idx,
                "bbox": (ox, oy, ow, oh),
                "bbox_resized": (x, y, w, h),
                "area_px": area,
                "contour_resized": contour,
            })

        return detections, steps

    
