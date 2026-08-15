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

    
