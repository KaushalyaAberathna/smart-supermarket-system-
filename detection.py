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