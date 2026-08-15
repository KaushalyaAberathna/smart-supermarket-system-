import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config


class ImagePreprocessor:

    

    def __init__(
        self,
        resize_width=config.PREPROCESS_RESIZE_WIDTH,
        gaussian_kernel=config.GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma=config.GAUSSIAN_SIGMA,
        threshold_method=config.THRESHOLD_METHOD,
        adaptive_block_size=config.ADAPTIVE_THRESH_BLOCK_SIZE,
        adaptive_c=config.ADAPTIVE_THRESH_C,
        binary_thresh_value=config.BINARY_THRESH_VALUE,
        invert=config.THRESHOLD_INVERT,
        canny_low=config.CANNY_LOW_THRESHOLD,
        canny_high=config.CANNY_HIGH_THRESHOLD,
        morph_kernel_size=config.MORPH_KERNEL_SIZE,
        morph_iterations=config.MORPH_ITERATIONS,
    ):
        self.resize_width = resize_width
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.threshold_method = threshold_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.binary_thresh_value = binary_thresh_value
        self.invert = invert
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel = np.ones(morph_kernel_size, dtype=np.uint8)
        self.morph_iterations = morph_iterations


    def resize(self, image):
        
        h, w = image.shape[:2]
        if w == self.resize_width:
            return image.copy()
        scale = self.resize_width / float(w)
        new_size = (self.resize_width, int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)