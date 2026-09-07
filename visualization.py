import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import category_mapping


# --------------------------------------------------------------------------
# CATEGORY COLOR PALETTE
# --------------------------------------------------------------------------
# Fixed hue per category (validated categorical palette, assigned in a
# stable order -- see category_mapping.CATEGORY_NAMES, which is alphabetical
# and has exactly 6 entries). Hex strings are used for matplotlib; BGR
# tuples (derived below) are used for OpenCV drawing.
CATEGORY_COLORS_HEX = {
    "Bakery & Grains":     "#2a78d6",  # blue
    "Beverages":           "#008300",  # green
    "Dairy":               "#e87ba4",  # magenta
    "Pantry & Condiments": "#eda100",  # yellow
    "Protein":             "#1baf7a",  # aqua
    "Snacks & Sweets":     "#eb6834",  # orange
}
UNKNOWN_COLOR_HEX = "#898781"  # neutral gray -- not a real category, so no categorical hue


def _hex_to_bgr(hex_color):
    """Convert '#rrggbb' to an OpenCV-style (B, G, R) integer tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
