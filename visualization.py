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

def _category_color_hex(category):
    return CATEGORY_COLORS_HEX.get(category, UNKNOWN_COLOR_HEX)

def _readable_text_color(bgr_color):
    """Pick black or white text for readability against a given BGR
    background, using standard relative-luminance weighting.
    """
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


# --------------------------------------------------------------------------
# FINAL ANNOTATED IMAGE
# --------------------------------------------------------------------------
def draw_final_annotations(image, crops):
    """Draw a bounding box plus a two-line label chip (product label on top,
    category below) for every classified+mapped crop, e.g.:

        +----------------------+
        | MILK                 |
        | Dairy                |
        +----------------------+

    The chip is filled with the product's category color, so categories are
    visually distinguishable at a glance without reading text.
    """
    annotated = image.copy()
    font = getattr(cv2, config.LABEL_FONT)

    for c in crops:
        x, y, w, h = c["bbox"]
        category = c.get("category", category_mapping.UNKNOWN_CATEGORY)
        color_bgr = _hex_to_bgr(_category_color_hex(category))
        text_color = _readable_text_color(color_bgr)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, config.BOUNDING_BOX_THICKNESS)

        line1 = str(c.get("label", "Unknown"))
        line2 = str(category)
        (w1, h1), base1 = cv2.getTextSize(line1, font, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)
        (w2, h2), base2 = cv2.getTextSize(line2, font, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)

        chip_w = max(w1, w2) + 10
        chip_h = h1 + h2 + base1 + base2 + 10
        chip_top = max(0, y - chip_h)

        cv2.rectangle(annotated, (x, chip_top), (x + chip_w, y), color_bgr, -1)
        cv2.putText(
            annotated, line1, (x + 5, chip_top + h1 + 4), font,
            config.LABEL_FONT_SCALE, text_color, config.LABEL_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, line2, (x + 5, chip_top + h1 + h2 + base1 + 6), font,
            config.LABEL_FONT_SCALE, text_color, config.LABEL_THICKNESS, cv2.LINE_AA,
        )

    return annotated


def show_final_image(annotated_image, save_path=None, show=config.SHOW_PLOTS):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Smart Checkout -- Final Detection Result")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved final annotated image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------------
# CONSOLE REPORT
# ----------------------------------------------------------------------------
def print_console_report(stats):
    """Print the "SMART CHECKOUT REPORT" console summary.

    All 6 real categories are always listed (even at 0), matching
    statistics.py's zero-filled counts, so the report is consistent across
    runs. "Unknown" (low-confidence classifications) is only shown if it
    actually occurred this run -- it isn't a real product category.
    """
    total = stats["total_products"]
    counts = stats["category_counts"]
    percentages = stats["category_percentages"]

    categories = list(category_mapping.CATEGORY_NAMES)
    if counts.get(category_mapping.UNKNOWN_CATEGORY, 0) > 0:
        categories.append(category_mapping.UNKNOWN_CATEGORY)

    name_width = max(len(c) for c in categories) + 2

    print("=" * 28)
    print("SMART CHECKOUT REPORT")
    print("=" * 28)
    print(f"Total Products : {total}\n")

    for cat in categories:
        print(f"{cat:<{name_width}}: {counts.get(cat, 0)}")

    print("\nDistribution")
    for cat in categories:
        pct = percentages.get(cat, 0.0)
        print(f"{cat:<{name_width}}{round(pct)}%")

    print("=" * 28)
#end of file
