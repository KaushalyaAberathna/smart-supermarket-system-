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


# PERSISTENCE (optional -- for debugging / report figures)

def save_crops(crops, output_dir=config.CROPS_DIR):
    """Write each crop to disk as output/crops/product_<id>.png.

    WHY optional and separate from crop(): the pipeline's primary data flow
    keeps crops in memory and passes them straight to classification.py.
    Saving to disk is only useful for visually inspecting what segmentation
    produced (e.g. while tuning MIN_CONTOUR_AREA) or for report screenshots,
    so it is not run by default -- callers opt in explicitly.
    """
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)  # clear stale crops from a previous run
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for c in crops:
        path = os.path.join(output_dir, f"product_{c['id']:02d}.png")
        cv2.imwrite(path, c["crop"])
        paths.append(path)

    print(f"[segmentation] Saved {len(paths)} crop(s) to: {output_dir}")
    return paths



# VISUALIZATION

def visualize_crops(crops, save_path=None, show=config.SHOW_PLOTS):
    """Display every segmented product candidate in a grid, labeled by id."""
    n = len(crops)
    if n == 0:
        print("[segmentation] No crops to display.")
        return

    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, crops):
        ax.imshow(cv2.cvtColor(c["crop"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Product {c['id']}", fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:  # hide unused grid cells
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[segmentation] Saved crops grid figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


