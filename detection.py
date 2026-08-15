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

        # ------------------------------------------------------------------
    def draw_detections(self, image, detections):
        """Draw a bounding box and a "Product N" label for each detection."""
        annotated = image.copy()
        font = getattr(cv2, config.LABEL_FONT)

        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h),
                config.BOUNDING_BOX_COLOR, config.BOUNDING_BOX_THICKNESS,
            )

            label = f"Product {det['id']}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS
            )
            # Filled label background above the box for readability; clipped
            # to y=0 so labels on products near the top edge stay on-screen.
            label_top = max(0, y - text_h - baseline - 4)
            cv2.rectangle(
                annotated, (x, label_top), (x + text_w + 4, y),
                config.BOUNDING_BOX_COLOR, -1,
            )
            cv2.putText(
                annotated, label, (x + 2, y - 4), font,
                config.LABEL_FONT_SCALE, (0, 0, 0), config.LABEL_THICKNESS, cv2.LINE_AA,
            )

        return annotated


# CONVENIENCE FUNCTION

def detect_products(image, display=False, save_path=None):
    """Detect products in `image`, draw annotated bounding boxes, and
    optionally display/save the result. Returns (detections, annotated_image, steps).
    """
    detector = ProductDetector()
    detections, steps = detector.detect(image)
    annotated = detector.draw_detections(image, detections)

    print(f"[detection] Found {len(detections)} product candidate(s) "
          f"(min_contour_area={config.MIN_CONTOUR_AREA}px^2 at working resolution).")

    if display or save_path:
        _visualize(annotated, save_path=save_path, show=display)

    return detections, annotated, steps


def _visualize(annotated_image, save_path=None, show=True):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Products")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[detection] Saved annotated detection image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


        
# DEMO / SELF-TEST

if __name__ == "__main__":
    # Same fallback strategy as preprocessing.py: prefer a real basket/table
    # photo from images/, else fall back to a single Freiburg sample. Note
    # that a single close-up product photo is NOT representative of this
    # module's intended input (multiple separated products on a surface) --
    # it mainly proves the code path runs. Test with a real multi-product
    # photo in images/ once available.
    demo_image_path = None
    if os.path.isdir(config.TEST_IMAGES_DIR):
        candidates = [
            f for f in os.listdir(config.TEST_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if candidates:
            demo_image_path = os.path.join(config.TEST_IMAGES_DIR, candidates[0])

    if demo_image_path is None:
        fallback_class = config.CLASS_NAMES[0]  # "BEANS"
        fallback_dir = os.path.join(config.DATASET_DIR, fallback_class)
        fallback_file = sorted(os.listdir(fallback_dir))[0]
        demo_image_path = os.path.join(fallback_dir, fallback_file)
        print(
            "[detection] No image found in images/. Using a Freiburg dataset "
            f"sample instead for this demo: {demo_image_path}\n"
            "[detection] Add a real multi-product basket/table photo to "
            "images/ to test detection on its intended input."
        )

    demo_image = cv2.imread(demo_image_path)
    if demo_image is None:
        raise FileNotFoundError(f"Could not read demo image: {demo_image_path}")

    output_path = os.path.join(config.OUTPUT_DIR, "detected_products_demo.png")
    detect_products(demo_image, display=config.SHOW_PLOTS, save_path=output_path)




    
