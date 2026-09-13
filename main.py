import argparse
import os

import config
from utils import load_image, list_images_in_directory
from detection import detect_products
from segmentation import segment_products
from classification import ProductClassifier
from category_mapping import map_products
from statistics import analyze_products
from visualization import visualize_final_output


def run_pipeline(image_path, classifier, output_dir=config.OUTPUT_DIR, display=config.SHOW_PLOTS):
    """Run the full detect -> segment -> classify -> map -> analyze ->
    visualize pipeline on a single image. Returns the computed stats dict.
    """
    print(f"\n{'#' * 60}")
    print(f"Processing: {image_path}")
    print(f"{'#' * 60}")

    image = load_image(image_path)

    detections, _, _ = detect_products(image, display=False)
    if not detections:
        print("[main] No products detected in this image -- nothing further to classify.")

    crops = segment_products(image, detections, display=False)

    classifier.predict_batch(crops)
    for c in crops:
        print(f"[main] Product {c['id']}: {c['label']} (confidence={c['confidence']:.2f})")

    map_products(crops)
    stats = analyze_products(crops)

    os.makedirs(output_dir, exist_ok=True)
    visualize_final_output(image, crops, stats, output_dir=output_dir, display=display)

    return stats
