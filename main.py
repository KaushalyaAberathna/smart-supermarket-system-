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

def main():
    parser = argparse.ArgumentParser(description="Smart Supermarket Product Identification System")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to process. If omitted, every image in images/ is processed.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Save outputs to output/ without popping up matplotlib windows.",
    )
    args = parser.parse_args()
    display = config.SHOW_PLOTS and not args.no_display

    try:
        classifier = ProductClassifier()
    except FileNotFoundError as e:
        print(f"[main] {e}")
        return

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = (
            list_images_in_directory(config.TEST_IMAGES_DIR)
            if os.path.isdir(config.TEST_IMAGES_DIR) else []
        )
        if not image_paths:
            print(
                f"[main] No images found in {config.TEST_IMAGES_DIR}. "
                "Add basket/product-layout photos there, or run with --image <path>."
            )
            return

    # Single image -> write straight to output/. Multiple images -> one
    # subfolder per image, so a batch run doesn't overwrite each image's
    # final_annotated.png/bar_chart.png/pie_chart.png with the next one's.
    multiple = len(image_paths) > 1

    for image_path in image_paths:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        per_image_output_dir = (
            os.path.join(config.OUTPUT_DIR, image_stem) if multiple else config.OUTPUT_DIR
        )
        try:
            run_pipeline(image_path, classifier, output_dir=per_image_output_dir, display=display)
        except (FileNotFoundError, ValueError) as e:
            print(f"[main] Skipping '{image_path}': {e}")


if __name__ == "__main__":
    main()
