import io
import contextlib
import sys

import cv2
import gradio as gr
import pandas as pd

import config
import category_mapping
from preprocessing import preprocess_image
from detection import detect_products
from segmentation import segment_products
from classification import classify_products_with
from category_mapping import map_products
from statistics import analyze_products
from visualization import draw_final_annotations, plot_bar_chart, plot_pie_chart, print_console_report
from predict import get_predictor


# --------------------------------------------------------------------------
# LOAD THE TRAINED MODEL ONCE, AT STARTUP
# --------------------------------------------------------------------------
try:
    _predictor = get_predictor()
except FileNotFoundError as e:
    print(f"[gradio_app] {e}")
    print("[gradio_app] Run train_model.py first -- gradio_app.py only performs inference.")
    sys.exit(1)

print(f"[gradio_app] Loaded trained model from {config.MODEL_PATH} (once, at startup).")


def _load_model_accuracy():
    """Read the test-set accuracy evaluate.py already computed and saved, so
    the UI can show it without gradio_app.py re-evaluating the model itself
    (evaluation happens once, offline, via evaluate.py -- see README).
    Returns a human-readable string; falls back gracefully if evaluate.py
    hasn't been run yet.
    """
    import os
    import json

    metrics_path = os.path.join(config.OUTPUT_DIR, "evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        return "Not yet evaluated -- run `python evaluate.py` to compute test accuracy."

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    status = "MET" if metrics.get("target_met") else "NOT MET"
    return (
        f"**Test accuracy: {metrics['accuracy'] * 100:.2f}%** "
        f"(target {metrics['target_accuracy'] * 100:.0f}% -- {status}) &nbsp;|&nbsp; "
        f"Macro F1: {metrics['macro_f1'] * 100:.2f}% &nbsp;|&nbsp; "
        f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%"
    )


_MODEL_ACCURACY_TEXT = _load_model_accuracy()



# FULL PIPELINE, ADAPTED FOR GRADIO'S IN-MEMORY IMAGES

def run_full_pipeline(image_rgb):

    if image_rgb is None:
        empty_df = pd.DataFrame(columns=["Product #", "Product", "Confidence (%)", "Category"])
        empty_cat_df = pd.DataFrame(columns=["Category", "Count", "Percentage (%)"])
        return (
            None, None, None, None,
            empty_df, empty_cat_df, 0,
            "Upload a basket/table photo to run the pipeline.",
            None, None,
        )

    # Gradio gives RGB; every other module in this project (OpenCV-based)
    # works in BGR, so convert once at the boundary.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # ---- Preprocessing + Detection (Modules 2-3) ----
    detections, detected_annotated, steps = detect_products(image_bgr, display=False)
    preprocessed_mask = steps["final_mask"]  # the binary mask detection.py actually finds contours on

    if not detections:
        empty_df = pd.DataFrame(columns=["Product #", "Product", "Confidence (%)", "Category"])
        empty_cat_df = pd.DataFrame(columns=["Category", "Count", "Percentage (%)"])
        return (
            image_rgb,
            preprocessed_mask,
            cv2.cvtColor(detected_annotated, cv2.COLOR_BGR2RGB),
            image_rgb,
            empty_df, empty_cat_df, 0,
            "No products detected -- try a photo with clearly separated items on a plain surface.",
            None, None,
        )

    # ---- Segmentation (Module 4) ----
    crops = segment_products(image_bgr, detections, display=False)

    # ---- Classification (Module 5), reusing the ONE model loaded at startup ----
    classify_products_with(_predictor.classifier, crops)

    # ---- Category Mapping (Module 6) ----
    map_products(crops)

    # ---- Counting + Statistics (Module 7) ----
    stats = analyze_products(crops)

    # ---- Visualization (Modules 8-10) ----
    final_annotated = draw_final_annotations(image_bgr, crops)

    report_buffer = io.StringIO()
    with contextlib.redirect_stdout(report_buffer):
        print_console_report(stats)
    report_text = report_buffer.getvalue()

    bar_fig = plot_bar_chart(stats, save_path=None, show=False)
    pie_fig = plot_pie_chart(stats, save_path=None, show=False)  # None if total_products == 0 (won't happen here)

    # ---- Shape tabular outputs ----
    product_rows = [
        {
            "Product #": c["id"],
            "Product": c.get("label", "Unknown"),
            "Confidence (%)": round(c.get("confidence", 0.0) * 100, 1),
            "Category": c.get("category", category_mapping.UNKNOWN_CATEGORY),
        }
        for c in crops
    ]
    product_df = pd.DataFrame(product_rows)

    category_rows = [
        {
            "Category": cat,
            "Count": stats["category_counts"][cat],
            "Percentage (%)": round(stats["category_percentages"][cat], 1),
        }
        for cat in category_mapping.CATEGORY_NAMES
        if stats["category_counts"][cat] > 0
    ]
    if stats["category_counts"].get(category_mapping.UNKNOWN_CATEGORY, 0) > 0:
        category_rows.append({
            "Category": category_mapping.UNKNOWN_CATEGORY,
            "Count": stats["category_counts"][category_mapping.UNKNOWN_CATEGORY],
            "Percentage (%)": round(
                stats["category_percentages"].get(category_mapping.UNKNOWN_CATEGORY, 0.0), 1
            ),
        })
    category_df = pd.DataFrame(category_rows)

    return (
        image_rgb,
        preprocessed_mask,
        cv2.cvtColor(detected_annotated, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(final_annotated, cv2.COLOR_BGR2RGB),
        product_df,
        category_df,
        stats["total_products"],
        report_text,
        bar_fig,
        pie_fig,
    )