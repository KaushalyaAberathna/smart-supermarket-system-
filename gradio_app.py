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