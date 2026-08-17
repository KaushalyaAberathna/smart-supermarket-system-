# Image Processing Based Smart Supermarket Product Identification System

A fully offline pipeline that takes a photo of multiple supermarket products
laid out on a table/basket, detects each product with classical OpenCV
image processing, classifies it with a locally fine-tuned MobileNetV2 model,
maps it to a supermarket category, and reports counts/statistics -- with
both a CLI (`main.py`) and an interactive Gradio web app (`gradio_app.py`).

No cloud APIs or online inference services are used anywhere in this project.

## Pipeline

```
Input Image
  -> Image Acquisition        (utils.py)
  -> Image Preprocessing       (preprocessing.py)
  -> Object Detection          (detection.py)
  -> Product Segmentation      (segmentation.py)
  -> Classification            (classification.py, MobileNetV2)
  -> Category Mapping          (category_mapping.py)
  -> Counting                  (statistics.py)
  -> Statistical Analysis      (statistics.py)
  -> Visualization             (visualization.py)
  -> Gradio Output             (gradio_app.py)
```