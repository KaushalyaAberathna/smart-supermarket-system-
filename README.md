Digital Image Processing: Custom Segmentation & Auto-Labeling Pipeline
A modular Digital Image Processing (DIP) project designed to train machine learning models on individual images, synthesize learned patterns, and run a collective pipeline to segment, extract, and automatically label object features across datasets.

Key Features
Single-Image Training: Train dedicated models fine-tuned on individual images to capture granular visual features and textures.
Collective Inference & Extraction: Aggregate trained parameters to run batch processing across entire image collections.
Feature Segmentation: Automatically isolate and crop objects, ROI (Regions of Interest), or specific contours.
Auto-Labeling & Naming: Intelligently classify and generate dynamic, descriptive labels for each extracted component.
Customizable DIP Operations: Built-in support for standard image processing filters (thresholding, edge detection, morphological transforms) prior to model inference.
Workflow Overview
Preprocessing: Apply noise reduction, normalization, and color-space transformations.
Individual Model Training: Train per-image feature extractors to identify key attributes.
Collective Aggregation: Pass dataset through the unified model framework.
Segmentation & Extraction: Isolate target elements and save them as individual assets.
Auto-Naming: Output structured data with corresponding metadata and descriptive labels.
Getting Started
Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

pip install opencv-python numpy scikit-image torch torchvision matplotlib
