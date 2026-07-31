import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "freiburg_groceries")
SPLITS_DIR = os.path.join(BASE_DIR, "dataset", "splits") 
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images") 
OUTPUT_DIR = os.path.join(BASE_DIR, "output") 
MODELS_DIR = os.path.join(BASE_DIR, "models") 
REPORT_DIR = os.path.join(BASE_DIR, "report") 