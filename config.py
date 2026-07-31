import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "freiburg_groceries")
SPLITS_DIR = os.path.join(BASE_DIR, "dataset", "splits") 
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images") 
OUTPUT_DIR = os.path.join(BASE_DIR, "output") 
MODELS_DIR = os.path.join(BASE_DIR, "models") 
REPORT_DIR = os.path.join(BASE_DIR, "report") 

MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_freiburg.keras")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")

CLASS_NAMES = [
    "BEANS", "CAKE", "CANDY", "CEREAL", "CHIPS", "CHOCOLATE", "COFFEE",
    "CORN", "FISH", "FLOUR", "HONEY", "JAM", "JUICE", "MILK", "NUTS",
    "OIL", "PASTA", "RICE", "SODA", "SPICES", "SUGAR", "TEA",
    "TOMATO_SAUCE", "VINEGAR", "WATER",
]
NUM_CLASSES = len(CLASS_NAMES)