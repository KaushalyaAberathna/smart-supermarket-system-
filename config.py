"""
Every other module reads its tunable values (paths, thresholds, image sizes,
class names) from here instead of hardcoding them. This is the single place
to change behaviour project-wide, per the "avoid hardcoded values" requirement.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "freiburg_groceries") # raw training data (25 class folders)
SPLITS_DIR = os.path.join(BASE_DIR, "dataset", "splits") # generated train/val/test file lists
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images") # user-supplied basket/table photos
OUTPUT_DIR = os.path.join(BASE_DIR, "output") # annotated images, charts, reports
MODELS_DIR = os.path.join(BASE_DIR, "models") # trained model weights + class index map
REPORT_DIR = os.path.join(BASE_DIR, "report") # project report source/output

MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_freiburg.keras")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")

# The 25 Freiburg Groceries classes, in alphabetical order. This ordering
# matters: Keras' image_dataset_from_directory / ImageDataGenerator assign
# label indices by sorting class-folder names alphabetically, so any code
# that maps a predicted index back to a class name must use this exact order.
CLASS_NAMES = [
    "BEANS", "CAKE", "CANDY", "CEREAL", "CHIPS", "CHOCOLATE", "COFFEE",
    "CORN", "FISH", "FLOUR", "HONEY", "JAM", "JUICE", "MILK", "NUTS",
    "OIL", "PASTA", "RICE", "SODA", "SPICES", "SUGAR", "TEA",
    "TOMATO_SAUCE", "VINEGAR", "WATER",
]
NUM_CLASSES = len(CLASS_NAMES)

# Dataset split ratios (no official split files ship with Freiburg Groceries,
# so we generate a stratified split ourselves). Must sum to 1.0.

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# CLASSIFICATION (MobileNetV2) SETTINGS
CLASSIFIER_IMG_SIZE = (224, 224) # MobileNetV2's native input resolution
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_EPOCHS_HEAD = 15 # epochs for training the new classification head
CLASSIFIER_EPOCHS_FINE_TUNE = 20 # additional epochs for fine-tuning top MobileNetV2 layers (early stopping guards against overrun)

CLASSIFIER_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 3e-5 
FINE_TUNE_WEIGHT_DECAY = 1e-2 
FINE_TUNE_AT_LAYER = 100   # unfreeze layers from this index onward during fine-tuning (of MobileNetV2's 154 layers) 
USE_CLASS_WEIGHT = False 
OVERSAMPLE_MINORITY_CLASSES = True 
LABEL_SMOOTHING = 0.1 
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5
CLASSIFIER_EARLY_STOPPING_PATIENCE = 4 
USE_DATA_AUGMENTATION = True  
USE_MIXUP = True 
MIXUP_ALPHA = 0.2 

PREPROCESS_RESIZE_WIDTH = 800 # working resolution for detection (keeps aspect ratio)
GAUSSIAN_KERNEL_SIZE = (5, 5) 
GAUSSIAN_SIGMA = 0
ADAPTIVE_THRESH_BLOCK_SIZE = 21 
ADAPTIVE_THRESH_C = 5
THRESHOLD_METHOD = "adaptive"
BINARY_THRESH_VALUE = 127
THRESHOLD_INVERT = True
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS = 2

MIN_CONTOUR_AREA = 1500 
BOUNDING_BOX_COLOR = (0, 255, 0) 
BOUNDING_BOX_THICKNESS = 2
LABEL_FONT = "FONT_HERSHEY_SIMPLEX"
LABEL_FONT_SCALE = 0.6
LABEL_COLOR = (0, 255, 0)
LABEL_THICKNESS = 2

SHOW_PLOTS = True 
SAVE_INTERMEDIATE_STEPS = True