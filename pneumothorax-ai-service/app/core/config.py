import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
TEST_IMAGES_DIR = BASE_DIR / "test_images" / "demo_cases_raw"
RESULTS_DIR = BASE_DIR / "results"

# Ensure results dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model weight paths
UNET_WEIGHTS_PATH = WEIGHTS_DIR / "best_unet_plus_plus.pth"
YOLO_B_WEIGHTS_PATH = WEIGHTS_DIR / "best_B_m1280.pt"
YOLO_A_WEIGHTS_PATH = WEIGHTS_DIR / "best_A_l1024.pt"

# Macro Classes for VinDr-CXR
VINDR_CLASSES = {
    0: "Cardiomegaly",
    1: "Aortic enlargement",
    2: "Pleural pathology",
    3: "Parenchymal opacity"
}

# Colors for bounding boxes & masks (Hex or RGB)
CLASS_COLORS = {
    "Cardiomegaly": (255, 69, 0),        # Red-Orange
    "Aortic enlargement": (255, 165, 0),   # Orange
    "Pleural pathology": (30, 144, 255),   # Dodger Blue
    "Parenchymal opacity": (153, 50, 204), # Dark Orchid
    "Pneumothorax": (0, 255, 127)         # Spring Green (Mask)
}

# Thresholds
UNET_THRESHOLD = 0.60
UNET_MIN_PIXELS = 800  # Connected components filter out smaller noise
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
