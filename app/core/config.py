from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"

MAX_CONTENT_LENGTH = 80 * 1024 * 1024
MAX_FILES_PER_REQUEST = 20
MODEL_INPUT_SIZE = (512, 512)
PREDICTION_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".pgm", ".pbm"}
CHECKPOINT_CANDIDATES = [
    CHECKPOINT_DIR / "best_model_800.pth",
    CHECKPOINT_DIR / "best_model.pth",
    CHECKPOINT_DIR / "final_model.pth",
]
DEFAULT_METRICS = {
    "dice": 86.5,
    "sensitivity": 82.0,
    "specificity": 97.5,
}                                                                            