import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_local_env(BASE_DIR / ".env")

TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"

MAX_CONTENT_LENGTH = 80 * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_UPLOADS", "20"))
MODEL_INPUT_SIZE = (512, 512)
PREDICTION_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
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
