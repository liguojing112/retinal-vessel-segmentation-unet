"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

系统配置模块。

负责加载 .env 并集中管理路径、阈值、上传限制与默认指标配置。"""


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
# 演示服务优先加载 demo_system/models/ 下的权重（当前为消融实验 loss_bce_dice 最优 best_model.pth）
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"

MAX_CONTENT_LENGTH = 80 * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_UPLOADS", "20"))
MODEL_INPUT_SIZE = (512, 512)
PREDICTION_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".pgm", ".pbm"}
CHECKPOINT_CANDIDATES = [
    MODELS_DIR / "best_model.pth",
    CHECKPOINT_DIR / "best_model_800.pth",
    CHECKPOINT_DIR / "best_model.pth",
    CHECKPOINT_DIR / "final_model.pth",
]
# 与 experiments/results/loss_bce_dice/metrics.json 一致，作为无 checkpoint 元数据时的兜底展示
DEFAULT_METRICS = {
    "dice": 89.72,
    "sensitivity": 88.14,
    "specificity": 99.48,
}
