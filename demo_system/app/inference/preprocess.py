from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import MODEL_INPUT_SIZE


def robust_imread(filepath: str | Path) -> np.ndarray:
    path = str(filepath)

    try:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None and image.size > 0:
            return image
    except Exception:
        pass

    try:
        pil_img = Image.open(path)
        image = np.array(pil_img)
        if len(image.shape) == 2 and pil_img.mode == "P":
            pil_img = pil_img.convert("RGB")
            image = np.array(pil_img)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception as exc:
        raise ValueError(f"无法读取图像: {filepath}") from exc


def preprocess(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    image = robust_imread(filepath)
    if image is None:
        raise ValueError("图像为空")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    green_channel = image_rgb[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    corrected = cv2.LUT(enhanced, table)

    normalized = corrected.astype(np.float32) / 255.0
    resized = cv2.resize(normalized, MODEL_INPUT_SIZE)
    return resized, image_rgb
