"""图像预处理。"""

import cv2
import numpy as np

from .config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, MODEL_INPUT_SIZE, PREPROCESS_GAMMA
from .image_codec import robust_imread


def preprocess(img_path):
    try:
        img = robust_imread(img_path)
    except Exception as e:
        raise ValueError(f"图像读取失败: {str(e)}") from e

    if img is None:
        raise ValueError("图像为空")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green = img_rgb[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(green)

    inv_gamma = 1.0 / PREPROCESS_GAMMA
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    corrected = cv2.LUT(enhanced, table)

    normalized = corrected.astype(np.float32) / 255.0
    resized = cv2.resize(normalized, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    return resized, img_rgb
