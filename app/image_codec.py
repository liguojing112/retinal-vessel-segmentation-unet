"""图像读取与编码工具。"""

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def robust_imread(filepath):
    img = None
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass

    try:
        pil_img = Image.open(filepath)
        img = np.array(pil_img)
        if len(img.shape) == 2 and pil_img.mode == 'P':
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        pass

    raise ValueError(f"无法读取图像: {filepath}")


def to_base64(arr):
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
