"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

图像编码工具。

将 NumPy 图像数组编码为 Base64 字符串，便于 API JSON 返回。"""


import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _to_base64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")