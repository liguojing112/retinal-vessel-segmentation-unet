import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _to_base64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
