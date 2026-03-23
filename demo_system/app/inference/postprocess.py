import cv2
import numpy as np

from app.core.config import PREDICTION_THRESHOLD


def threshold_prediction(prediction: np.ndarray, threshold: float = PREDICTION_THRESHOLD) -> np.ndarray:
    return (prediction > threshold).astype(np.uint8) * 255


def resize_mask(mask: np.ndarray, original_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = original_shape[:2]
    return cv2.resize(mask, (width, height))


def overlay_mask(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = original_rgb.copy()
    overlay[mask > 128] = [255, 0, 0]
    return overlay
