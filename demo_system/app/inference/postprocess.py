import numpy as np

from app.core.config import PREDICTION_THRESHOLD


def _import_cv2():
    import cv2

    return cv2


def threshold_prediction(prediction: np.ndarray, threshold: float = PREDICTION_THRESHOLD) -> np.ndarray:
    return (prediction > threshold).astype(np.uint8) * 255


def resize_mask(mask: np.ndarray, original_shape: tuple[int, int, int]) -> np.ndarray:
    cv2 = _import_cv2()
    height, width = original_shape[:2]
    return cv2.resize(mask, (width, height))


def overlay_mask(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = original_rgb.copy()
    overlay[mask > 128] = [255, 0, 0]
    return overlay


def evaluate_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = (pred_mask > 128).astype(np.uint8)
    gt = (gt_mask > 128).astype(np.uint8)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    dice = (2 * tp) / max((2 * tp + fp + fn), 1)
    sensitivity = tp / max((tp + fn), 1)
    specificity = tn / max((tn + fp), 1)
    return {
        "dice": round(dice * 100, 3),
        "sensitivity": round(sensitivity * 100, 3),
        "specificity": round(specificity * 100, 3),
    }
