from dataclasses import dataclass
from typing import Any

import torch

from app.core.config import BASE_DIR, CHECKPOINT_CANDIDATES, DEFAULT_METRICS
from app.inference.model_def import ImprovedUNet


@dataclass
class LoadedModel:
    model: ImprovedUNet
    checkpoint: str
    metrics: dict[str, float]


class ModelLoader:
    def __init__(self) -> None:
        self._loaded_model: LoadedModel | None = None

    def load(self) -> LoadedModel:
        if self._loaded_model is not None:
            return self._loaded_model

        model = ImprovedUNet()
        metrics = DEFAULT_METRICS.copy()
        loaded_checkpoint = "(not loaded)"

        for checkpoint_path in CHECKPOINT_CANDIDATES:
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                continue
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state_dict = self._extract_state_dict(checkpoint)
                metrics.update(self._extract_metrics(checkpoint))
                model.load_state_dict(state_dict)
                loaded_checkpoint = str(checkpoint_path.relative_to(BASE_DIR.parent))
                break
            except Exception as exc:
                print(f"[警告] 加载失败 {checkpoint_path}: {exc}")

        model.eval()
        self._loaded_model = LoadedModel(model=model, checkpoint=loaded_checkpoint, metrics=metrics)
        return self._loaded_model

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("checkpoint 格式不支持")

    @staticmethod
    def _extract_metrics(checkpoint: Any) -> dict[str, float]:
        if not isinstance(checkpoint, dict):
            return {}
        metrics = {}
        for key in ("dice", "sensitivity", "specificity"):
            value = checkpoint.get(key)
            if value is not None:
                metrics[key] = float(value) * 100
        return metrics


model_loader = ModelLoader()