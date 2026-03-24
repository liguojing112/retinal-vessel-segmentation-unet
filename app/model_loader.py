"""模型加载与指标管理。"""

import os

import torch

from .config import CHECKPOINT_CANDIDATES, DEFAULT_MODEL_DICE, DEFAULT_MODEL_SENS, DEFAULT_MODEL_SPEC
from .model_def import ImprovedUNet


class ModelBundle:
    def __init__(self):
        self.model = ImprovedUNet()
        self.model_dice = DEFAULT_MODEL_DICE
        self.model_sens = DEFAULT_MODEL_SENS
        self.model_spec = DEFAULT_MODEL_SPEC
        self.loaded_checkpoint = '(not loaded)'

    def metrics(self):
        return {
            'dice': round(self.model_dice, 1),
            'sensitivity': round(self.model_sens, 1),
            'specificity': round(self.model_spec, 1),
        }


def load_model():
    bundle = ModelBundle()
    loaded = False

    for checkpoint_path in CHECKPOINT_CANDIDATES:
        if not os.path.exists(checkpoint_path):
            continue
        try:
            if os.path.getsize(checkpoint_path) == 0:
                continue

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict):
                if 'dice' in checkpoint:
                    bundle.model_dice = float(checkpoint['dice']) * 100
                if 'sensitivity' in checkpoint:
                    bundle.model_sens = float(checkpoint['sensitivity']) * 100
                if 'specificity' in checkpoint:
                    bundle.model_spec = float(checkpoint['specificity']) * 100

                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                bundle.model.load_state_dict(state_dict)
                bundle.loaded_checkpoint = checkpoint_path
                print(f"[系统启动] (>_<) 已加载模型: {checkpoint_path}")
                print(
                    f"[模型性能] Dice: {bundle.model_dice:.1f}%, 敏感度: {bundle.model_sens:.1f}%, 特异性: {bundle.model_spec:.1f}%"
                )
                loaded = True
                break
        except Exception as e:
            print(f"[警告] 加载失败 {checkpoint_path}: {e}")
            continue

    if not loaded:
        print('[警告] 未能加载任何模型权重，使用随机初始化')

    bundle.model.eval()
    return bundle
