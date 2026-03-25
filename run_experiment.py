"""配置驱动的一键实验入口。

Usage:
    python run_experiment.py --config experiments/configs/baseline.yaml
    python run_experiment.py --config experiments/configs/baseline.yaml --seed 3407
    python run_experiment.py --config experiments/configs/baseline.yaml --eval-only
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 可配置模型（支持消融开关）
# ---------------------------------------------------------------------------

class PlainConvBlock(nn.Module):
    """不带残差跳连的纯卷积块。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class ConfigurableUNet(nn.Module):
    """支持消融开关的 U-Net。

    与 demo_system 里 ImprovedUNet 架构一致，但可通过参数关闭各模块。
    当全部开关为 True 时，权重结构完全兼容 ImprovedUNet。
    """

    def __init__(
        self,
        use_residual: bool = True,
        use_channel_attention: bool = True,
        use_multiscale: bool = True,
    ) -> None:
        super().__init__()
        block_cls = ResidualBlock if use_residual else PlainConvBlock

        self.enc1 = block_cls(1, 64)
        self.enc2 = block_cls(64, 128)
        self.enc3 = block_cls(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.att3 = ChannelAttention(256) if use_channel_attention else nn.Identity()

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = block_cls(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block_cls(128, 64)

        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.ms_conv = nn.ModuleList([
                nn.Conv2d(64, 32, 3, padding=1),
                nn.Conv2d(64, 32, 5, padding=2),
            ])
            self.ms_fuse = nn.Conv2d(64, 64, 1)
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e3 = self.att3(e3)
        d3 = self.up3(e3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        if self.use_multiscale:
            ms_feats = [conv(d2) for conv in self.ms_conv]
            d2 = self.ms_fuse(torch.cat(ms_feats, dim=1))
        else:
            d2 = self.single_conv(d2)

        return self.sigmoid(self.final(d2))


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------

class RandomAugment:
    ANGLES = [0, 90, 180, 270]

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.3:
            angle = random.choice(self.ANGLES)
            img = self._rotate(img, angle)
            mask = self._rotate(mask, angle)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = np.clip(cv2.convertScaleAbs(img, alpha=alpha, beta=beta), 0, 255).astype(np.uint8)
        return img, mask

    @staticmethod
    def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------------------
# 数据集（可配置预处理）
# ---------------------------------------------------------------------------

class FundusDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        use_clahe: bool = True,
        use_gamma: bool = True,
        gamma: float = 1.2,
        clahe_clip: float = 2.0,
        input_size: int = 512,
    ) -> None:
        self.train = train
        self.aug = RandomAugment() if train else None
        self.use_clahe = use_clahe
        self.use_gamma = use_gamma
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.input_size = input_size
        self.samples: list[tuple[str, str]] = []

        split = "training" if train else "test"
        img_dir = os.path.join(root_dir, split, "images")
        mask_dir = os.path.join(root_dir, split, "mask")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(".png"):
                continue
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                self.samples.append((os.path.join(img_dir, fname), mask_path))

        tag = "训练" if train else "测试"
        print(f"  [{tag}集] 加载 {len(self.samples)} 张 from {img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise ValueError(f"读取失败: {img_path}")

        if self.train and self.aug:
            img, mask = self.aug(img, mask)

        green = img[:, :, 1] if len(img.shape) == 3 else img

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            green = clahe.apply(green)

        if self.use_gamma:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            green = cv2.LUT(green, table)

        normalized = green.astype(np.float32) / 255.0
        resized = cv2.resize(normalized, (self.input_size, self.input_size))
        img_tensor = torch.from_numpy(resized).unsqueeze(0)

        mask_bin = (mask > 128).astype(np.float32)
        mask_resized = cv2.resize(mask_bin, (self.input_size, self.input_size))
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)
        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class ComboLoss(nn.Module):
    """Weighted BCE + Dice + FN penalty，与 train_final.py 一致。"""

    def __init__(self, w_dice: float = 0.7, w_bce: float = 0.3) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice = DiceLoss()
        self.bce = nn.BCELoss(reduction="none")
        self.pos_weight = 10.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_raw = self.bce(pred, target)
        weights = torch.where(target > 0.5, self.pos_weight, 1.0)
        weighted_bce = (bce_raw * weights).mean()
        fn_penalty = ((target - pred).clamp(min=0) ** 2).mean() * 3.0
        return self.w_dice * dice_loss + self.w_bce * weighted_bce + fn_penalty


class BCEDiceLoss(nn.Module):
    """Plain BCE + Dice，无 pos_weight/fn_penalty。"""

    def __init__(self, w_dice: float = 0.5, w_bce: float = 0.5) -> None:
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(pred, target) + self.w_bce * self.bce(pred, target)


def build_criterion(loss_name: str) -> nn.Module:
    """根据配置名返回损失函数。"""
    mapping: dict[str, nn.Module] = {
        "combo": ComboLoss(),
        "bce": nn.BCELoss(),
        "dice": DiceLoss(),
        "bce_dice": BCEDiceLoss(),
    }
    if loss_name not in mapping:
        raise ValueError(f"不支持的损失函数: {loss_name}，可选: {list(mapping.keys())}")
    return mapping[loss_name]


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """计算分割指标，返回值为 0~1 浮点数。"""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)
    smooth = 1e-6
    tp = int(((pred_bin == 1) & (target_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (target_bin == 0)).sum())
    fp = int(((pred_bin == 1) & (target_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (target_bin == 1)).sum())
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else 0.0
    specificity = (tn + smooth) / (tn + fp + smooth) if (tn + fp) > 0 else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 设置随机种子
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------

def train(cfg: dict) -> Path:
    """执行训练，返回最佳 checkpoint 路径。"""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    prep_cfg = cfg.get("preprocessing", {})
    exp_name = cfg["name"]

    device = torch.device(train_cfg.get("device", "cpu"))
    if train_cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，回退到 CPU")
        device = torch.device("cpu")

    print(f"\n{'=' * 60}")
    print(f"实验: {exp_name} | seed={seed} | device={device}")
    print(f"{'=' * 60}")

    model = ConfigurableUNet(
        use_residual=model_cfg.get("use_residual", True),
        use_channel_attention=model_cfg.get("use_channel_attention", True),
        use_multiscale=model_cfg.get("use_multiscale", True),
    ).to(device)
    print(f"模型参数量: {count_parameters(model):,}")

    dataset_dir = cfg.get("dataset_dir", "dataset")
    ds_kwargs = dict(
        use_clahe=prep_cfg.get("use_clahe", True),
        use_gamma=prep_cfg.get("use_gamma", True),
        gamma=prep_cfg.get("gamma", 1.2),
        clahe_clip=prep_cfg.get("clahe_clip", 2.0),
        input_size=prep_cfg.get("input_size", 512),
    )
    train_ds = FundusDataset(dataset_dir, train=True, **ds_kwargs)
    val_ds = FundusDataset(dataset_dir, train=False, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 4), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = build_criterion(train_cfg.get("loss", "combo"))
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 0.001), weight_decay=train_cfg.get("weight_decay", 1e-4))

    scheduler_name = train_cfg.get("scheduler", "cosine_warm")
    if scheduler_name == "cosine_warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    epochs = train_cfg.get("epochs", 150)
    patience = train_cfg.get("patience", 40)
    accum_steps = train_cfg.get("accumulation_steps", 4)
    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    out_dir = Path("experiments/results") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pth"

    best_dice = 0.0
    counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_dice": [], "val_sens": [], "val_spec": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks) / accum_steps
            loss.backward()
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * accum_steps

        if len(train_loader) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = train_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        model.eval()
        val_metrics_list: dict[str, list[float]] = {"dice": [], "sensitivity": [], "specificity": []}
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                m = calculate_metrics(outputs.cpu().numpy()[0, 0], masks.cpu().numpy()[0, 0], threshold)
                for k in val_metrics_list:
                    val_metrics_list[k].append(m[k])

        avg_dice = float(np.mean(val_metrics_list["dice"]))
        avg_sens = float(np.mean(val_metrics_list["sensitivity"]))
        avg_spec = float(np.mean(val_metrics_list["specificity"]))
        history["val_dice"].append(avg_dice)
        history["val_sens"].append(avg_sens)
        history["val_spec"].append(avg_spec)

        if scheduler is not None:
            scheduler.step(epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1:3d}] Loss={avg_loss:.4f} | Dice={avg_dice:.4f} "
            f"Sens={avg_sens:.4f} Spec={avg_spec:.4f} | LR={lr_now:.6f}"
        )

        if avg_dice > best_dice:
            best_dice = avg_dice
            counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "dice": best_dice,
                    "sensitivity": avg_sens,
                    "specificity": avg_spec,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  >>> 新最佳 Dice={best_dice:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"\n早停: {patience} 轮未提升")
                break

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n训练完成 | 最佳 Dice: {best_dice:.4f} ({best_dice * 100:.1f}%)")
    return ckpt_path


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------

def evaluate(cfg: dict, ckpt_path: Path) -> dict[str, float]:
    """加载 checkpoint 在测试集上评估，返回指标字典。"""
    model_cfg = cfg.get("model", {})
    prep_cfg = cfg.get("preprocessing", {})
    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    device = torch.device("cpu")
    model = ConfigurableUNet(
        use_residual=model_cfg.get("use_residual", True),
        use_channel_attention=model_cfg.get("use_channel_attention", True),
        use_multiscale=model_cfg.get("use_multiscale", True),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dataset_dir = cfg.get("dataset_dir", "dataset")
    val_ds = FundusDataset(
        dataset_dir,
        train=False,
        use_clahe=prep_cfg.get("use_clahe", True),
        use_gamma=prep_cfg.get("use_gamma", True),
        gamma=prep_cfg.get("gamma", 1.2),
        clahe_clip=prep_cfg.get("clahe_clip", 2.0),
        input_size=prep_cfg.get("input_size", 512),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    all_metrics: dict[str, list[float]] = {"dice": [], "iou": [], "sensitivity": [], "specificity": []}
    inference_times: list[float] = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="评估中"):
            images = images.to(device)
            start = time.perf_counter()
            outputs = model(images)
            elapsed = (time.perf_counter() - start) * 1000
            inference_times.append(elapsed)

            m = calculate_metrics(outputs.cpu().numpy()[0, 0], masks.numpy()[0, 0], threshold)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    result = {
        "name": cfg["name"],
        "dice": round(float(np.mean(all_metrics["dice"])), 4),
        "iou": round(float(np.mean(all_metrics["iou"])), 4),
        "sensitivity": round(float(np.mean(all_metrics["sensitivity"])), 4),
        "specificity": round(float(np.mean(all_metrics["specificity"])), 4),
        "params": count_parameters(model),
        "avg_inference_ms": round(float(np.mean(inference_times)), 2),
        "seed": cfg.get("seed", 42),
        "threshold": threshold,
        "notes": cfg.get("notes", ""),
    }
    return result


# ---------------------------------------------------------------------------
# 写 metrics.json
# ---------------------------------------------------------------------------

def save_metrics(cfg: dict, metrics: dict) -> Path:
    out_dir = Path("experiments/results") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已保存: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="配置驱动的一键实验")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    parser.add_argument("--eval-only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定 checkpoint 路径（eval-only 时有效）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed

    exp_name = cfg["name"]

    if args.eval_only:
        ckpt = Path(args.checkpoint) if args.checkpoint else Path("experiments/results") / exp_name / "best_model.pth"
        if not ckpt.exists():
            print(f"[错误] checkpoint 不存在: {ckpt}")
            return
        metrics = evaluate(cfg, ckpt)
    else:
        ckpt = train(cfg)
        metrics = evaluate(cfg, ckpt)

    save_metrics(cfg, metrics)
    print(f"\n实验 [{exp_name}] 全部完成。")
    print(f"  Dice={metrics['dice']:.4f}  Sens={metrics['sensitivity']:.4f}  "
          f"Spec={metrics['specificity']:.4f}  Params={metrics['params']:,}")


if __name__ == "__main__":
    main()
