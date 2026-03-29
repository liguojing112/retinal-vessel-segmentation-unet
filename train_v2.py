"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

训练脚本 v2。

在基础版本上加入训练策略改进，用于第二轮实验对比。"""





import os

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np

import cv2

from PIL import Image

from tqdm import tqdm

import json

import random





# ==================== 模型定义（保持不变） ====================

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

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



    def forward(self, x):

        return self.relu(self.conv(x) + self.skip(x))





class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(channels, channels // reduction, 1, bias=False),

            nn.ReLU(),

            nn.Conv2d(channels // reduction, channels, 1, bias=False),

        )

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))

        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out) * x





class ImprovedUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = ResidualBlock(1, 64)

        self.enc2 = ResidualBlock(64, 128)

        self.enc3 = ResidualBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.att3 = ChannelAttention(256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.dec3 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec2 = ResidualBlock(128, 64)

        self.ms_conv = nn.ModuleList(

            [

                nn.Conv2d(64, 32, 3, padding=1),

                nn.Conv2d(64, 32, 5, padding=2),

            ]

        )

        self.ms_fuse = nn.Conv2d(64, 64, 1)

        self.final = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e3 = self.att3(e3)

        d3 = self.up3(e3)

        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)

        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        ms_feats = [conv(d2) for conv in self.ms_conv]

        d2 = self.ms_fuse(torch.cat(ms_feats, dim=1))

        return self.sigmoid(self.final(d2))





# ==================== 数据增强（新增） ====================

class RandomAugment:

    """随机数据增强"""



    def __init__(self):

        self.angles = [0, 90, 180, 270]



    def __call__(self, img, mask):

        # 随机旋转

        if random.random() > 0.5:

            angle = random.choice(self.angles)

            img = self.rotate(img, angle)

            mask = self.rotate(mask, angle)



        # 随机水平翻转

        if random.random() > 0.5:

            img = cv2.flip(img, 1)

            mask = cv2.flip(mask, 1)



        # 随机垂直翻转

        if random.random() > 0.5:

            img = cv2.flip(img, 0)

            mask = cv2.flip(mask, 0)



        return img, mask



    def rotate(self, img, angle):

        if angle == 0:

            return img

        h, w = img.shape[:2]

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)





# ==================== 数据集（修复路径 + 增强） ====================

class DRIVEDataset(Dataset):

    def __init__(self, root_dir, train=True, augment=True):

        self.root_dir = root_dir

        self.train = train

        self.augment = augment and train

        self.aug = RandomAugment()

        self.samples = []



        # 自动检测文件夹结构

        if train:

            img_dir = os.path.join(root_dir, "training", "images")

            # 训练集可能是 1st_manual 或 mask

            if os.path.exists(os.path.join(root_dir, "training", "1st_manual")):

                mask_dir = os.path.join(root_dir, "training", "1st_manual")

            else:

                mask_dir = os.path.join(root_dir, "training", "mask")

        else:

            img_dir = os.path.join(root_dir, "test", "images")

            # 测试集可能是 mask 或 1st_manual

            if os.path.exists(os.path.join(root_dir, "test", "mask")):

                mask_dir = os.path.join(root_dir, "test", "mask")

            elif os.path.exists(os.path.join(root_dir, "test", "1st_manual")):

                mask_dir = os.path.join(root_dir, "test", "1st_manual")

            else:

                raise ValueError(

                    "找不到 test 集的标注文件夹，请确认有 test/mask 或 test/1st_manual"

                )



        print(f"加载 {'训练' if train else '测试'}集:")

        print(f"  图像路径: {img_dir}")

        print(f"  标注路径: {mask_dir}")



        # 构建文件列表（增强文件名匹配）

        for f in sorted(os.listdir(img_dir)):

            if f.endswith((".tif", ".ppm", ".png", ".jpg", ".gif")):

                img_path = os.path.join(img_dir, f)

                base_name = f.split("_")[0]  # 如 "01" 从 "01_test.tif"



                # 尝试多种可能的标注文件名（支持真实DRIVE和合成数据）

                possible_masks = [

                    f"{base_name}_test_mask.gif",

                    f"{base_name}_manual1.gif",  # 标准DRIVE: 01_manual1.gif

                    f"{base_name}_test.gif",  # 合成数据可能: 01_test.gif

                    f"{base_name}.gif",  # 简化: 01.gif

                    f"{base_name}_mask.gif",  # 其他: 01_mask.gif

                    f"{base_name}_manual.gif",  # 变种: 01_manual.gif

                ]



                mask_path = None

                for pm in possible_masks:

                    mp = os.path.join(mask_dir, pm)

                    if os.path.exists(mp):

                        mask_path = mp

                        break



                if mask_path:

                    self.samples.append((img_path, mask_path))

                else:

                    print(f"  ⚠️ 跳过 {f}，未找到对应标注（尝试了: {possible_masks}）")



        print(f"  ✅ 成功加载: {len(self.samples)} 张图像")

        if len(self.samples) == 0:

            print(

                f"  ❌ 错误: {mask_dir} 中的文件: {os.listdir(mask_dir)[:5]}..."

            )  # 显示实际文件

            raise ValueError("未找到匹配的数据对")



    def preprocess(self, img_path, mask_path):

        # 读取图像

        img = cv2.imread(img_path)

        if img is None:

            img = np.array(Image.open(img_path).convert("RGB"))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



        # 读取掩膜（gif格式）

        try:

            mask = np.array(Image.open(mask_path).convert("L"))

        except Exception as e:

            print(f"读取mask失败 {mask_path}: {e}")

            mask = np.zeros((584, 565), dtype=np.uint8)



        # 数据增强（训练时）

        if self.train and self.augment:

            img, mask = self.aug(img, mask)



        # 统一尺寸为512x512

        img = cv2.resize(img, (512, 512))

        mask = cv2.resize(mask, (512, 512))



        # 预处理：绿色通道 + CLAHE + 伽马校正

        if len(img.shape) == 3:

            green = img[:, :, 1]

        else:

            green = img



        # CLAHE

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        enhanced = clahe.apply(green)



        # 伽马校正

        gamma = 1.2

        inv_gamma = 1.0 / gamma

        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(

            np.uint8

        )

        corrected = cv2.LUT(enhanced, table)



        # 归一化

        img_tensor = torch.from_numpy(corrected.astype(np.float32) / 255.0).unsqueeze(0)



        # 掩膜二值化（DRIVE中血管为255，背景为0）

        mask = (mask > 128).astype(np.float32)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)



        return img_tensor, mask_tensor



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]

        return self.preprocess(img_path, mask_path)





# ==================== 损失函数（权重调整） ====================

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, pred, target):

        smooth = 1e-5

        intersection = (pred * target).sum()

        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1 - dice





class CombinedLoss(nn.Module):

    def __init__(self, w_dice=0.8, w_bce=0.2):  # Dice 权重提高到 0.8

        super().__init__()

        self.w_dice = w_dice

        self.w_bce = w_bce

        self.dice = DiceLoss()

        self.bce = nn.BCELoss(reduction="none")

        self.pos_weight = 20.0  # 血管像素权重提高到 20 倍（之前是 8）



    def forward(self, pred, target):

        dice_loss = self.dice(pred, target)



        bce_loss = self.bce(pred, target)

        # 使用二元交叉熵的 pos_weight 参数

        weights = torch.where(

            target > 0.5, torch.tensor(self.pos_weight), torch.tensor(1.0)

        )

        weighted_bce = (bce_loss * weights).mean()



        # 额外惩罚假阴性（漏检血管）

        fn_penalty = ((1 - pred) * target).mean() * 10.0  # 漏检惩罚



        return self.w_dice * dice_loss + self.w_bce * weighted_bce + fn_penalty





# ==================== 指标计算（修复NaN） ====================

def calculate_metrics(pred, target, threshold=0.5):

    """修复版指标计算，防止NaN"""

    pred_bin = (pred > threshold).astype(np.uint8)

    target_bin = (target > threshold).astype(np.uint8)



    smooth = 1e-6



    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))

    tn = int(np.sum((pred_bin == 0) & (target_bin == 0)))

    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))

    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))



    total = tp + tn + fp + fn

    if total == 0:

        return {

            "dice": 0.0,

            "iou": 0.0,

            "sensitivity": 0.0,

            "specificity": 0.0,

            "accuracy": 0.0,

        }



    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    sensitivity = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else 0.0

    specificity = (tn + smooth) / (tn + fp + smooth) if (tn + fp) > 0 else 0.0

    accuracy = (tp + tn) / total



    return {

        "dice": float(dice),

        "iou": float(iou),

        "sensitivity": float(sensitivity),

        "specificity": float(specificity),

        "accuracy": float(accuracy),

    }





# ==================== 训练流程（增加早停） ====================

def train_model(

    root_dir, epochs=150, batch_size=4, lr=0.001, device="cuda", patience=20

):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")



    # 数据集（训练时开启增强）

    train_dataset = DRIVEDataset(root_dir, train=True, augment=True)

    val_dataset = DRIVEDataset(root_dir, train=False, augment=False)



    train_loader = DataLoader(

        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0

    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



    # 模型

    model = ImprovedUNet().to(device)



    # 损失和优化器

    criterion = CombinedLoss(w_dice=0.6, w_bce=0.4)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)



    # 学习率调度（带预热）

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(

        optimizer, T_0=10, T_mult=2

    )



    # 早停机制

    best_dice = 0.0

    counter = 0

    history = {"train_loss": [], "val_dice": [], "val_sens": [], "val_spec": []}



    os.makedirs("checkpoints", exist_ok=True)



    print(f"\n开始训练: {epochs} epochs, batch_size={batch_size}, lr={lr}")

    print("=" * 70)



    for epoch in range(epochs):

        # === 训练阶段 ===

        model.train()

        train_loss = 0.0



        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):

            images = images.to(device)

            masks = masks.to(device)



            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()



            # 梯度裁剪防止爆炸

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)



            optimizer.step()

            train_loss += loss.item()



        avg_train_loss = train_loss / len(train_loader)

        history["train_loss"].append(avg_train_loss)



        # === 验证阶段 ===

        model.eval()

        val_metrics = {"dice": [], "sensitivity": [], "specificity": [], "accuracy": []}



        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)

                masks = masks.to(device)



                outputs = model(images)

                outputs_np = outputs.cpu().numpy()

                masks_np = masks.cpu().numpy()



                metrics = calculate_metrics(outputs_np[0, 0], masks_np[0, 0])

                for k in val_metrics:

                    val_metrics[k].append(metrics[k])



        # 计算平均

        avg_val_dice = np.mean(val_metrics["dice"])

        avg_val_sens = np.mean(val_metrics["sensitivity"])

        avg_val_spec = np.mean(val_metrics["specificity"])

        avg_val_acc = np.mean(val_metrics["accuracy"])



        history["val_dice"].append(avg_val_dice)

        history["val_sens"].append(avg_val_sens)



        # 学习率调整

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]



        # 打印进度

        print(

            f"[Epoch {epoch + 1:3d}] Loss: {avg_train_loss:.4f} | "

            f"Dice: {avg_val_dice:.4f} | Sens: {avg_val_sens:.4f} | "

            f"Spec: {avg_val_spec:.4f} | Acc: {avg_val_acc:.4f} | LR: {current_lr:.6f}"

        )



        # 保存最佳模型 + 早停判断

        if avg_val_dice > best_dice:

            best_dice = avg_val_dice

            counter = 0

            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "dice": best_dice,

                    "sensitivity": avg_val_sens,

                    "specificity": avg_val_spec,

                },

                "checkpoints/best_model.pth",

            )

            print(f"  >>> 保存最佳模型 (Dice: {best_dice:.4f})")

        else:

            counter += 1

            if counter >= patience:

                print(f"\n早停: {patience} 轮未提升，最佳 Dice: {best_dice:.4f}")

                break



    # 保存最终模型和历史

    torch.save(model.state_dict(), "checkpoints/final_model.pth")

    with open("checkpoints/history_v2.json", "w") as f:

        json.dump(history, f)



    print("=" * 70)

    print(f"训练完成！最佳 Dice: {best_dice:.4f} ({best_dice * 100:.1f}%)")

    print(f"目标: 84.0%，差距: {abs(0.84 - best_dice) * 100:.1f}%")

    print("模型保存在: checkpoints/best_model.pth")





if __name__ == "__main__":

    DATASET_PATH = "dataset"



    if not os.path.exists(DATASET_PATH):

        print(f"错误: 数据集路径 {DATASET_PATH} 不存在！")

        exit(1)



    # 开始训练（150轮，早停耐心20轮）

    train_model(

        root_dir=DATASET_PATH,

        epochs=150,

        batch_size=4,

        lr=0.001,

        device="cuda",

        patience=20,  # 20轮没提升就停

    )

