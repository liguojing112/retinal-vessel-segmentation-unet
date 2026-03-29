"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

训练脚本 v3。

包含进一步结构/损失优化与实验配置，用于第三轮性能提升实验。"""





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

import torch.nn.functional as F



# ==================== 模型定义（增加Deep Supervision可选） ====================

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch)

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

            nn.Conv2d(channels, channels//reduction, 1, bias=False),

            nn.ReLU(),

            nn.Conv2d(channels//reduction, channels, 1, bias=False)

        )

        self.sigmoid = nn.Sigmoid()

    

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))

        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out) * x



class ImprovedUNet(nn.Module):

    def __init__(self, deep_supervision=False):

        super().__init__()

        self.deep_supervision = deep_supervision

        

        # Encoder

        self.enc1 = ResidualBlock(1, 64)

        self.enc2 = ResidualBlock(64, 128)

        self.enc3 = ResidualBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.att3 = ChannelAttention(256)

        

        # Decoder

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.dec3 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec2 = ResidualBlock(128, 64)

        

        # Multi-scale

        self.ms_conv = nn.ModuleList([

            nn.Conv2d(64, 32, 3, padding=1),

            nn.Conv2d(64, 32, 5, padding=2),

        ])

        self.ms_fuse = nn.Conv2d(64, 64, 1)

        

        # Heads

        self.final = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()

        

        # Deep supervision heads (optional)

        if deep_supervision:

            self.aux1 = nn.Conv2d(128, 1, 1)

            self.aux2 = nn.Conv2d(64, 1, 1)

    

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e3 = self.att3(e3)

        

        # Decoder

        d3 = self.up3(e3)

        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        

        d2 = self.up2(d3)

        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        

        # Multi-scale

        ms_feats = [conv(d2) for conv in self.ms_conv]

        d2 = self.ms_fuse(torch.cat(ms_feats, dim=1))

        

        out = self.sigmoid(self.final(d2))

        

        if self.deep_supervision and self.training:

            aux1 = self.sigmoid(self.aux1(d3))

            aux2 = self.sigmoid(self.aux2(d2))

            return out, aux1, aux2

        

        return out



# ==================== 高级数据增强（关键！） ====================

class ElasticDeform:

    """弹性形变（模拟血管形变）"""

    def __init__(self, alpha=100, sigma=10):

        self.alpha = alpha

        self.sigma = sigma

    

    def __call__(self, img, mask):

        if random.random() > 0.7:  # 30%概率

            shape = img.shape[:2]

            dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0,0), self.sigma) * self.alpha

            dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0,0), self.sigma) * self.alpha

            

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

            map_x = (x + dx).astype(np.float32)

            map_y = (y + dy).astype(np.float32)

            

            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        

        return img, mask



class RandomAugment:

    def __init__(self):

        self.angles = [0, 90, 180, 270]

        self.elastic = ElasticDeform(alpha=80, sigma=8)  # 添加弹性形变

    

    def __call__(self, img, mask):

        # 弹性形变

        img, mask = self.elastic(img, mask)

        

        # 随机旋转

        if random.random() > 0.3:

            angle = random.choice(self.angles)

            img = self.rotate(img, angle)

            mask = self.rotate(mask, angle)

        

        # 随机翻转

        if random.random() > 0.5:

            img = cv2.flip(img, 1)

            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:

            img = cv2.flip(img, 0)

            mask = cv2.flip(mask, 0)

        

        # 随机亮度/对比度（颜色抖动）

        if random.random() > 0.5:

            alpha = random.uniform(0.8, 1.2)  # 对比度

            beta = random.uniform(-20, 20)    # 亮度

            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            img = np.clip(img, 0, 255).astype(np.uint8)

        

        # 随机伽马校正

        if random.random() > 0.5:

            gamma = random.uniform(0.8, 1.4)

            inv_gamma = 1.0 / gamma

            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

            img = cv2.LUT(img, table)

        

        return img, mask

    

    def rotate(self, img, angle):

        if angle == 0:

            return img

        h, w = img.shape[:2]

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)



# ==================== 数据集 ====================

class DRIVEDataset(Dataset):

    def __init__(self, root_dir, train=True, augment=True):

        self.root_dir = root_dir

        self.train = train

        self.augment = augment and train

        self.aug = RandomAugment()

        self.samples = []

        

        if train:

            img_dir = os.path.join(root_dir, 'training', 'images')

            mask_dir = os.path.join(root_dir, 'training', '1st_manual')

        else:

            img_dir = os.path.join(root_dir, 'test', 'images')

            mask_dir = os.path.join(root_dir, 'test', 'mask')

        

        print(f"加载 {'训练' if train else '测试'}集:")

        print(f"  图像路径: {img_dir}")

        print(f"  标注路径: {mask_dir}")

        

        for f in sorted(os.listdir(img_dir)):

            if f.endswith(('.tif', '.ppm', '.png', '.jpg', '.gif')):

                img_path = os.path.join(img_dir, f)

                base_name = f.split('_')[0]

                

                possible_masks = [

                    f"{base_name}_test_mask.gif",

                    f"{base_name}_manual1.gif",

                    f"{base_name}_test.gif",

                    f"{base_name}.gif",

                ]

                

                mask_path = None

                for pm in possible_masks:

                    mp = os.path.join(mask_dir, pm)

                    if os.path.exists(mp):

                        mask_path = mp

                        break

                

                if mask_path:

                    self.samples.append((img_path, mask_path))

        

        print(f"  ✅ 成功加载: {len(self.samples)} 张图像")

        if len(self.samples) == 0:

            raise ValueError("未找到数据")

    

    def preprocess(self, img_path, mask_path):

        img = cv2.imread(img_path)

        if img is None:

            img = np.array(Image.open(img_path).convert('RGB'))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        

        try:

            mask = np.array(Image.open(mask_path).convert('L'))

        except:

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        

        # 数据增强（训练时）

        if self.train and self.augment:

            img, mask = self.aug(img, mask)

        

        # Resize

        img = cv2.resize(img, (512, 512))

        mask = cv2.resize(mask, (512, 512))

        

        # 预处理：绿色通道 + CLAHE + 伽马

        if len(img.shape) == 3:

            green = img[:,:,1]

        else:

            green = img

        

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        enhanced = clahe.apply(green)

        

        gamma = 1.2

        inv_gamma = 1.0 / gamma

        table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype(np.uint8)

        corrected = cv2.LUT(enhanced, table)

        

        img_tensor = torch.from_numpy(corrected.astype(np.float32)/255.0).unsqueeze(0)

        mask = (mask > 128).astype(np.float32)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        

        return img_tensor, mask_tensor

    

    def __len__(self):

        return len(self.samples)

    

    def __getitem__(self, idx):

        img_path, mask_path = self.samples[idx]

        return self.preprocess(img_path, mask_path)



# ==================== 高级损失函数（关键！解决类别不平衡） ====================

class TverskyLoss(nn.Module):

    """

    Tversky Loss: 可控制FP和FN的权重

    alpha高 -> 惩罚FN（漏检），beta高 -> 惩罚FP（误检）

    对于血管分割，我们需要高alpha（不能漏掉血管）

    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):

        super().__init__()

        self.alpha = alpha  # False Negative权重（漏检）

        self.beta = beta    # False Positive权重（误检）

        self.smooth = smooth

    

    def forward(self, pred, target):

        pred = pred.view(-1)

        target = target.view(-1)

        

        TP = (pred * target).sum()

        FP = ((1-target) * pred).sum()

        FN = (target * (1-pred)).sum()

        

        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)

        return 1 - tversky



class FocalTverskyLoss(nn.Module):

    """Focal Tversky: 关注难例"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0):

        super().__init__()

        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

        self.gamma = gamma

    

    def forward(self, pred, target):

        tversky_loss = self.tversky(pred, target)

        return torch.pow(tversky_loss, self.gamma)



class ComboLoss(nn.Module):

    """组合损失：Focal Tversky + BCE（带OHEM）"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, w_ft=0.8, w_bce=0.2):

        super().__init__()

        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

        self.bce = nn.BCELoss(reduction='none')

        self.w_ft = w_ft

        self.w_bce = w_bce

        self.pos_weight = 10.0

    

    def forward(self, pred, target):

        # Focal Tversky（主要损失，解决不平衡）

        ft_loss = self.ft(pred, target)

        

        # 加权BCE（辅助）

        bce_loss = self.bce(pred, target)

        weights = torch.where(target > 0.5, 

                             torch.tensor(self.pos_weight), 

                             torch.tensor(1.0))

        weighted_bce = (bce_loss * weights).mean()

        

        # 额外惩罚：假阴性（漏检血管）- 关键！

        fn_penalty = ((target - pred).clamp(min=0) ** 2).mean() * 5.0

        

        return self.w_ft * ft_loss + self.w_bce * weighted_bce + fn_penalty



# ==================== 指标计算 ====================

def calculate_metrics(pred, target, threshold=0.5):

    pred_bin = (pred > threshold).astype(np.uint8)

    target_bin = (target > threshold).astype(np.uint8)

    

    smooth = 1e-6

    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))

    tn = int(np.sum((pred_bin == 0) & (target_bin == 0)))

    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))

    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))

    

    total = tp + tn + fp + fn

    if total == 0:

        return {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}

    

    dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    sensitivity = (tp + smooth) / (tp + fn + smooth)

    specificity = (tn + smooth) / (tn + fp + smooth)

    

    return {

        'dice': float(dice),

        'iou': float(iou),

        'sensitivity': float(sensitivity),

        'specificity': float(specificity)

    }



# ==================== 训练流程（含Warmup和梯度累积） ====================

class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs=10, total_epochs=150, base_lr=0.001):

        self.optimizer = optimizer

        self.warmup_epochs = warmup_epochs

        self.total_epochs = total_epochs

        self.base_lr = base_lr

    

    def step(self, epoch):

        if epoch < self.warmup_epochs:

            # Warmup阶段：线性增加

            lr = self.base_lr * (epoch + 1) / self.warmup_epochs

        else:

            # Cosine退火

            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        

        for param_group in self.optimizer.param_groups:

            param_group['lr'] = lr

        return lr



def train_model(root_dir, epochs=200, batch_size=4, lr=0.001, device='cpu', patience=30):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    

    # 数据

    train_dataset = DRIVEDataset(root_dir, train=True, augment=True)

    val_dataset = DRIVEDataset(root_dir, train=False, augment=False)

    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    

    # 模型

    model = ImprovedUNet(deep_supervision=False).to(device)

    

    # 损失：使用Focal Tversky（alpha=0.7高惩罚漏检）

    criterion = ComboLoss(alpha=0.7, beta=0.3, gamma=2.0, w_ft=0.8, w_bce=0.2)

    

    # 优化器：AdamW（比Adam更好）

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    

    # 学习率调度（Warmup 10轮 + Cosine）

    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs, base_lr=lr)

    

    # 梯度累积（模拟大batch）

    accumulation_steps = 4  # 每4步更新一次，等效batch_size=16

    

    best_dice = 0.0

    counter = 0

    history = {'train_loss': [], 'val_dice': [], 'val_sens': []}

    

    os.makedirs('checkpoints', exist_ok=True)

    

    print(f"\n开始训练: {epochs} epochs, 等效batch={batch_size*accumulation_steps}, lr={lr}")

    print("="*70)

    print("使用Focal Tversky Loss (alpha=0.7, 高惩罚漏检) + 弹性形变增强")

    print("="*70)

    

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        optimizer.zero_grad()

        

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):

            images = images.to(device)

            masks = masks.to(device)

            

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss = loss / accumulation_steps  # 梯度累积

            loss.backward()

            

            if (batch_idx + 1) % accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                optimizer.zero_grad()

            

            train_loss += loss.item() * accumulation_steps

        

        # 处理最后一个不完整的accumulation

        if len(train_loader) % accumulation_steps != 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            optimizer.zero_grad()

        

        avg_train_loss = train_loss / len(train_loader)

        history['train_loss'].append(avg_train_loss)

        

        # 验证

        model.eval()

        val_metrics = {'dice': [], 'sensitivity': [], 'specificity': []}

        

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)

                masks = masks.to(device)

                

                outputs = model(images)

                outputs_np = outputs.cpu().numpy()

                masks_np = masks.cpu().numpy()

                

                metrics = calculate_metrics(outputs_np[0,0], masks_np[0,0])

                for k in val_metrics:

                    val_metrics[k].append(metrics[k])

        

        avg_val_dice = np.mean(val_metrics['dice'])

        avg_val_sens = np.mean(val_metrics['sensitivity'])

        avg_val_spec = np.mean(val_metrics['specificity'])

        

        history['val_dice'].append(avg_val_dice)

        history['val_sens'].append(avg_val_sens)

        

        # 学习率

        current_lr = scheduler.step(epoch)

        

        print(f"[Epoch {epoch+1:3d}] Loss: {avg_train_loss:.4f} | "

              f"Dice: {avg_val_dice:.4f} | Sens: {avg_val_sens:.4f} | "

              f"Spec: {avg_val_spec:.4f} | LR: {current_lr:.6f}")

        

        # 保存最佳

        if avg_val_dice > best_dice:

            best_dice = avg_val_dice

            counter = 0

            torch.save({

                'epoch': epoch,

                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),

                'dice': best_dice,

                'sensitivity': avg_val_sens,

                'specificity': avg_val_spec,

            }, 'checkpoints/best_model.pth')

            print(f"  >>> 新最佳模型 (Dice: {best_dice:.4f})")

        else:

            counter += 1

            if counter >= patience:

                print(f"\n早停: {patience}轮未提升")

                break

    

    torch.save(model.state_dict(), 'checkpoints/final_model.pth')

    with open('checkpoints/history_v3.json', 'w') as f:

        json.dump(history, f)

    

    print("="*70)

    print(f"训练完成！最佳 Dice: {best_dice:.4f} ({best_dice*100:.1f}%)")

    print(f"目标: 84.0%")

    print("="*70)



if __name__ == "__main__":

    DATASET_PATH = "dataset"

    

    if not os.path.exists(DATASET_PATH):

        print(f"错误: 数据集路径不存在")

        exit(1)

    

    # 40小时足够跑200轮

    train_model(

        root_dir=DATASET_PATH,

        epochs=200,        # 200轮

        batch_size=4,      # 实际等效16（梯度累积）

        lr=0.001,

        device='cpu',      # 你当前是CPU，如果能用GPU改成'cuda'

        patience=40        # 40轮早停耐心

    )