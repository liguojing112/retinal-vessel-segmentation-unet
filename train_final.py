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
        self.ms_conv = nn.ModuleList([
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(64, 32, 5, padding=2),
        ])
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

# ==================== 数据增强（针对800张大样本优化） ====================
class RandomAugment:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    
    def elastic_deform(self, img, mask, alpha=80, sigma=8):
        if random.random() > 0.7:
            shape = img.shape[:2]
            dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0,0), sigma) * alpha
            dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0,0), sigma) * alpha
            
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return img, mask
    
    def __call__(self, img, mask):
        # 弹性形变
        img, mask = self.elastic_deform(img, mask)
        
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
        
        # 随机亮度/对比度
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
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

# ==================== 数据集（适配800张PNG） ====================
class FundusDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.aug = RandomAugment() if train else None
        self.samples = []
        
        # 确定路径（现在全是标准结构）
        if train:
            img_dir = os.path.join(root_dir, 'training', 'images')
            mask_dir = os.path.join(root_dir, 'training', 'mask')
        else:
            img_dir = os.path.join(root_dir, 'test', 'images')
            mask_dir = os.path.join(root_dir, 'test', 'mask')
        
        print(f"加载 {'训练' if train else '测试'}集:")
        print(f"  图像路径: {img_dir}")
        print(f"  标注路径: {mask_dir}")
        
        # 读取所有PNG文件（现在全是png，且图像mask同名）
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        for f in img_files:
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, f)  # 同名
            
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                print(f"  ⚠️ 跳过 {f}: 找不到对应mask")
        
        print(f"  ✅ 成功加载: {len(self.samples)} 张图像")
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到数据，请检查路径是否存在: {img_dir}")
    
    def preprocess(self, img_path, mask_path):
        # 读取（现在全是PNG）
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            raise ValueError(f"读取失败: {img_path} 或 {mask_path}")
        
        # 数据增强（训练时）
        if self.train and self.aug:
            img, mask = self.aug(img, mask)
        
        # 预处理：绿色通道 + CLAHE + 伽马校正
        if len(img.shape) == 3:
            green = img[:,:,1]
        else:
            green = img
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(green)
        
        # 伽马校正
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(enhanced, table)
        
        # 归一化
        img_tensor = torch.from_numpy(corrected.astype(np.float32)/255.0).unsqueeze(0)
        
        # 掩膜二值化
        mask = (mask > 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return img_tensor, mask_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return self.preprocess(img_path, mask_path)

# ==================== 损失函数（针对血管分割优化） ====================
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, w_dice=0.7, w_bce=0.3):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice = DiceLoss()
        self.bce = nn.BCELoss(reduction='none')
        self.pos_weight = 10.0  # 血管像素权重
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        
        bce_loss = self.bce(pred, target)
        weights = torch.where(target > 0.5, 
                             torch.tensor(self.pos_weight), 
                             torch.tensor(1.0))
        weighted_bce = (bce_loss * weights).mean()
        
        # 假阴性惩罚（漏检血管）
        fn_penalty = ((target - pred).clamp(min=0) ** 2).mean() * 3.0
        
        return self.w_dice * dice_loss + self.w_bce * weighted_bce + fn_penalty

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
    sensitivity = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else 0.0
    specificity = (tn + smooth) / (tn + fp + smooth) if (tn + fp) > 0 else 0.0
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }

# ==================== 训练流程 ====================
def train_model(root_dir, epochs=150, batch_size=4, lr=0.001, device='cpu', patience=30):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据集路径: {root_dir}")
    
    # 数据集（800张！）
    train_dataset = FundusDataset(root_dir, train=True)
    val_dataset = FundusDataset(root_dir, train=False)
    
    # DataLoader（800张可以加大batch，但CPU保持4）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"\n数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  总计: {len(train_dataset) + len(val_dataset)} 张\n")
    
    # 模型
    model = ImprovedUNet().to(device)
    
    # 损失和优化器
    criterion = ComboLoss(w_dice=0.7, w_bce=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 梯度累积（模拟大batch）
    accumulation_steps = 4
    
    best_dice = 0.0
    counter = 0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': [], 'val_spec': []}
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"开始训练: {epochs} epochs, batch_size={batch_size} (等效{batch_size*accumulation_steps})")
    print("="*70)
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        # 处理剩余梯度
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
            }, 'checkpoints/best_model_800.pth')
            print(f"  >>> 新最佳模型 (Dice: {best_dice:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"\n早停: {patience}轮未提升")
                break
    
    torch.save(model.state_dict(), 'checkpoints/final_model_800.pth')
    with open('checkpoints/history_800.json', 'w') as f:
        json.dump(history, f)
    
    print("="*70)
    print(f"训练完成！最佳 Dice: {best_dice:.4f} ({best_dice*100:.1f}%)")
    print(f"目标: 84.0%，差距: {abs(0.84 - best_dice)*100:.1f}%")
    print("模型保存在: checkpoints/best_model_800.pth")

if __name__ == "__main__":
    DATASET_PATH = "dataset"
    
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集路径 {DATASET_PATH} 不存在！")
        exit(1)
    
    # 开始训练（800张，150轮，足够时间）
    train_model(
        root_dir=DATASET_PATH,
        epochs=150,        # 150轮
        batch_size=4,      # CPU保持4，等效16（梯度累积）
        lr=0.001,
        device='cpu',      # 如果有GPU改成'cuda'
        patience=40        # 40轮早停耐心
    )