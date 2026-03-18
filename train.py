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

# ==================== 模型定义（与你之前的一致） ====================
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

# ==================== 数据集类 ====================
class DRIVEDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.samples = []
        
        # 根据你的实际目录修改
        if train:
            img_dir = os.path.join(root_dir, 'training', 'images')
            mask_dir = os.path.join(root_dir, 'training', '1st_manual')  # 训练集用 1st_manual
        else:
            img_dir = os.path.join(root_dir, 'test', 'images')
            mask_dir = os.path.join(root_dir, 'test', 'mask')  # 测试集用 mask（你这里叫 mask）
        
        # 构建文件列表
        for f in sorted(os.listdir(img_dir)):
            if f.endswith(('.tif', '.ppm', '.png')):
                img_path = os.path.join(img_dir, f)
                # DRIVE标注文件命名：01_manual1.gif 对应 01_training.tif
                base_name = f.split('_')[0]
                mask_name = f"{base_name}_manual1.gif"
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
        
        print(f"{'训练' if train else '测试'}集加载完成: {len(self.samples)} 张图像")
    
    def preprocess(self, img_path, mask_path):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            # 尝试PIL（针对gif或其他格式）
            img = np.array(Image.open(img_path).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 读取掩膜（DRIVE通常是gif）
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
        except:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 统一尺寸为512x512
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        
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
        
        # 掩膜二值化（DRIVE中血管为255，背景为0）
        mask = (mask > 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return img_tensor, mask_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img, mask = self.preprocess(img_path, mask_path)
        return img, mask

# ==================== 损失函数 ====================
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
    """复合损失：Dice + 加权交叉熵"""
    def __init__(self, w_dice=0.5, w_bce=0.5):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice = DiceLoss()
        # 加权BCE（血管像素少，给血管更高权重）
        self.bce = nn.BCELoss(reduction='none')
        self.pos_weight = 5.0  # 血管像素权重是背景的5倍
    
    def forward(self, pred, target):
        # Dice Loss
        dice_loss = self.dice(pred, target)
        
        # 加权BCE
        bce_loss = self.bce(pred, target)
        weights = torch.where(target > 0.5, 
                             torch.tensor(self.pos_weight), 
                             torch.tensor(1.0))
        weighted_bce = (bce_loss * weights).mean()
        
        return self.w_dice * dice_loss + self.w_bce * weighted_bce

# ==================== 指标计算 ====================
def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)
    
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    tn = np.sum((pred_bin == 0) & (target_bin == 0))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))
    
    smooth = 1e-5
    dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)  # Recall
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

# ==================== 训练流程 ====================
def train_model(root_dir, epochs=100, batch_size=4, lr=0.001, device='cuda'):
    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集
    train_dataset = DRIVEDataset(root_dir, train=True)
    val_dataset = DRIVEDataset(root_dir, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 模型
    model = ImprovedUNet().to(device)
    
    # 损失和优化器
    criterion = CombinedLoss(w_dice=0.5, w_bce=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练记录
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': []}
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"\n开始训练: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print("="*60)
    
    for epoch in range(epochs):
        # === 训练阶段 ===
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # === 验证阶段 ===
        model.eval()
        val_metrics = {'dice': [], 'sensitivity': [], 'specificity': []}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                outputs_np = outputs.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                # 计算指标
                metrics = calculate_metrics(outputs_np[0,0], masks_np[0,0])
                for k in val_metrics:
                    val_metrics[k].append(metrics[k])
        
        # 计算平均指标
        avg_val_dice = np.mean(val_metrics['dice'])
        avg_val_sens = np.mean(val_metrics['sensitivity'])
        avg_val_spec = np.mean(val_metrics['specificity'])
        
        history['val_dice'].append(avg_val_dice)
        history['val_sens'].append(avg_val_sens)
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        print(f"[Epoch {epoch+1:3d}] Loss: {avg_train_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | Sens: {avg_val_sens:.4f} | "
              f"Spec: {avg_val_spec:.4f} | LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
            }, 'checkpoints/best_model.pth')
            print(f"  >>> 保存最佳模型 (Dice: {best_dice:.4f})")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    
    # 保存训练历史
    with open('checkpoints/history.json', 'w') as f:
        json.dump(history, f)
    
    print("="*60)
    print(f"训练完成！最佳 Dice: {best_dice:.4f}")
    print("模型保存在: checkpoints/best_model.pth")

if __name__ == "__main__":
    # 修改为你的DRIVE数据集路径
    DATASET_PATH = "dataset"  # 例如: "/home/user/DRIVE"
    
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集路径 {DATASET_PATH} 不存在！")
        print("请修改脚本中的 DATASET_PATH 变量指向你的DRIVE文件夹")
        exit(1)
    
    train_model(
        root_dir=DATASET_PATH,
        epochs=10,
        batch_size=4,  # RTX 4090可调大，CPU建议保持4
        lr=0.001,
        device='cuda'  # 没GPU改为 'cpu'，但训练很慢
    )