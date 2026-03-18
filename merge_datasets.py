import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def process_and_copy(src_img, src_mask, dst_img, dst_mask, idx):
    """
    处理并复制一对图像-标注
    统一转换为512x512的PNG格式
    """
    try:
        # 读取图像（支持tif/png/jpg）
        if src_img.endswith('.tif') or src_img.endswith('.tiff'):
            img = cv2.imread(src_img, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(src_img, cv2.IMREAD_COLOR)
        
        if img is None:
            # 尝试PIL读取
            img = np.array(Image.open(src_img).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 读取掩膜（支持gif/png）
        if src_mask.endswith('.gif'):
            mask = np.array(Image.open(src_mask).convert('L'))
        else:
            mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(src_mask).convert('L'))
        
        # 统一尺寸为512x512
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        
        # 确保掩膜是二值图（0或255）
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 保存为PNG（统一格式）
        cv2.imwrite(dst_img, img)
        cv2.imwrite(dst_mask, mask)
        
        return True
    except Exception as e:
        print(f"❌ 处理失败 {src_img}: {e}")
        return False

print("="*60)
print("开始整合数据集：DRIVE(40张) + Kaggle(800张) = 840张")
print("="*60)

# 目标路径（标准结构）
TRAIN_IMG_DIR = "dataset/training/images"
TRAIN_MASK_DIR = "dataset/training/mask"
TEST_IMG_DIR = "dataset/test/images"
TEST_MASK_DIR = "dataset/test/mask"

# 清空并重建目标目录（谨慎！如需保留旧数据请备份）
print("\n1. 清理旧数据...")
for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    ensure_dir(d)

counter = 0

# ==================== 2. 处理DRIVE数据（40张）====================
print("\n2. 整合DRIVE数据集（40张）...")
# 训练集（20张）
drive_train_imgs = sorted([f for f in os.listdir("dataset/training/images") 
                          if f.endswith(('.tif', '.ppm', '.png')) and os.path.exists(f"dataset/training/images/{f}")])
for i, f in enumerate(drive_train_imgs, 1):
    base = f.split('_')[0]
    src_img = f"dataset/training/images/{f}"
    # 尝试找对应的mask（可能是manual1.gif或mask.gif）
    possible_masks = [
        f"dataset/training/1st_manual/{base}_manual1.gif",
        f"dataset/training/mask/{base}_test_mask.gif",
        f"dataset/training/mask/{base}.gif"
    ]
    src_mask = None
    for pm in possible_masks:
        if os.path.exists(pm):
            src_mask = pm
            break
    
    if src_mask:
        dst_img = f"{TRAIN_IMG_DIR}/drive_{i:03d}.png"
        dst_mask = f"{TRAIN_MASK_DIR}/drive_{i:03d}.png"
        if process_and_copy(src_img, src_mask, dst_img, dst_mask, i):
            counter += 1
            print(f"  DRIVE训练集 {i}/20: {f} -> drive_{i:03d}.png")

# 测试集（20张）
drive_test_imgs = sorted([f for f in os.listdir("dataset/test/images") 
                         if f.endswith(('.tif', '.ppm', '.png'))])
for i, f in enumerate(drive_test_imgs, 1):
    base = f.split('_')[0]
    src_img = f"dataset/test/images/{f}"
    possible_masks = [
        f"dataset/test/mask/{base}_test_mask.gif",
        f"dataset/test/mask/{base}_manual1.gif",
        f"dataset/test/1st_manual/{base}_manual1.gif"
    ]
    src_mask = None
    for pm in possible_masks:
        if os.path.exists(pm):
            src_mask = pm
            break
    
    if src_mask:
        dst_img = f"{TEST_IMG_DIR}/drive_{i:03d}.png"
        dst_mask = f"{TEST_MASK_DIR}/drive_{i:03d}.png"
        if process_and_copy(src_img, src_mask, dst_img, dst_mask, i):
            counter += 1
            print(f"  DRIVE测试集 {i}/20: {f} -> drive_{i:03d}.png")

# ==================== 3. 处理Kaggle数据（800张）====================
print("\n3. 整合Kaggle数据集（800张）...")
ARCHIVE_DIR = "dataset/archive (2)"

if os.path.exists(ARCHIVE_DIR):
    # 训练集（600张）
    kaggle_train_dir = f"{ARCHIVE_DIR}/train/Original"
    kaggle_train_mask_dir = f"{ARCHIVE_DIR}/train/Ground truth"
    
    if os.path.exists(kaggle_train_dir):
        train_files = sorted([f for f in os.listdir(kaggle_train_dir) if f.endswith('.png')])
        for i, f in enumerate(train_files, 1):
            src_img = f"{kaggle_train_dir}/{f}"
            src_mask = f"{kaggle_train_mask_dir}/{f}"  # 同名
            
            if os.path.exists(src_mask):
                dst_img = f"{TRAIN_IMG_DIR}/kaggle_{i:03d}.png"
                dst_mask = f"{TRAIN_MASK_DIR}/kaggle_{i:03d}.png"
                if process_and_copy(src_img, src_mask, dst_img, dst_mask, i):
                    counter += 1
                    if i % 100 == 0:
                        print(f"  Kaggle训练集 {i}/{len(train_files)} 完成...")
    
    # 测试集（200张）
    kaggle_test_dir = f"{ARCHIVE_DIR}/test/Original"
    kaggle_test_mask_dir = f"{ARCHIVE_DIR}/test/Ground truth"
    
    if os.path.exists(kaggle_test_dir):
        test_files = sorted([f for f in os.listdir(kaggle_test_dir) if f.endswith('.png')])
        for i, f in enumerate(test_files, 1):
            src_img = f"{kaggle_test_dir}/{f}"
            src_mask = f"{kaggle_test_mask_dir}/{f}"
            
            if os.path.exists(src_mask):
                dst_img = f"{TEST_IMG_DIR}/kaggle_{i:03d}.png"
                dst_mask = f"{TEST_MASK_DIR}/kaggle_{i:03d}.png"
                if process_and_copy(src_img, src_mask, dst_img, dst_mask, i):
                    counter += 1
                    if i % 50 == 0:
                        print(f"  Kaggle测试集 {i}/{len(test_files)} 完成...")

# ==================== 4. 统计结果 ====================
print("\n" + "="*60)
print("整合完成！")
print("="*60)

train_imgs = len([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.png')])
train_masks = len([f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.png')])
test_imgs = len([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')])
test_masks = len([f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.png')])

print(f"\n训练集: {train_imgs} 张图像, {train_masks} 张标注")
print(f"测试集: {test_imgs} 张图像, {test_masks} 张标注")
print(f"总计: {train_imgs + test_imgs} 张图像")

if train_imgs == train_masks and test_imgs == test_masks:
    print("✅ 检查通过：图像与标注数量匹配")
else:
    print("⚠️ 警告：图像与标注数量不匹配，请检查")