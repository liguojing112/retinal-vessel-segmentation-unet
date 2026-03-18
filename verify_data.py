import os

base = "dataset"  # 如果数据集在当前目录下的dataset文件夹里，保持这样就行

train_imgs = os.listdir(os.path.join(base, "training", "images"))
train_masks = os.listdir(os.path.join(base, "training", "1st_manual"))

print(f"训练图像数量: {len(train_imgs)}")  # 应该输出 20
print(f"训练标注数量: {len(train_masks)}")  # 应该输出 20
print("示例文件:", train_imgs[0] if train_imgs else "无")