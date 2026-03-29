"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

数据完整性快速检查脚本。

用于统计训练图像与标注数量，快速确认数据目录是否匹配。"""





import os



base = "dataset"  # 如果数据集在当前目录下的dataset文件夹里，保持这样就行



train_imgs = os.listdir(os.path.join(base, "training", "images"))

train_masks = os.listdir(os.path.join(base, "training", "1st_manual"))



print(f"训练图像数量: {len(train_imgs)}")  # 应该输出 20

print(f"训练标注数量: {len(train_masks)}")  # 应该输出 20

print("示例文件:", train_imgs[0] if train_imgs else "无")