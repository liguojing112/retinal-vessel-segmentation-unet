import json
import matplotlib.pyplot as plt
import os

# 选择要查看的版本（修改这里）
filename = 'checkpoints/history_v3.json'  # 或 history.json, history_v2.json

if not os.path.exists(filename):
    print(f"找不到 {filename}")
    print("现有文件:", [f for f in os.listdir('checkpoints') if f.endswith('.json')])
    exit(1)

with open(filename, 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(12, 4))

# 子图1: Loss曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图2: Dice曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
if 'val_sens' in history:
    plt.plot(epochs, history['val_sens'], 'g--', label='Val Sensitivity', alpha=0.7)
plt.title('Validation Dice Curve')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
print(f"已保存训练曲线图: training_curve.png")
print(f"最佳Dice: {max(history['val_dice']):.4f} (第{history['val_dice'].index(max(history['val_dice']))+1}轮)")
plt.show()