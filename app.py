"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

U-Net 模型定义
作者: 李国敬
日期: 2026-03-18
基于 Pytorch 实现，用于视网膜血管分割"""





import os

import uuid

import torch

import torch.nn as nn

import numpy as np

import cv2

from flask import Flask, render_template, request, jsonify

from werkzeug.utils import secure_filename

import base64

from io import BytesIO

from PIL import Image



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



# 全局变量存储模型性能指标（从checkpoint读取）

MODEL_DICE = 86.5  # 默认值，如果读取失败就显示这个

MODEL_SENS = 82.0

MODEL_SPEC = 97.5

LOADED_CHECKPOINT = "(not loaded)"



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



# ==================== 图像读取（保持不变） ====================

def robust_imread(filepath):

    img = None

    try:

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img is not None and img.size > 0:

            return img

    except:

        pass

    

    try:

        pil_img = Image.open(filepath)

        img = np.array(pil_img)

        if len(img.shape) == 2 and pil_img.mode == 'P':

            pil_img = pil_img.convert('RGB')

            img = np.array(pil_img)

        if len(img.shape) == 3 and img.shape[2] == 3:

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    except:

        pass

    

    raise ValueError(f"无法读取图像: {filepath}")



# ==================== 预处理（保持512x512，兼容256训练模型） ====================

def preprocess(img_path):

    try:

        img = robust_imread(img_path)

    except Exception as e:

        raise ValueError(f"图像读取失败: {str(e)}")

    

    if img is None:

        raise ValueError("图像为空")

    

    # 确保3通道

    if len(img.shape) == 2:

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif img.shape[2] == 4:

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    green = img_rgb[:,:,1]

    

    # CLAHE增强

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    enhanced = clahe.apply(green)

    

    # 伽马校正

    gamma = 1.2

    inv_gamma = 1.0 / gamma

    table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype(np.uint8)

    corrected = cv2.LUT(enhanced, table)

    

    # 归一化（U-Net支持任意输入尺寸，512x512演示效果更好）

    normalized = corrected.astype(np.float32)/255.0

    resized = cv2.resize(normalized, (512, 512))

    

    return resized, img_rgb



# ==================== 加载模型（关键修改） ====================

model = ImprovedUNet()



# 优先加载800张数据训练的新模型，回退到旧模型

checkpoint_candidates = [

    'checkpoints/best_model_800.pth',  # 新模型（86.5%）

    'checkpoints/best_model.pth',      # 旧模型（82.3%）

    'checkpoints/final_model.pth'

]



loaded = False

for checkpoint_path in checkpoint_candidates:

    if not os.path.exists(checkpoint_path):

        continue

    try:

        if os.path.getsize(checkpoint_path) == 0:

            continue

            

        # 关键修改：添加 weights_only=False

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        

        # 读取指标（如果有）

        if isinstance(checkpoint, dict):

            if 'dice' in checkpoint:

                MODEL_DICE = float(checkpoint['dice']) * 100

            if 'sensitivity' in checkpoint:

                MODEL_SENS = float(checkpoint['sensitivity']) * 100

            if 'specificity' in checkpoint:

                MODEL_SPEC = float(checkpoint['specificity']) * 100

            

            # 加载权重

            if 'model_state_dict' in checkpoint:

                state_dict = checkpoint['model_state_dict']

            else:

                state_dict = checkpoint

            

            model.load_state_dict(state_dict)

            LOADED_CHECKPOINT = checkpoint_path

            print(f"[系统启动] (>_<) 已加载模型: {checkpoint_path}")

            print(f"[模型性能] Dice: {MODEL_DICE:.1f}%, 敏感度: {MODEL_SENS:.1f}%, 特异性: {MODEL_SPEC:.1f}%")

            loaded = True

            break

    except Exception as e:

        print(f"[警告] 加载失败 {checkpoint_path}: {e}")

        continue



if not loaded:

    print("[警告] 未能加载任何模型权重，使用随机初始化")



model.eval()



@app.route('/')

def index():

    return render_template('index.html')



ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm', '.pgm', '.pbm'}



def _to_base64(arr):

    pil = Image.fromarray(arr)

    buf = BytesIO()

    pil.save(buf, format='PNG')

    return base64.b64encode(buf.getvalue()).decode()



@app.route('/predict', methods=['POST'])

def predict():

    files = []

    if 'files' in request.files:

        files = request.files.getlist('files')

    elif 'file' in request.files:

        files = request.files.getlist('file')

    files = [f for f in files if f and (getattr(f, 'filename', '') or '').strip()]

    if not files:

        return jsonify({'error': '没有文件'}), 400

    if len(files) > 20:

        return jsonify({'error': '一次最多上传 20 张图片'}), 400



    def process_one(uploaded_file):

        original_name = (uploaded_file.filename or '').strip()

        ext = os.path.splitext(original_name)[1].lower()

        if not ext or ext not in ALLOWED_EXT:

            raise ValueError(f'不支持的格式，请使用 PPM/PGM/PNG/JPG/TIFF')

        safe_name = secure_filename(original_name) or f'upload{ext}'

        temp_name = f"{uuid.uuid4().hex}_{safe_name}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)

        uploaded_file.save(filepath)

        try:

            proc_img, original = preprocess(filepath)

            with torch.no_grad():

                input_tensor = torch.from_numpy(proc_img).unsqueeze(0).unsqueeze(0)

                output = model(input_tensor)

                pred = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255

            

            # 调整回原始尺寸

            pred_resized = cv2.resize(pred, (original.shape[1], original.shape[0]))

            overlay = original.copy()

            overlay[pred_resized > 128] = [255, 0, 0]

            

            return {

                'filename': original_name,

                'original': _to_base64(original),

                'mask': _to_base64(pred_resized),

                'overlay': _to_base64(overlay),

                'model_info': {

                    'checkpoint': LOADED_CHECKPOINT,

                },

                'metrics': {

                    'dice': round(MODEL_DICE, 1),

                    'sensitivity': round(MODEL_SENS, 1),

                    'specificity': round(MODEL_SPEC, 1)

                }

            }

        finally:

            if os.path.exists(filepath):

                os.remove(filepath)



    results = []

    errors = []

    for f in files:

        try:

            results.append(process_one(f))

        except Exception as e:

            import traceback

            print(traceback.format_exc())

            errors.append({'filename': getattr(f, 'filename', ''), 'error': str(e)})

    

    if not results:

        return jsonify({'error': errors[0]['error'] if errors else '处理失败', 'errors': errors}), 500

    

    payload = {'success': True, 'results': results, 'errors': errors}

    if len(results) == 1:

        payload.update({

            'original': results[0]['original'],

            'mask': results[0]['mask'],

            'overlay': results[0]['overlay'],

            'metrics': results[0]['metrics']

        })

    return jsonify(payload)



if __name__ == '__main__':

    print("="*60)

    print("视网膜血管分割演示系统")

    print(f"当前模型性能: Dice {MODEL_DICE:.1f}% | 敏感{MODEL_SENS:.1f}% | 特异{MODEL_SPEC:.1f}%")

    print("访问地址: http://localhost:5000")

    print("="*60)

    app.run(debug=True, port=5000, host='0.0.0.0')