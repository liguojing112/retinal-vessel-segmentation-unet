"""推理服务逻辑。"""

import os
import traceback
import uuid

import cv2
import numpy as np
import torch
from werkzeug.utils import secure_filename

from .config import ALLOWED_EXTENSIONS, MAX_FILES_PER_REQUEST, PREDICTION_THRESHOLD
from .image_codec import to_base64
from .preprocess import preprocess


class PredictionService:
    def __init__(self, app, model_bundle):
        self.app = app
        self.model_bundle = model_bundle

    def process_one(self, uploaded_file):
        original_name = (uploaded_file.filename or '').strip()
        ext = os.path.splitext(original_name)[1].lower()
        if not ext or ext not in ALLOWED_EXTENSIONS:
            raise ValueError('不支持的格式，请使用 PPM/PGM/PNG/JPG/TIFF')
        safe_name = secure_filename(original_name) or f'upload{ext}'
        temp_name = f"{uuid.uuid4().hex}_{safe_name}"
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], temp_name)
        uploaded_file.save(filepath)
        try:
            proc_img, original = preprocess(filepath)
            with torch.no_grad():
                input_tensor = torch.from_numpy(proc_img).unsqueeze(0).unsqueeze(0)
                output = self.model_bundle.model(input_tensor)
                pred = (output.squeeze().numpy() > PREDICTION_THRESHOLD).astype(np.uint8) * 255

            pred_resized = cv2.resize(pred, (original.shape[1], original.shape[0]))
            overlay = original.copy()
            overlay[pred_resized > 128] = [255, 0, 0]

            return {
                'filename': original_name,
                'original': to_base64(original),
                'mask': to_base64(pred_resized),
                'overlay': to_base64(overlay),
                'model_info': {
                    'checkpoint': self.model_bundle.loaded_checkpoint,
                },
                'metrics': self.model_bundle.metrics(),
            }
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def predict_files(self, request_files):
        files = []
        if 'files' in request_files:
            files = request_files.getlist('files')
        elif 'file' in request_files:
            files = request_files.getlist('file')
        files = [f for f in files if f and (getattr(f, 'filename', '') or '').strip()]
        if not files:
            return {'error': '没有文件'}, 400
        if len(files) > MAX_FILES_PER_REQUEST:
            return {'error': f'一次最多上传 {MAX_FILES_PER_REQUEST} 张图片'}, 400

        results = []
        errors = []
        for uploaded_file in files:
            try:
                results.append(self.process_one(uploaded_file))
            except Exception as e:
                print(traceback.format_exc())
                errors.append({'filename': getattr(uploaded_file, 'filename', ''), 'error': str(e)})

        if not results:
            return {'error': errors[0]['error'] if errors else '处理失败', 'errors': errors}, 500

        payload = {'success': True, 'results': results, 'errors': errors}
        if len(results) == 1:
            payload.update({
                'original': results[0]['original'],
                'mask': results[0]['mask'],
                'overlay': results[0]['overlay'],
                'metrics': results[0]['metrics'],
            })
        return payload, 200
