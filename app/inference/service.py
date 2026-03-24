import os
import uuid
from pathlib import Path

import torch
from fastapi import UploadFile

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILES_PER_REQUEST, UPLOAD_DIR
from app.core.schemas import PredictionErrorResponse, PredictionItemResponse, PredictionResponse
from app.inference.model_loader import model_loader
from app.inference.postprocess import overlay_mask, resize_mask, threshold_prediction
from app.inference.preprocess import preprocess
from app.utils.image_codec import _to_base64


class InferenceService:
    def __init__(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.loaded_model = model_loader.load()

    async def process_batch(self, files: list[UploadFile]) -> PredictionResponse:
        valid_files = [file for file in files if file and (file.filename or "").strip()]
        if not valid_files:
            raise ValueError("没有文件")
        if len(valid_files) > MAX_FILES_PER_REQUEST:
            raise ValueError(f"一次最多上传 {MAX_FILES_PER_REQUEST} 张图片")

        results: list[PredictionItemResponse] = []
        errors: list[PredictionErrorResponse] = []

        for upload in valid_files:
            try:
                results.append(await self.process_one(upload))
            except Exception as exc:
                errors.append(PredictionErrorResponse(filename=upload.filename, error=str(exc)))

        return PredictionResponse(success=True, results=results, errors=errors)

    async def process_one(self, uploaded_file: UploadFile) -> PredictionItemResponse:
        original_name = (uploaded_file.filename or "").strip()
        extension = Path(original_name).suffix.lower()
        if not extension or extension not in ALLOWED_EXTENSIONS:
            raise ValueError("不支持的格式，请使用 PPM/PGM/PBM/PNG/JPG/TIFF")

        temp_name = f"{uuid.uuid4().hex}_{Path(original_name).name}"
        temp_path = UPLOAD_DIR / temp_name

        content = await uploaded_file.read()
        temp_path.write_bytes(content)

        try:
            processed_image, original_rgb = preprocess(temp_path)
            with torch.no_grad():
                input_tensor = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0)
                output = self.loaded_model.model(input_tensor)
                prediction = output.squeeze().cpu().numpy()

            mask = threshold_prediction(prediction)
            resized_mask = resize_mask(mask, original_rgb.shape)
            overlay = overlay_mask(original_rgb, resized_mask)

            return PredictionItemResponse(
                filename=original_name,
                original=_to_base64(original_rgb),
                mask=_to_base64(resized_mask),
                overlay=_to_base64(overlay),
                metrics=self.loaded_model.metrics,
                model_info={"checkpoint": self.loaded_model.checkpoint},
            )
        finally:
            if temp_path.exists():
                os.remove(temp_path)


inference_service = InferenceService() 