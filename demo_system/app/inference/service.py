import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import UploadFile

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILES_PER_REQUEST, UPLOAD_DIR
from app.core.errors import InputError, ProcessingError
from app.inference.model_loader import model_loader
from app.inference.postprocess import (
    evaluate_binary_metrics,
    overlay_mask,
    resize_mask,
    threshold_prediction,
)
from app.inference.preprocess import preprocess, robust_imread
from app.utils.image_codec import _to_base64

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.loaded_model = model_loader.load()

    async def process_batch(self, files: list[UploadFile]) -> tuple[dict, int]:
        started = time.perf_counter()
        valid_files = [file for file in files if file and (file.filename or "").strip()]
        if not valid_files:
            raise InputError("没有文件")
        if len(valid_files) > MAX_FILES_PER_REQUEST:
            raise InputError(f"一次最多上传 {MAX_FILES_PER_REQUEST} 张图片")

        results: list[dict] = []
        errors: list[dict] = []

        for upload in valid_files:
            try:
                results.append(await self.process_one(upload))
            except Exception as exc:
                logger.exception("单文件处理失败: %s", upload.filename)
                errors.append({"filename": upload.filename or "", "error": str(exc)})

        if not results:
            return {"error": errors[0]["error"] if errors else "处理失败", "errors": errors}, 500

        payload: dict = {"success": True, "results": results, "errors": errors}
        if len(results) == 1:
            payload.update(
                {
                    "original": results[0]["original"],
                    "mask": results[0]["mask"],
                    "overlay": results[0]["overlay"],
                    "metrics": results[0]["metrics"],
                }
            )

        elapsed = round((time.perf_counter() - started) * 1000, 2)
        logger.info("批量推理完成 files=%s ok=%s err=%s cost_ms=%s", len(valid_files), len(results), len(errors), elapsed)
        return payload, 200

    async def process_one(self, uploaded_file: UploadFile) -> dict:
        original_name = (uploaded_file.filename or "").strip()
        extension = Path(original_name).suffix.lower()
        if not extension or extension not in ALLOWED_EXTENSIONS:
            raise InputError("不支持的格式，请使用 PPM/PGM/PNG/JPG/TIFF")

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

            return {
                "filename": original_name,
                "original": _to_base64(original_rgb),
                "mask": _to_base64(resized_mask),
                "overlay": _to_base64(overlay),
                "metrics": {
                    "dice": round(self.loaded_model.metrics["dice"], 1),
                    "sensitivity": round(self.loaded_model.metrics["sensitivity"], 1),
                    "specificity": round(self.loaded_model.metrics["specificity"], 1),
                },
                "model_info": {"checkpoint": self.loaded_model.checkpoint},
            }
        except InputError:
            raise
        except Exception as exc:
            raise ProcessingError(str(exc)) from exc
        finally:
            if temp_path.exists():
                os.remove(temp_path)

    async def evaluate_one(self, image_file: UploadFile, mask_file: UploadFile) -> tuple[dict, int]:
        infer_result = await self.process_one(image_file)

        mask_name = (mask_file.filename or "").strip()
        ext = Path(mask_name).suffix.lower()
        if not ext or ext not in ALLOWED_EXTENSIONS:
            raise InputError("评估掩码格式不支持")

        mask_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(mask_name).name}"
        mask_path.write_bytes(await mask_file.read())
        try:
            gt_raw = robust_imread(mask_path)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[:, :, 0]
            from io import BytesIO
            from PIL import Image
            import base64

            pred_img = np.array(Image.open(BytesIO(base64.b64decode(infer_result["mask"]))))
            if pred_img.ndim == 3:
                pred_img = pred_img[:, :, 0]
            gt_resized = np.array(
                Image.fromarray(gt_raw.astype(np.uint8)).resize((pred_img.shape[1], pred_img.shape[0]))
            )
            online_metrics = evaluate_binary_metrics(pred_img, gt_resized)
            return {"success": True, "result": infer_result, "online_metrics": online_metrics}, 200
        finally:
            if mask_path.exists():
                os.remove(mask_path)

    def model_info(self) -> dict:
        return {
            "checkpoint": self.loaded_model.checkpoint,
            "metrics": self.loaded_model.metrics,
            "loaded_at": self.loaded_model.loaded_at,
        }


inference_service = InferenceService()
