"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

预测接口路由定义。

包含页面入口、健康检查、模型信息查询、批量预测和单图评估接口。"""


from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import TEMPLATES_DIR
from app.inference.service import inference_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/health")
async def health() -> dict:
    return {"success": True, "status": "ok"}


@router.get("/model_info")
async def model_info() -> dict:
    return {"success": True, "model": inference_service.model_info()}


@router.post("/predict")
async def predict(files: list[UploadFile] = File(default=[]), file: list[UploadFile] = File(default=[])):
    selected_files = files or file
    payload, status_code = await inference_service.process_batch(selected_files)
    return JSONResponse(status_code=status_code, content=payload)


@router.post("/evaluate")
async def evaluate(image: UploadFile = File(...), mask: UploadFile = File(...)):
    payload, status_code = await inference_service.evaluate_one(image, mask)
    return JSONResponse(status_code=status_code, content=payload)
