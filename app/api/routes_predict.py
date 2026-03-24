from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import TEMPLATES_DIR
from app.core.schemas import PredictionResponse
from app.inference.service import inference_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/predict", response_model=PredictionResponse)
async def predict(files: list[UploadFile] = File(default=[]), file: list[UploadFile] = File(default=[])) -> PredictionResponse:
    selected_files = files or file
    try:
        return await inference_service.process_batch(selected_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc