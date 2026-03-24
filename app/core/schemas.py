from typing import List, Optional

from pydantic import BaseModel, Field


class MetricsResponse(BaseModel):
    dice: float
    sensitivity: float
    specificity: float


class ModelInfoResponse(BaseModel):
    checkpoint: str


class PredictionItemResponse(BaseModel):
    filename: str
    original: str
    mask: str
    overlay: str
    metrics: MetricsResponse
    model_info: ModelInfoResponse


class PredictionErrorResponse(BaseModel):
    filename: Optional[str] = None
    error: str


class PredictionResponse(BaseModel):
    success: bool = True
    results: List[PredictionItemResponse] = Field(default_factory=list)
    errors: List[PredictionErrorResponse] = Field(default_factory=list)