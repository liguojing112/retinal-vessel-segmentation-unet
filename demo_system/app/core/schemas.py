from typing import TypedDict


class ErrorResponse(TypedDict):
    success: bool
    error: str
    code: str
    request_id: str


class PredictionError(TypedDict):
    filename: str
    error: str
