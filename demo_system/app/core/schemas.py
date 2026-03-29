"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

接口响应类型定义。

使用 TypedDict 描述错误响应与预测错误项字段。"""


from typing import TypedDict


class ErrorResponse(TypedDict):
    success: bool
    error: str
    code: str
    request_id: str


class PredictionError(TypedDict):
    filename: str
    error: str
