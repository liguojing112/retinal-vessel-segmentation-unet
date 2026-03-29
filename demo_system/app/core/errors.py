"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

应用自定义异常定义。

统一错误码与 HTTP 状态码，便于接口层做标准化错误返回。"""


class AppError(Exception):
    code = "APP_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InputError(AppError):
    code = "INPUT_ERROR"
    status_code = 400


class ModelError(AppError):
    code = "MODEL_ERROR"
    status_code = 500


class ProcessingError(AppError):
    code = "PROCESSING_ERROR"
    status_code = 500
