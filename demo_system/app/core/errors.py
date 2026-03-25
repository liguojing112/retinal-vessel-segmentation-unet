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
