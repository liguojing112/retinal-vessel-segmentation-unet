import asyncio
import io

import pytest
from starlette.datastructures import UploadFile

from app.core.config import MAX_FILES_PER_REQUEST
from app.core.errors import InputError
from app.inference.service import InferenceService


def test_process_batch_no_files_raises_input_error():
    service = InferenceService()
    with pytest.raises(InputError):
        asyncio.run(service.process_batch([]))


def test_process_batch_too_many_files_raises_input_error():
    service = InferenceService()
    files = [UploadFile(filename=f"{idx}.png", file=io.BytesIO(b"x")) for idx in range(MAX_FILES_PER_REQUEST + 1)]
    with pytest.raises(InputError):
        asyncio.run(service.process_batch(files))


def test_process_one_rejects_invalid_extension():
    service = InferenceService()
    upload = UploadFile(filename="bad.txt", file=io.BytesIO(b"abc"))
    with pytest.raises(InputError):
        asyncio.run(service.process_one(upload))
