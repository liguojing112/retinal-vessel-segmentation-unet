"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

推理服务单元测试。

本文件聚焦输入校验相关场景：
- 空文件列表；
- 超过最大上传数；
- 非法扩展名。"""


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
