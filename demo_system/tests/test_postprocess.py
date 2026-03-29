"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

后处理模块单元测试。

本文件测试目标：
1) 阈值化输出是否严格为二值掩膜（0/255）；
2) 二分类评估指标在完美匹配场景下是否为 100%。"""


import numpy as np

from app.inference.postprocess import evaluate_binary_metrics, threshold_prediction


def test_threshold_prediction_binary_output():
    pred = np.array([[0.2, 0.9], [0.51, 0.49]], dtype=np.float32)
    mask = threshold_prediction(pred, threshold=0.5)
    assert mask.tolist() == [[0, 255], [255, 0]]


def test_evaluate_binary_metrics_perfect_match():
    pred = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    gt = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    metrics = evaluate_binary_metrics(pred, gt)
    assert metrics["dice"] == 100.0
    assert metrics["sensitivity"] == 100.0
    assert metrics["specificity"] == 100.0
