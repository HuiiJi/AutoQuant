"""
AutoQuant - 专业的模型量化工具链条

支持PTQ/QAT、混合精度量化、ONNX导出等功能

Author: jihui
Date: 2026-03-13
"""

__version__ = "1.0.0"

# 导出核心模块
from .core import (
    QuantDtype,
    QScheme,
    round_ste,
    clamp_grad,
    fake_quantize_ste,
    lsq_quantize,
)

# 导出Observer
from .observer import (
    ObserverBase,
    MinMaxObserver,
    HistogramObserver,
)

# 导出FakeQuant
from .fake_quant import (
    FakeQuantizeBase,
    PTQFakeQuantize,
    LSQFakeQuantize,
)

# 导出量化工具
from .quantization.model_quantizer import ModelQuantizer
from .quantization.api import (
    prepare,
    prepare_qat,
    convert,
    calibrate,
)
from .onnx_export import (
    ONNXExporter,
    ONNXOptimizer,
)
from .utils import (
    QConfig,
    get_default_qconfig,
    get_trt_qconfig,
    get_ort_qconfig,
    get_lsq_qconfig,
    SensitivityAnalyzer,
)

from .evaluation import (
    QuantizationEvaluator,
    compute_psnr,
    compute_ssim,
    compute_l1_error,
    compute_l2_error,
    compute_cosine_similarity,
)

from .special_models import (
    NAFNet_flow,
    NAFNet_dgf_4c,
    NAFNet_dgf
)

__all__ = [
    # 核心
    "QuantDtype",
    "QScheme",
    "round_ste",
    "clamp_grad",
    "fake_quantize_ste",
    "lsq_quantize",
    # Observer
    "ObserverBase",
    "MinMaxObserver",
    "HistogramObserver",
    # FakeQuant
    "FakeQuantizeBase",
    "PTQFakeQuantize",
    "LSQFakeQuantize",
    # 工具
    "ModelQuantizer",
    "prepare",
    "prepare_qat",
    "convert",
    "calibrate",
    "ONNXExporter",
    "ONNXOptimizer",
    "QConfig",
    "get_default_qconfig",
    "get_trt_qconfig",
    "get_ort_qconfig",
    "get_lsq_qconfig",
    "SensitivityAnalyzer",
    # 评估
    "QuantizationEvaluator",
    "compute_psnr",
    "compute_ssim",
    "compute_l1_error",
    "compute_l2_error",
    "compute_cosine_similarity",
    # 特殊模型支持
    "NAFNet_flow",
    "NAFNet_dgf_4c",
    "NAFNet_dgf"
]
