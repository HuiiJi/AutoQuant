"""
AutoQuant - 专业的模型量化工具链条

支持PTQ/QAT、混合精度量化、ONNX导出等功能
参考MQBench框架设计
"""

__version__ = "1.0.0"

# 导出核心模块
from autoquant.core import (
    QuantDtype,
    QScheme,
    round_ste,
    clamp_grad,
    fake_quantize_ste,
    lsq_quantize,
    pact_quantize,
)

# 导出Observer
from autoquant.observer import (
    ObserverBase,
    MinMaxObserver,
    HistogramObserver,
    MovingAverageMinMaxObserver,
)

# 导出FakeQuant
from autoquant.fake_quant import (
    FakeQuantizeBase,
    FakeQuantize,
    FixedFakeQuantize,
    LSQFakeQuantize,
    PACTFakeQuantize,
)

# 导出量化工具
from autoquant.quantization import ModelQuantizer
from autoquant.onnx_export import (
    ONNXExporter,
    InferenceEngine,
    EngineConfig,
    get_engine_config,
    get_qconfig_for_engine,
    get_supported_engines,
    print_engine_info,
    ONNXOptimizer,
    optimize_onnx,
    simplify_with_onnxsim,
)
from autoquant.utils import (
    QConfig,
    get_default_qconfig,
    get_per_channel_qconfig,
    get_per_tensor_qconfig,
    get_lsq_qconfig,
    get_pact_qconfig,
    get_histogram_qconfig,
    MixedPrecisionQuantizer,
    LayerSelector,
    SensitivityAnalyzer,
)

from autoquant.evaluation import (
    QuantizationEvaluator,
    compute_accuracy,
    compute_psnr,
    compute_ssim,
    compute_l1_error,
    compute_l2_error,
    compute_cosine_similarity,
)

from autoquant.special_models import (
    TransformerQuantizer,
    SmoothQuantQuantizer,
    KVCacheQuantizer,
    get_transformer_qconfig,
    get_smoothquant_qconfig,
)

__all__ = [
    # 核心
    "QuantDtype",
    "QScheme",
    "round_ste",
    "clamp_grad",
    "fake_quantize_ste",
    "lsq_quantize",
    "pact_quantize",
    # Observer
    "ObserverBase",
    "MinMaxObserver",
    "HistogramObserver",
    "MovingAverageMinMaxObserver",
    "PercentileObserver",
    "MSEObserver",
    # FakeQuant
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedFakeQuantize",
    "LSQFakeQuantize",
    "PACTFakeQuantize",
    # 工具
    "ModelQuantizer",
    "ONNXExporter",
    "InferenceEngine",
    "EngineConfig",
    "get_engine_config",
    "get_qconfig_for_engine",
    "get_supported_engines",
    "print_engine_info",
    "ONNXOptimizer",
    "optimize_onnx",
    "simplify_with_onnxsim",
    "QConfig",
    "get_default_qconfig",
    "get_per_channel_qconfig",
    "get_per_tensor_qconfig",
    "get_lsq_qconfig",
    "get_pact_qconfig",
    "get_histogram_qconfig",
    "MixedPrecisionQuantizer",
    "LayerSelector",
    "SensitivityAnalyzer",
    # 评估
    "QuantizationEvaluator",
    "compute_accuracy",
    "compute_psnr",
    "compute_ssim",
    "compute_l1_error",
    "compute_l2_error",
    "compute_cosine_similarity",
    # 特殊模型支持
    "TransformerQuantizer",
    "SmoothQuantQuantizer",
    "KVCacheQuantizer",
    "get_transformer_qconfig",
    "get_smoothquant_qconfig",
]

