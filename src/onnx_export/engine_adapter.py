"""
推理引擎适配模块
为不同的推理引擎（TensorRT、ONNX Runtime、OpenVINO、MNN等）
提供最佳的QDQ ONNX配置

Author: jihui
Date: 2026-03-13
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from autoquant.core import QuantDtype, QScheme


class InferenceEngine(Enum):
    """支持的推理引擎"""
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    MNN = "mnn"
    TFLITE = "tflite"
    COREML = "coreml"


@dataclass
class EngineConfig:
    """
    引擎配置 - 详细定义每个引擎支持的量化特性

    字段说明:
        engine: 引擎枚举
        activation_dtype: 激活值量化数据类型 (QUINT8/QINT8)
        weight_dtype: 权重量化数据类型 (QUINT8/QINT8)
        activation_qscheme: 激活值量化方案
        weight_qscheme: 权重量化方案
        activation_observer: 激活值observer类型 ("minmax"/"histogram"/"moving_avg")
        weight_observer: 权重observer类型
        supports_per_channel_activation: 是否支持激活值per-channel量化 (大多不支持)
        supports_per_channel_weight: 是否支持权重per-channel量化 (大多支持)
        supports_asymmetric_activation: 是否支持激活值非对称量化
        supports_asymmetric_weight: 是否支持权重非对称量化
        recommended_qat_method: 推荐的QAT方法 (如果有)
        note: 额外说明注释
    """
    engine: InferenceEngine
    activation_dtype: QuantDtype
    weight_dtype: QuantDtype
    activation_qscheme: QScheme
    weight_qscheme: QScheme
    activation_observer: str
    weight_observer: str
    supports_per_channel_activation: bool
    supports_per_channel_weight: bool
    supports_asymmetric_activation: bool
    supports_asymmetric_weight: bool
    recommended_qat_method: Optional[str] = None
    note: Optional[str] = None


# 各引擎的最佳配置 - 详细验证过的专业配置
ENGINE_CONFIGS: Dict[InferenceEngine, EngineConfig] = {

    InferenceEngine.TENSORRT: EngineConfig(
        engine=InferenceEngine.TENSORRT,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        activation_observer="histogram",  # Histogram量化精度更好
        weight_observer="minmax",
        supports_per_channel_activation=False,  # TRT不支持Activation per-channel
        supports_per_channel_weight=True,        # TRT支持Weight per-channel
        supports_asymmetric_activation=True,     # 支持Activation非对称
        supports_asymmetric_weight=False,         # 只支持Weight对称 (-128~127)
        recommended_qat_method="lsq",
        note="TensorRT: 推荐 Weight=PER_CHANNEL_SYMMETRIC, Activation=PER_TENSOR_AFFINE"
    ),

    InferenceEngine.ONNXRUNTIME: EngineConfig(
        engine=InferenceEngine.ONNXRUNTIME,
        activation_dtype=QuantDtype.QINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="histogram",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,  # ORT支持Weight非对称
        recommended_qat_method="lsq",
        note="ONNX Runtime: 支持非对称量化，推荐 HistogramObserver"
    ),

    InferenceEngine.OPENVINO: EngineConfig(
        engine=InferenceEngine.OPENVINO,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        # OpenVINO 最佳实践：
        # - 类似ORT，MinMaxObserver 速度快
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
        note="OpenVINO: 推荐 MinMaxObserver (速度快)"
    ),

    InferenceEngine.MNN: EngineConfig(
        engine=InferenceEngine.MNN,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        # MNN 最佳实践：
        # - 推荐对称量化 PER_TENSOR/PER_CHANNEL_SYMMETRIC
        # - MovingAverageMinMaxObserver 适合动态场景
        activation_qscheme=QScheme.PER_TENSOR_SYMMETRIC,
        weight_qscheme=QScheme.PER_CHANNEL_SYMMETRIC,
        activation_observer="moving_avg",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=False,  # MNN偏好对称量化
        supports_asymmetric_weight=False,
        recommended_qat_method="lsq",
        note="MNN: 推荐对称量化 PER_TENSOR/PER_CHANNEL_SYMMETRIC"
    ),

    InferenceEngine.TFLITE: EngineConfig(
        engine=InferenceEngine.TFLITE,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
        note="TFLite: 类似 ORT/OpenVINO"
    ),

    InferenceEngine.COREML: EngineConfig(
        engine=InferenceEngine.COREML,
        activation_dtype=QuantDtype.QUINT8,
        weight_dtype=QuantDtype.QINT8,
        activation_qscheme=QScheme.PER_TENSOR_AFFINE,
        weight_qscheme=QScheme.PER_CHANNEL_AFFINE,
        activation_observer="minmax",
        weight_observer="minmax",
        supports_per_channel_activation=False,
        supports_per_channel_weight=True,
        supports_asymmetric_activation=True,
        supports_asymmetric_weight=True,
        recommended_qat_method=None,
        note="CoreML: Apple平台，类似 ORT"
    ),
}


def get_engine_config(engine: str) -> EngineConfig:
    """
    获取指定推理引擎的配置

    Args:
        engine: 引擎名称，支持 'tensorrt', 'onnxruntime', 'openvino', 'mnn'

    Returns:
        EngineConfig对象
    """
    try:
        engine_enum = InferenceEngine(engine.lower())
        return ENGINE_CONFIGS[engine_enum]
    except ValueError:
        raise ValueError(f"不支持的推理引擎: {engine}. 支持的引擎: {[e.value for e in InferenceEngine]}")


def get_qconfig_for_engine(engine: str, use_qat: bool = False, verbose: bool = True) -> 'QConfig':
    """
    获取指定推理引擎的最佳QConfig

    Args:
        engine: 引擎名称 (tensorrt/onnxruntime/openvino/mnn/...)
        use_qat: 是否使用QAT配置（默认False，使用PTQ配置）
        verbose: 是否打印详细的量化方案信息（默认True）

    Returns:
        QConfig对象
    """
    from autoquant.utils import (
        get_default_qconfig,
        get_lsq_qconfig,
    )

    config = get_engine_config(engine)

    # 打印详细的量化方案信息
    if verbose:
        print("\n" + "=" * 80)
        print(f"📋 量化方案配置 - 引擎: {engine.upper()} {'(QAT模式)' if use_qat else '(PTQ模式)'}")
        print("=" * 80)
        _print_single_engine_info(config)
        print("=" * 80 + "\n")

    # 根据推荐选择QAT方法（仅在use_qat=True时使用）
    if use_qat and config.recommended_qat_method == "lsq":
        return get_lsq_qconfig(
            activation_dtype=config.activation_dtype,
            weight_dtype=config.weight_dtype,
            activation_qscheme=config.activation_qscheme,
            weight_qscheme=config.weight_qscheme,
        )
    else:
        # PTQ 默认都使用 get_default_qconfig
        return get_default_qconfig(
            activation_dtype=config.activation_dtype,
            weight_dtype=config.weight_dtype,
            activation_qscheme=config.activation_qscheme,
            weight_qscheme=config.weight_qscheme,
            activation_observer_type=config.activation_observer,
            weight_observer_type=config.weight_observer,
        )


def get_supported_engines() -> list:
    """获取所有支持的推理引擎列表"""
    return [e.value for e in InferenceEngine]


def print_engine_info(engine: Optional[str] = None):
    """
    打印推理引擎信息

    Args:
        engine: 可选，指定引擎名称，不指定则打印所有引擎信息
    """
    if engine:
        config = get_engine_config(engine)
        _print_single_engine_info(config)
    else:
        for config in ENGINE_CONFIGS.values():
            _print_single_engine_info(config)
            print("-" * 80)


def _print_single_engine_info(config: EngineConfig):
    """打印单个引擎的详细信息"""
    engine_name = config.engine.value.title().replace('_', ' ')
    print(f"\n{'=' * 80}")
    print(f"推理引擎: {engine_name}")
    print(f"{'=' * 80}")
    print(f"  激活值类型: {config.activation_dtype.name}")
    print(f"  权重类型: {config.weight_dtype.name}")
    print(f"  激活值量化方案: {config.activation_qscheme.name}")
    print(f"  权重量化方案: {config.weight_qscheme.name}")
    print(f"  激活值Observer: {config.activation_observer}")
    print(f"  权重Observer: {config.weight_observer}")
